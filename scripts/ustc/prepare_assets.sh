#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/scc/pb22020481/projects/Gated-StreamV2V}"
CONDA_ROOT="${CONDA_ROOT:-/public/app/miniconda3/py312_24.4.0-0}"
ENV_DIR="${ENV_DIR:-/home/scc/pb22020481/conda-envs/gated-streamv2v}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

export HF_HOME="${HF_HOME:-/home/scc/pb22020481/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export GSV2V_MODEL_ID="${GSV2V_MODEL_ID:-runwayml/stable-diffusion-v1-5}"
export GSV2V_LCM_LORA_ID="${GSV2V_LCM_LORA_ID:-latent-consistency/lcm-lora-sdv1-5}"
export GSV2V_CLIP_MODEL_ID="${GSV2V_CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
export GSV2V_RAFT_WEIGHTS="${GSV2V_RAFT_WEIGHTS:-$PROJECT_DIR/data/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"

LORA_FOLDER_URL="${LORA_FOLDER_URL:-https://drive.google.com/drive/folders/1D7g-dnCQnjjogTPX-B3fttgdrp9nKeKw}"
LORA_DIR="${LORA_DIR:-$PROJECT_DIR/vid2vid/lora_weights}"
RAFT_URL="${RAFT_URL:-https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"
LORA_DOWNLOAD_TIMEOUT="${LORA_DOWNLOAD_TIMEOUT:-1800}"

export PROJECT_DIR LORA_FOLDER_URL LORA_DIR RAFT_URL LORA_DOWNLOAD_TIMEOUT

echo "PROJECT_DIR=$PROJECT_DIR"
echo "ENV_DIR=$ENV_DIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "GSV2V_MODEL_ID=$GSV2V_MODEL_ID"
echo "GSV2V_LCM_LORA_ID=$GSV2V_LCM_LORA_ID"
echo "GSV2V_CLIP_MODEL_ID=$GSV2V_CLIP_MODEL_ID"
echo "GSV2V_RAFT_WEIGHTS=$GSV2V_RAFT_WEIGHTS"
echo "LORA_DIR=$LORA_DIR"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

python -m pip install -i "$PYPI_INDEX_URL" -U gdown

mkdir -p "$PROJECT_DIR/data/checkpoints"
mkdir -p "$LORA_DIR"

python - <<'PY'
import os

from huggingface_hub import snapshot_download

downloads = [
    (
        "Stable Diffusion",
        os.environ["GSV2V_MODEL_ID"],
        [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/*",
            "unet/*",
            "vae/*",
            "feature_extractor/*",
            "safety_checker/*",
        ],
    ),
    (
        "LCM-LoRA",
        os.environ["GSV2V_LCM_LORA_ID"],
        ["*.json", "*.safetensors", "*.bin", "*.md"],
    ),
    (
        "CLIP",
        os.environ["GSV2V_CLIP_MODEL_ID"],
        [
            "*.json",
            "*.txt",
            "merges.txt",
            "vocab.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "model.safetensors",
        ],
    ),
]

for label, repo_id, allow_patterns in downloads:
    print(f"Prefetch {label}: {repo_id}")
    snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)
PY

if [[ -s "$GSV2V_RAFT_WEIGHTS" ]]; then
  echo "RAFT weights already exist: $GSV2V_RAFT_WEIGHTS"
else
  echo "Downloading RAFT weights: $RAFT_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 5 "$RAFT_URL" -o "$GSV2V_RAFT_WEIGHTS"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$GSV2V_RAFT_WEIGHTS" "$RAFT_URL"
  else
    python - <<'PY'
import os
import urllib.request

urllib.request.urlretrieve(os.environ["RAFT_URL"], os.environ["GSV2V_RAFT_WEIGHTS"])
PY
  fi
  test -s "$GSV2V_RAFT_WEIGHTS"
  echo "Saved RAFT weights: $GSV2V_RAFT_WEIGHTS"
fi

timeout "$LORA_DOWNLOAD_TIMEOUT" python - <<'PY'
import os
import sys
import gdown

out_dir = os.environ["LORA_DIR"]
downloads = [
    (
        "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
        "1_-kEVFw_LnV1J2Nho6nZt4PUbymamypK",
    ),
    (
        "low_poly.safetensors",
        "1ZClfRljzKmxsU1Jj5OMwIuXQcnA1DwO9",
    ),
    (
        "Claymation.safetensors",
        "1GvPCbrPqJYj0_nRppSc2UD_1eRME-1tG",
    ),
    (
        "doodle.safetensors",
        "12ZMOy8CMzwB32RHSmff0h2TJC3lFDBmW",
    ),
    (
        "Sketch_offcolor.safetensors",
        "1NIBujegFMvFdjCW0vdrmD6fbNFKNROE4",
    ),
    (
        "bichu-v0612.safetensors",
        "1fmS3fGeja0RM8YbZtbKw20fjXNzHrnxz",
    ),
]

print("Downloading LoRA files into:", out_dir)
for filename, file_id in downloads:
    output = os.path.join(out_dir, filename)
    if os.path.exists(output) and os.path.getsize(output) > 1_000_000:
        print("LoRA already exists:", output)
        continue
    print("Downloading LoRA:", filename)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url=url, output=output, quiet=False, use_cookies=False)
    except TypeError:
        gdown.download(url=url, output=output, quiet=False)
    except Exception as exc:
        print(f"LoRA download failed for {filename}: {exc}", file=sys.stderr)
PY

python - <<'PY'
import os
import sys
from pathlib import Path

lora_dir = Path(os.environ["LORA_DIR"])
required = [
    "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
    "low_poly.safetensors",
    "Claymation.safetensors",
    "doodle.safetensors",
    "Sketch_offcolor.safetensors",
    "bichu-v0612.safetensors",
]
missing = [
    name
    for name in required
    if not (lora_dir / name).exists() or (lora_dir / name).stat().st_size <= 1_000_000
]
if missing:
    print("Missing LoRA weights:", file=sys.stderr)
    for name in missing:
        print("  -", lora_dir / name, file=sys.stderr)
    print("The Google Drive download may require manual access or a Civitai fallback.", file=sys.stderr)
    raise SystemExit(1)
print("All expected LoRA weights are present.")
print("Assets are ready.")
PY
