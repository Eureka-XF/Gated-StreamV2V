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
LORA_DOWNLOAD_TIMEOUT="${LORA_DOWNLOAD_TIMEOUT:-120}"

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

download_lora() {
  local filename="$1"
  local url="$2"
  local required="$3"
  local output="$LORA_DIR/$filename"
  local tmp="$output.tmp"

  if [[ -s "$output" ]] && [[ "$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output")" -gt 1000000 ]]; then
    echo "LoRA already exists: $output"
    return 0
  fi

  echo "Downloading LoRA: $filename"
  rm -f "$tmp"
  if command -v curl >/dev/null 2>&1; then
    if timeout "$LORA_DOWNLOAD_TIMEOUT" curl -L --fail --retry 2 --retry-delay 5 -o "$tmp" "$url"; then
      mv "$tmp" "$output"
      echo "Saved LoRA: $output"
      return 0
    fi
  elif command -v wget >/dev/null 2>&1; then
    if timeout "$LORA_DOWNLOAD_TIMEOUT" wget -O "$tmp" "$url"; then
      mv "$tmp" "$output"
      echo "Saved LoRA: $output"
      return 0
    fi
  fi
  rm -f "$tmp"

  if [[ "$required" == "required" ]]; then
    echo "Required LoRA download failed: $filename" >&2
    return 1
  fi
  echo "Optional LoRA unavailable: $filename" >&2
  return 0
}

echo "Preparing LoRA files in: $LORA_DIR"
download_lora \
  "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors" \
  "${PIXELART_LORA_URL:-$HF_ENDPOINT/artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5/resolve/main/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors}" \
  required
download_lora \
  "Sketch_offcolor.safetensors" \
  "${SKETCH_LORA_URL:-https://civitai.com/api/download/models/174421}" \
  required
download_lora \
  "bichu-v0612.safetensors" \
  "${OILPAINTING_LORA_URL:-https://civitai.com/api/download/models/94277}" \
  required

python - <<'PY'
import os
import sys
from pathlib import Path

lora_dir = Path(os.environ["LORA_DIR"])
required = [
    "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
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
    print("Upload the missing files to vid2vid/lora_weights or provide reachable *_LORA_URL overrides.", file=sys.stderr)
    raise SystemExit(1)
optional = [
    "low_poly.safetensors",
    "Claymation.safetensors",
    "doodle.safetensors",
]
for name in optional:
    path = lora_dir / name
    if not path.exists():
        print("Optional LoRA not present:", path)
print("All paper-reproduction LoRA weights are present.")
print("Assets are ready.")
PY
