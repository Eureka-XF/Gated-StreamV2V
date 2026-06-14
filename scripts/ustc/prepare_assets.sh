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

for label, repo_id in [
    ("Stable Diffusion", os.environ["GSV2V_MODEL_ID"]),
    ("LCM-LoRA", os.environ["GSV2V_LCM_LORA_ID"]),
    ("CLIP", os.environ["GSV2V_CLIP_MODEL_ID"]),
]:
    print(f"Prefetch {label}: {repo_id}")
    snapshot_download(repo_id=repo_id)
PY

python - <<'PY'
import os
import torch
from torchvision.models.optical_flow import Raft_Large_Weights

out_path = os.environ["GSV2V_RAFT_WEIGHTS"]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
    print("RAFT weights already exist:", out_path)
else:
    print("Downloading torchvision RAFT Large C_T_SKHT_V2 weights")
    weights = Raft_Large_Weights.C_T_SKHT_V2
    state_dict = weights.get_state_dict(progress=True)
    torch.save(state_dict, out_path)
    print("Saved RAFT weights:", out_path)
PY

python - <<'PY'
import os
import sys
import gdown

url = os.environ["LORA_FOLDER_URL"]
out_dir = os.environ["LORA_DIR"]
print("Downloading LoRA folder:", url)
try:
    gdown.download_folder(url=url, output=out_dir, quiet=False, use_cookies=False)
except TypeError:
    gdown.download_folder(url=url, output=out_dir, quiet=False)
except Exception as exc:
    print("LoRA download failed:", exc, file=sys.stderr)
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
missing = [name for name in required if not (lora_dir / name).exists()]
if missing:
    print("Missing LoRA weights:", file=sys.stderr)
    for name in missing:
        print("  -", lora_dir / name, file=sys.stderr)
    print("The Google Drive download may require manual access or a Civitai fallback.", file=sys.stderr)
    raise SystemExit(1)
print("All expected LoRA weights are present.")
print("Assets are ready.")
PY
