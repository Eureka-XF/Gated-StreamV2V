#!/usr/bin/env bash
#SBATCH --job-name=gsv2v-single
#SBATCH --partition=Students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/scc/pb22020481/projects/Gated-StreamV2V/slurm-%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/scc/pb22020481/projects/Gated-StreamV2V}"
CONDA_ROOT="${CONDA_ROOT:-/public/app/miniconda3/py312_24.4.0-0}"
ENV_DIR="${ENV_DIR:-/home/scc/pb22020481/conda-envs/gated-streamv2v}"

export HF_HOME="${HF_HOME:-/home/scc/pb22020481/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export GSV2V_MODEL_ID="${GSV2V_MODEL_ID:-runwayml/stable-diffusion-v1-5}"
export GSV2V_LCM_LORA_ID="${GSV2V_LCM_LORA_ID:-latent-consistency/lcm-lora-sdv1-5}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "ENV_DIR=$ENV_DIR"
echo "GSV2V_MODEL_ID=$GSV2V_MODEL_ID"
echo "GSV2V_LCM_LORA_ID=$GSV2V_LCM_LORA_ID"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

cd "$PROJECT_DIR/vid2vid"

python main.py \
  --input ./demo_selfie/tennis.mp4 \
  --prompt "Ukiyo-e Art - a man holding a tennis racket on a tennis court" \
  --model_id "$GSV2V_MODEL_ID" \
  --lcm_lora_id "$GSV2V_LCM_LORA_ID" \
  --output_dir ./outputs_ustc/single_similarity \
  --video_name tennis_similarity_ustc \
  --cached_attn_style similarity \
  --cache_interval 1 \
  --diffusion_steps 4 \
  --noise_strength 0.4 \
  --use_feature_injection True

python - <<'PY'
import cv2
from pathlib import Path

path = Path("outputs_ustc/single_similarity/tennis_similarity_ustc.mp4")
print("output:", path.resolve())
print("exists:", path.exists())
print("size:", path.stat().st_size if path.exists() else 0)
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit("Output video is missing or empty")

cap = cv2.VideoCapture(str(path))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print("frames:", frames)
print("fps:", fps)
print("width:", width)
print("height:", height)
if frames <= 0:
    raise SystemExit("Output video has no frames")
PY
