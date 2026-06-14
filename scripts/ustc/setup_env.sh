#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/scc/pb22020481/projects/Gated-StreamV2V}"
CONDA_ROOT="${CONDA_ROOT:-/public/app/miniconda3/py312_24.4.0-0}"
ENV_DIR="${ENV_DIR:-/home/scc/pb22020481/conda-envs/gated-streamv2v}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONDA_ROOT=$CONDA_ROOT"
echo "ENV_DIR=$ENV_DIR"
echo "PYPI_INDEX_URL=$PYPI_INDEX_URL"

if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  echo "Cannot find conda at $CONDA_ROOT/bin/conda" >&2
  exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if [ ! -d "$ENV_DIR" ]; then
  conda create -y -p "$ENV_DIR" python=3.10
fi

conda activate "$ENV_DIR"
python -m pip install -i "$PYPI_INDEX_URL" -U pip wheel
python -m pip install -i "$PYPI_INDEX_URL" "setuptools==69.5.1"

python -m pip install \
  torch==2.1.1 \
  torchvision==0.16.1 \
  xformers==0.0.23 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url "$PYPI_INDEX_URL"

python -m pip install -i "$PYPI_INDEX_URL" \
  accelerate==0.21.0 \
  av \
  diffusers==0.27.0 \
  einops==0.6.1 \
  fire==0.7.1 \
  huggingface-hub==0.25.2 \
  numpy==1.26.4 \
  opencv-python==4.9.0.80 \
  peft==0.13.0 \
  pillow==10.4.0 \
  safetensors \
  scipy \
  tokenizers==0.15.1 \
  tqdm \
  transformers==4.37.2

cd "$PROJECT_DIR"
python -m pip install -e .

python - <<'PY'
import cv2
import diffusers
import torch
import transformers
import xformers

print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available on login node:", torch.cuda.is_available())
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("xformers:", xformers.__version__)
print("opencv:", cv2.__version__)
PY

echo "Environment is ready: $ENV_DIR"
