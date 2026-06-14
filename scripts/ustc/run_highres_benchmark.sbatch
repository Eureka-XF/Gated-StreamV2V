#!/usr/bin/env bash
#SBATCH --job-name=gsv2v-highres
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
REPORT_DIR="${REPORT_DIR:-$PROJECT_DIR/reports/ustc_paper_repro}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/vid2vid/output_ustc_paper/highres}"
SCALES="${SCALES:-1 2 3}"
METHODS="${METHODS:-streamv2v_origin gated_similarity_reverse}"

export HF_HOME="${HF_HOME:-/home/scc/pb22020481/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export GSV2V_MODEL_ID="${GSV2V_MODEL_ID:-runwayml/stable-diffusion-v1-5}"
export GSV2V_LCM_LORA_ID="${GSV2V_LCM_LORA_ID:-latent-consistency/lcm-lora-sdv1-5}"

PROMPT="${PROMPT:-Ukiyo-e Art - a man holding a tennis racket on a tennis court}"
INPUT_VIDEO="${INPUT_VIDEO:-./source_video/tennis.mp4}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "ENV_DIR=$ENV_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SCALES=$SCALES"
echo "METHODS=$METHODS"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

mkdir -p "$REPORT_DIR/highres" "$OUTPUT_ROOT"
cd "$PROJECT_DIR/vid2vid"

for METHOD in $METHODS; do
  case "$METHOD" in
    streamv2v_origin)
      CACHED_ATTN_STYLE="origin"
      REVERSE_TAG="True"
      ;;
    gated_similarity_reverse)
      CACHED_ATTN_STYLE="similarity"
      REVERSE_TAG="True"
      ;;
    *)
      echo "Unknown highres METHOD=$METHOD" >&2
      exit 2
      ;;
  esac

  for SCALE in $SCALES; do
    VIDEO_NAME="${METHOD}_scale${SCALE}_tennis"
    OUT_DIR="$OUTPUT_ROOT/$METHOD/scale${SCALE}"
    BENCHMARK_JSON="$REPORT_DIR/highres/${METHOD}_scale${SCALE}.json"

    echo "Running highres benchmark: method=$METHOD scale=$SCALE"
    set +e
    python main.py \
      --input "$INPUT_VIDEO" \
      --prompt "$PROMPT" \
      --model_id "$GSV2V_MODEL_ID" \
      --lcm_lora_id "$GSV2V_LCM_LORA_ID" \
      --output_dir "$OUT_DIR" \
      --video_name "$VIDEO_NAME" \
      --cached_attn_style "$CACHED_ATTN_STYLE" \
      --reverse_tag "$REVERSE_TAG" \
      --cache_interval 1 \
      --diffusion_steps 4 \
      --noise_strength 0.4 \
      --use_feature_injection True \
      --scale "$SCALE" \
      --benchmark_json "$BENCHMARK_JSON"
    STATUS=$?
    set -e

    if [ "$STATUS" -ne 0 ]; then
      export BENCHMARK_JSON METHOD SCALE STATUS CACHED_ATTN_STYLE REVERSE_TAG
      python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["BENCHMARK_JSON"])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "status": "failed",
    "returncode": int(os.environ["STATUS"]),
    "method": os.environ["METHOD"],
    "scale": float(os.environ["SCALE"]),
    "cached_attn_style": os.environ["CACHED_ATTN_STYLE"],
    "reverse_tag": os.environ["REVERSE_TAG"] == "True",
}
path.write_text(json.dumps(payload, sort_keys=True, indent=4), encoding="utf-8")
print("Wrote failure benchmark:", path)
PY
    fi
  done
done
