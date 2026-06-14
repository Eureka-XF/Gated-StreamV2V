#!/usr/bin/env bash
#SBATCH --job-name=gsv2v-batch
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
METHOD="${METHOD:-gated_similarity_reverse}"
LIMIT="${LIMIT:-0}"
ONLY_VID_NAME="${ONLY_VID_NAME:-}"
JSON_FILE="${JSON_FILE:-$PROJECT_DIR/vid2vid/source_video/eval.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/vid2vid/output_ustc_paper}"

export HF_HOME="${HF_HOME:-/home/scc/pb22020481/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export GSV2V_MODEL_ID="${GSV2V_MODEL_ID:-runwayml/stable-diffusion-v1-5}"
export GSV2V_LCM_LORA_ID="${GSV2V_LCM_LORA_ID:-latent-consistency/lcm-lora-sdv1-5}"

case "$METHOD" in
  streamv2v_origin)
    CACHED_ATTN_STYLE="origin"
    REVERSE_TAG="True"
    ;;
  gated_similarity_reverse)
    CACHED_ATTN_STYLE="similarity"
    REVERSE_TAG="True"
    ;;
  gated_similarity_forward)
    CACHED_ATTN_STYLE="similarity"
    REVERSE_TAG="False"
    ;;
  confidence_gate)
    CACHED_ATTN_STYLE="confidence"
    REVERSE_TAG="True"
    ;;
  *)
    echo "Unknown METHOD=$METHOD" >&2
    echo "Use one of: streamv2v_origin, gated_similarity_reverse, gated_similarity_forward, confidence_gate" >&2
    exit 2
    ;;
esac

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "ENV_DIR=$ENV_DIR"
echo "METHOD=$METHOD"
echo "CACHED_ATTN_STYLE=$CACHED_ATTN_STYLE"
echo "REVERSE_TAG=$REVERSE_TAG"
echo "JSON_FILE=$JSON_FILE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "LIMIT=$LIMIT"
echo "ONLY_VID_NAME=$ONLY_VID_NAME"

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

ARGS=(
  --json_file "$JSON_FILE"
  --output_dir "$OUTPUT_ROOT/$METHOD"
  --model_id "$GSV2V_MODEL_ID"
  --lcm_lora_id "$GSV2V_LCM_LORA_ID"
  --noise_strength "0.4"
  --diffusion_steps "4"
  --cache_interval "1"
  --seed "2"
  --cached_attn_style "$CACHED_ATTN_STYLE"
  --reverse_tag "$REVERSE_TAG"
  --use_attn_concat "True"
  --use_feature_injection "True"
  --feature_similarity_threshold "0.98"
  --feature_injection_strength "0.5"
  --ttt_lr "1.0"
  --skip_existing "True"
  --limit "$LIMIT"
)

if [ -n "$ONLY_VID_NAME" ]; then
  ARGS+=(--only_vid_name "$ONLY_VID_NAME")
fi

python batch_eval.py "${ARGS[@]}"
