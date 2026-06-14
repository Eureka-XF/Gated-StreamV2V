#!/usr/bin/env bash
#SBATCH --job-name=gsv2v-metrics
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
SET_FILE_PATH="${SET_FILE_PATH:-$PROJECT_DIR/tools/user_study_upload/eval.json}"
REF_VIDEO_DIR="${REF_VIDEO_DIR:-$PROJECT_DIR/vid2vid/source_video}"
EDIT_VIDEO_DIR="${EDIT_VIDEO_DIR:-$PROJECT_DIR/vid2vid/output_ustc_paper/$METHOD}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_DIR/reports/ustc_paper_repro}"

export HF_HOME="${HF_HOME:-/home/scc/pb22020481/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export GSV2V_CLIP_MODEL_ID="${GSV2V_CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
export GSV2V_RAFT_WEIGHTS="${GSV2V_RAFT_WEIGHTS:-$PROJECT_DIR/data/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "METHOD=$METHOD"
echo "SET_FILE_PATH=$SET_FILE_PATH"
echo "REF_VIDEO_DIR=$REF_VIDEO_DIR"
echo "EDIT_VIDEO_DIR=$EDIT_VIDEO_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "GSV2V_CLIP_MODEL_ID=$GSV2V_CLIP_MODEL_ID"
echo "GSV2V_RAFT_WEIGHTS=$GSV2V_RAFT_WEIGHTS"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

mkdir -p "$REPORT_DIR/metrics/clip" "$REPORT_DIR/metrics/warp"

cd "$PROJECT_DIR/tools"

python clip_score.py \
  --device cuda \
  --method_version "$METHOD" \
  --set_file_path "$SET_FILE_PATH" \
  --edit_video_dir "$EDIT_VIDEO_DIR" \
  --clip_model_id_or_path "$GSV2V_CLIP_MODEL_ID" \
  --output_log_dir "$REPORT_DIR/metrics/clip"

python warp_error.py \
  --device cuda \
  --method_version "$METHOD" \
  --set_file_path "$SET_FILE_PATH" \
  --ref_video_dir "$REF_VIDEO_DIR" \
  --edit_video_dir "$EDIT_VIDEO_DIR" \
  --raft_weights "$GSV2V_RAFT_WEIGHTS" \
  --output_log_dir "$REPORT_DIR/metrics/warp"
