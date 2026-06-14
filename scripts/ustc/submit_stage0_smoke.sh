#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/scc/pb22020481/projects/Gated-StreamV2V}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_DIR/reports/ustc_paper_repro}"
SMOKE_VID_NAME="${SMOKE_VID_NAME:-tennis_ukiyoe_0}"
SMOKE_SET_FILE="$REPORT_DIR/stage0_${SMOKE_VID_NAME}.json"

mkdir -p "$REPORT_DIR"

python - <<'PY'
import json
import os
from pathlib import Path

project_dir = Path(os.environ["PROJECT_DIR"])
smoke_vid_name = os.environ["SMOKE_VID_NAME"]
out_path = Path(os.environ["SMOKE_SET_FILE"])
source = project_dir / "tools/user_study_upload/eval.json"
data = json.loads(source.read_text(encoding="utf-8"))
selected = [item for item in data if item.get("vid_name") == smoke_vid_name]
if len(selected) != 1:
    raise SystemExit(f"Expected exactly one item for {smoke_vid_name}, found {len(selected)}")
out_path.write_text(json.dumps(selected, sort_keys=True, indent=4), encoding="utf-8")
print("Wrote", out_path)
PY

METHODS=(
  streamv2v_origin
  gated_similarity_reverse
  gated_similarity_forward
  confidence_gate
)

for METHOD in "${METHODS[@]}"; do
  BATCH_JOB_ID=$(
    sbatch --parsable \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",REPORT_DIR="$REPORT_DIR",METHOD="$METHOD",ONLY_VID_NAME="$SMOKE_VID_NAME" \
      "$PROJECT_DIR/scripts/ustc/run_paper_batch.sbatch"
  )
  echo "Submitted batch $METHOD: $BATCH_JOB_ID"

  METRIC_JOB_ID=$(
    sbatch --parsable \
      --dependency=afterok:"$BATCH_JOB_ID" \
      --export=ALL,PROJECT_DIR="$PROJECT_DIR",REPORT_DIR="$REPORT_DIR",METHOD="$METHOD",SET_FILE_PATH="$SMOKE_SET_FILE" \
      "$PROJECT_DIR/scripts/ustc/run_paper_metrics.sbatch"
  )
  echo "Submitted metrics $METHOD: $METRIC_JOB_ID"
done

echo "Stage 0 submitted. Check with: squeue -u $USER"
