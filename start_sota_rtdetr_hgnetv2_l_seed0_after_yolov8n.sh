#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/neu-det-yolo26
PROJECT="$ROOT/runs_sota_rtdetr_hgnetv2_l_seed0_e250"
LOG="$ROOT/sota_rtdetr_hgnetv2_l_seed0_e250.log"

cd "$ROOT"
mkdir -p "$PROJECT"
if pgrep -f '[t]rain_sota_rtdetr_hgnetv2_l_seed0.py' >/dev/null; then
  echo "ALREADY_RUNNING"
  exit 0
fi
printf '{\n  "status": "queued",\n  "phase": "wait_for_gpu",\n  "model": "RT-DETR-HGNetv2-L",\n  "wait_for": ["train_sota_yolov8n_seed0.py", "train_sota_slf_yolo_seed0.py"]\n}\n' > "$PROJECT/status.json"

while pgrep -f '[t]rain_sota_yolov8n_seed0.py' >/dev/null || pgrep -f '[t]rain_sota_slf_yolo_seed0.py' >/dev/null; do
  echo "[$(date '+%F %T')] WAITING existing SOTA job still owns GPU" >> "$LOG"
  sleep 60
done

sleep 30
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
exec python -u "$ROOT/train_sota_rtdetr_hgnetv2_l_seed0.py" >> "$LOG" 2>&1