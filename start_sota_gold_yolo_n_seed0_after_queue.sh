#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/neu-det-yolo26
PROJECT="$ROOT/runs_sota_gold_yolo_n_seed0_e250"
LOG="$ROOT/gold_yolo_n_seed0_e250.log"

cd "$ROOT"
mkdir -p "$PROJECT"
if pgrep -f '[t]rain_sota_gold_yolo_n_seed0.py' >/dev/null; then
  echo "ALREADY_RUNNING"
  exit 0
fi
cat > "$PROJECT/status.json" <<JSON
{
  "status": "queued",
  "phase": "wait_for_gpu",
  "model": "Gold-YOLO-n",
  "wait_for": [
    "train_sota_rtdetr_hgnetv2_l_seed0.py",
    "start_sota_slf_yolo_seeds1_3_after_current.sh",
    "train_sota_slf_yolo_seed1.py",
    "train_sota_slf_yolo_seed2.py",
    "train_sota_slf_yolo_seed3.py",
    "train_sota_yolov8n_multiseed.py"
  ]
}
JSON

while pgrep -f '[t]rain_sota_rtdetr_hgnetv2_l_seed0.py' >/dev/null \
   || pgrep -f '[s]tart_sota_slf_yolo_seeds1_3_after_current.sh --queue-worker' >/dev/null \
   || pgrep -f '[t]rain_sota_slf_yolo_seed[123]\.py' >/dev/null \
   || pgrep -f '[t]rain_sota_yolov8n_multiseed.py' >/dev/null; do
  echo "[$(date '+%F %T')] WAITING RT-DETR/SLF-YOLO/YOLOv8n multiseed before Gold-YOLO-n" >> "$ROOT/gold_yolo_n_seed0_wait.log"
  sleep 60
done

sleep 60
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lightyolo
exec python -u "$ROOT/train_sota_gold_yolo_n_seed0.py" >> "$LOG" 2>&1
