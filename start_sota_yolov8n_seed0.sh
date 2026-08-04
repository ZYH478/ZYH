#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/neu-det-yolo26
LOG="$ROOT/sota_yolov8n_seed0_e250.log"

cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26

if pgrep -f '[t]rain_sota_yolov8n_seed0.py' >/dev/null; then
    echo "ALREADY_RUNNING"
    pgrep -af '[t]rain_sota_yolov8n_seed0.py'
    exit 0
fi

setsid python -u train_sota_yolov8n_seed0.py >"$LOG" 2>&1 </dev/null &
pid=$!
echo "$pid" > runs_sota_yolov8n_seed0_e250/launcher.pid
echo "STARTED pid=$pid log=$LOG"
