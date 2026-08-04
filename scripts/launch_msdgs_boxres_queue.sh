#!/usr/bin/env bash
set -euo pipefail
ROOT="${YOLO26_EXP_ROOT:-/root/autodl-tmp/neu-det-yolo26}"
PY="${YOLO26_PYTHON:-/root/miniconda3/envs/yolo26/bin/python}"
QUEUE_DIR="$ROOT/runs_msdgs_boxres_queue"
LOG="$QUEUE_DIR/queue.log"
mkdir -p "$QUEUE_DIR"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "GPU_BUSY_REFUSE_START"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 2
fi
cd "$ROOT"
setsid nohup "$PY" scripts/train_msdgs_boxres_queue.py >"$LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$QUEUE_DIR/launcher.pid"
echo "MSDGS_BOXRES_QUEUE_STARTED pid=$pid log=$LOG status=$QUEUE_DIR/status.json"
