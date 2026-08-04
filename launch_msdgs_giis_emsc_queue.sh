#!/usr/bin/env bash
set -euo pipefail
ROOT="${YOLO26_EXP_ROOT:-/root/autodl-tmp/neu-det-yolo26}"
cd "$ROOT"
mkdir -p runs_msdgs_literature_queue

if [[ -f runs_msdgs_literature_queue/queue.pid ]]; then
  old_pid="$(cat runs_msdgs_literature_queue/queue.pid || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "QUEUE_ALREADY_RUNNING pid=$old_pid"
    exit 3
  fi
fi

active="$(pgrep -af 'python.*(train_|ultralytics|yolo).*' || true)"
if [[ -n "$active" ]]; then
  echo "REFUSE_CONCURRENT_TRAINING"
  echo "$active"
  exit 4
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
nohup python -u run_msdgs_giis_emsc_queue.py \
  > runs_msdgs_literature_queue/queue.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > runs_msdgs_literature_queue/queue.pid
echo "QUEUE_STARTED pid=$pid log=$ROOT/runs_msdgs_literature_queue/queue.log"
