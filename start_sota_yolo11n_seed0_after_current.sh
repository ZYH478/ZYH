#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/neu-det-yolo26
PROJECT="$ROOT/runs_sota_yolo11n_seed0_e250"
LOG="$ROOT/sota_yolo11n_seed0_e250.log"
SCRIPT="$ROOT/start_sota_yolo11n_seed0_after_current.sh"
TRAIN="$ROOT/train_sota_yolo11n_seed0.py"
WAIT_PATTERNS=(
  '[t]rain_sota_rtdetr_hgnetv2_l_seed0.py'
  '[t]rain_sota_yolov8n_seed0.py'
  '[t]rain_sota_slf_yolo_seed0.py'
  '[s]tart_sota_slf_yolo_seeds1_3_after_current.sh --queue-worker'
  '[t]rain_sota_slf_yolo_seed[123]\.py'
  '[s]tart_sota_gold_yolo_n_seed0_after_queue.sh'
  '[t]rain_sota_gold_yolo_n_seed0.py'
)

write_status() {
  local status="$1"
  local phase="$2"
  local extra="${3:-}"
  mkdir -p "$PROJECT"
  cat > "$PROJECT/status.json" <<JSON
{
  "status": "$status",
  "phase": "$phase",
  "model": "yolo11n",
  "seed": 0,
  "datasets": ["neudet", "aluminum", "pcb"],
  "wait_for": ["train_sota_rtdetr_hgnetv2_l_seed0.py", "train_sota_yolov8n_seed0.py", "train_sota_slf_yolo_seed0.py", "start_sota_slf_yolo_seeds1_3_after_current.sh --queue-worker", "train_sota_slf_yolo_seed1.py", "train_sota_slf_yolo_seed2.py", "train_sota_slf_yolo_seed3.py", "start_sota_gold_yolo_n_seed0_after_queue.sh", "train_sota_gold_yolo_n_seed0.py"],
  "updated_at": "$(date '+%F %T')"$extra
}
JSON
}

running_blockers() {
  local found=1
  for pattern in "${WAIT_PATTERNS[@]}"; do
    if pgrep -f "$pattern" >/dev/null; then
      pgrep -af "$pattern"
      found=0
    fi
  done
  return "$found"
}

queue_worker() {
  cd "$ROOT"
  write_status "queued" "wait_for_gpu"
  echo "[$(date '+%F %T')] QUEUE_WORKER_START script=$TRAIN" >> "$LOG"
  while running_blockers >> "$LOG" 2>&1; do
    write_status "queued" "wait_for_gpu"
    echo "[$(date '+%F %T')] WAITING existing SOTA job still owns GPU" >> "$LOG"
    sleep 60
  done
  sleep 30
  if pgrep -f '[t]rain_sota_yolo11n_seed0.py' >/dev/null; then
    echo "[$(date '+%F %T')] TRAIN_ALREADY_RUNNING" >> "$LOG"
    exit 0
  fi
  write_status "running" "launching" ',
  "launcher_pid": '"$$"
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate yolo26
  echo "[$(date '+%F %T')] LAUNCH_TRAIN python=$TRAIN" >> "$LOG"
  exec python -u "$TRAIN" >> "$LOG" 2>&1
}

cd "$ROOT"
mkdir -p "$PROJECT"

if [[ "${1:-}" == "--queue-worker" ]]; then
  queue_worker
  exit $?
fi

if pgrep -f '[t]rain_sota_yolo11n_seed0.py' >/dev/null; then
  echo "ALREADY_RUNNING"
  pgrep -af '[t]rain_sota_yolo11n_seed0.py'
  exit 0
fi

if pgrep -f '[s]tart_sota_yolo11n_seed0_after_current.sh --queue-worker' >/dev/null; then
  echo "ALREADY_QUEUED"
  pgrep -af '[s]tart_sota_yolo11n_seed0_after_current.sh --queue-worker'
  exit 0
fi

write_status "queued" "wait_for_gpu"
setsid bash "$SCRIPT" --queue-worker >> "$LOG" 2>&1 </dev/null &
pid=$!
echo "$pid" > "$PROJECT/launcher.pid"
echo "QUEUED pid=$pid log=$LOG status=$PROJECT/status.json"
