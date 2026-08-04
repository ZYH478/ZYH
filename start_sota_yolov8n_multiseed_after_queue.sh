#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/neu-det-yolo26
PROJECT="$ROOT/runs_sota_yolov8n_multiseed_e250"
LOG="$ROOT/sota_yolov8n_multiseed_e250.log"
SCRIPT="$ROOT/start_sota_yolov8n_multiseed_after_queue.sh"
TRAIN="$ROOT/train_sota_yolov8n_multiseed.py"
WAIT_PATTERNS=(
  '[t]rain_sota_rtdetr_hgnetv2_l_seed0.py'
  '[s]tart_sota_rtdetr_hgnetv2_l_seed0_after_yolov8n.sh'
  '[t]rain_sota_slf_yolo_seed0.py'
  '[s]tart_sota_slf_yolo_seed0_after_yolov8n.sh'
  '[s]tart_sota_slf_yolo_seeds1_3_after_current.sh --queue-worker'
  '[t]rain_sota_slf_yolo_seed[123]\.py'
  '[s]tart_sota_gold_yolo_n_seed0_after_queue.sh'
  '[t]rain_sota_gold_yolo_n_seed0.py'
  '[s]tart_sota_yolo11n_seed0_after_current.sh --queue-worker'
  '[t]rain_sota_yolo11n_seed0.py'
)
WAIT_NAMES=(
  'train_sota_rtdetr_hgnetv2_l_seed0.py'
  'start_sota_rtdetr_hgnetv2_l_seed0_after_yolov8n.sh'
  'train_sota_slf_yolo_seed0.py'
  'start_sota_slf_yolo_seed0_after_yolov8n.sh'
  'start_sota_slf_yolo_seeds1_3_after_current.sh --queue-worker'
  'train_sota_slf_yolo_seed1.py'
  'train_sota_slf_yolo_seed2.py'
  'train_sota_slf_yolo_seed3.py'
  'start_sota_gold_yolo_n_seed0_after_queue.sh'
  'train_sota_gold_yolo_n_seed0.py'
  'start_sota_yolo11n_seed0_after_current.sh --queue-worker'
  'train_sota_yolo11n_seed0.py'
)

write_status() {
  local status="$1"
  local phase="$2"
  local extra="${3:-}"
  mkdir -p "$PROJECT"
  local wait_json
  wait_json=$(printf '"%s",' "${WAIT_NAMES[@]}")
  wait_json="[${wait_json%,}]"
  cat > "$PROJECT/status.json" <<JSON
{
  "status": "$status",
  "phase": "$phase",
  "model": "yolov8n",
  "seeds": [1, 2, 3],
  "datasets": ["neudet", "aluminum", "pcb"],
  "seed0_source": "$ROOT/runs_sota_yolov8n_seed0_e250",
  "wait_for": $wait_json,
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
  while true; do
    if running_blockers >> "$LOG" 2>&1; then
      write_status "queued" "wait_for_gpu"
      echo "[$(date '+%F %T')] WAITING existing SOTA queue/job still owns GPU" >> "$LOG"
      sleep 60
      continue
    fi
    sleep 30
    if running_blockers >> "$LOG" 2>&1; then
      write_status "queued" "race_guard"
      echo "[$(date '+%F %T')] RACE_GUARD blocker appeared during grace period" >> "$LOG"
      sleep 30
      continue
    fi
    break
  done
  if pgrep -f '[t]rain_sota_yolov8n_multiseed.py' >/dev/null; then
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

if pgrep -f '[t]rain_sota_yolov8n_multiseed.py' >/dev/null; then
  echo "ALREADY_RUNNING"
  pgrep -af '[t]rain_sota_yolov8n_multiseed.py'
  exit 0
fi
if pgrep -f '[s]tart_sota_yolov8n_multiseed_after_queue.sh --queue-worker' >/dev/null; then
  echo "ALREADY_QUEUED"
  pgrep -af '[s]tart_sota_yolov8n_multiseed_after_queue.sh --queue-worker'
  exit 0
fi

write_status "queued" "wait_for_gpu"
setsid bash "$SCRIPT" --queue-worker >> "$LOG" 2>&1 </dev/null &
pid=$!
echo "$pid" > "$PROJECT/launcher.pid"
echo "QUEUED pid=$pid log=$LOG status=$PROJECT/status.json"
