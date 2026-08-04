#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/neu-det-yolo26
PROJECT="$ROOT/runs_sota_slf_yolo_seeds1_3_e250"
LOG="$ROOT/slf_yolo_seeds1_3_e250.log"
SCRIPT="$ROOT/start_sota_slf_yolo_seeds1_3_after_current.sh"
SEEDS=(1 2 3)
WAIT_PATTERNS=(
  '[t]rain_sota_rtdetr_hgnetv2_l_seed0.py'
  '[s]tart_sota_rtdetr_hgnetv2_l_seed0_after_yolov8n.sh'
)

write_status() {
  local status="$1"
  local phase="$2"
  local current_seed="${3:-null}"
  local extra="${4:-}"
  mkdir -p "$PROJECT"
  cat > "$PROJECT/status.json" <<JSON
{
  "status": "$status",
  "phase": "$phase",
  "model": "SLF-YOLO",
  "seeds": [1, 2, 3],
  "datasets": ["neudet", "aluminum", "pcb"],
  "current_seed": $current_seed,
  "wait_for": ["train_sota_rtdetr_hgnetv2_l_seed0.py", "start_sota_rtdetr_hgnetv2_l_seed0_after_yolov8n.sh"],
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

cleanup_label_caches() {
  rm -f \
    "$ROOT/dataset/train/labels.cache" \
    "$ROOT/dataset/valid/labels.cache" \
    "$ROOT/dataset/test/labels.cache" \
    "$ROOT/generalization_aluminum_pcb/raw/aluminum_yolo26/labels/train.cache" \
    "$ROOT/generalization_aluminum_pcb/raw/aluminum_yolo26/labels/val.cache" \
    "$ROOT/generalization_aluminum_pcb/raw/aluminum_yolo26/labels/test.cache" \
    "$ROOT/generalization_aluminum_pcb/prepared/pcb/labels/train.cache" \
    "$ROOT/generalization_aluminum_pcb/prepared/pcb/labels/val.cache" \
    "$ROOT/generalization_aluminum_pcb/prepared/pcb/labels/test.cache" || true
}

queue_worker() {
  cd "$ROOT"
  write_status "queued" "wait_for_gpu"
  echo "[$(date '+%F %T')] QUEUE_WORKER_START SLF-YOLO seeds=1,2,3" >> "$LOG"
  while running_blockers >> "$LOG" 2>&1; do
    write_status "queued" "wait_for_gpu"
    echo "[$(date '+%F %T')] WAITING current RT-DETR job before SLF-YOLO multiseed" >> "$LOG"
    sleep 60
  done
  sleep 30
  if pgrep -f '[t]rain_sota_slf_yolo_seed[123]\.py' >/dev/null; then
    echo "[$(date '+%F %T')] SLF_MULTI_ALREADY_RUNNING" >> "$LOG"
    exit 0
  fi
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate lightyolo
  export PYTHONPATH="$ROOT/SLF-YOLO/SLF-YOLO:${PYTHONPATH:-}"
  export YOLO_CONFIG_DIR="$ROOT/.yolo_config_slf"
  export MPLCONFIGDIR="$ROOT/.mpl_config_slf"
  export TORCH_HOME="$ROOT/.torch_cache"
  for seed in "${SEEDS[@]}"; do
    write_status "running" "train_eval" "$seed"
    cleanup_label_caches
    echo "[$(date '+%F %T')] SLF_SEED_START seed=$seed script=$ROOT/train_sota_slf_yolo_seed${seed}.py" >> "$LOG"
    python -u "$ROOT/train_sota_slf_yolo_seed${seed}.py" >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] SLF_SEED_DONE seed=$seed" >> "$LOG"
  done
  write_status "done" "done" "null"
  echo "[$(date '+%F %T')] SLF_MULTI_DONE" >> "$LOG"
}

cd "$ROOT"
mkdir -p "$PROJECT"
if [[ "${1:-}" == "--queue-worker" ]]; then
  queue_worker
  exit $?
fi
if pgrep -f '[s]tart_sota_slf_yolo_seeds1_3_after_current.sh --queue-worker' >/dev/null; then
  echo "ALREADY_QUEUED"
  pgrep -af '[s]tart_sota_slf_yolo_seeds1_3_after_current.sh --queue-worker'
  exit 0
fi
if pgrep -f '[t]rain_sota_slf_yolo_seed[123]\.py' >/dev/null; then
  echo "ALREADY_RUNNING"
  pgrep -af '[t]rain_sota_slf_yolo_seed[123]\.py'
  exit 0
fi
write_status "queued" "wait_for_gpu"
setsid bash "$SCRIPT" --queue-worker >> "$LOG" 2>&1 </dev/null &
pid=$!
echo "$pid" > "$PROJECT/launcher.pid"
echo "QUEUED pid=$pid log=$LOG status=$PROJECT/status.json"
