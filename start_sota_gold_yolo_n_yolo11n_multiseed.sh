#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/neu-det-yolo26"
PROJECT="$ROOT/runs_sota_gold_yolo_n_yolo11n_multiseed_e250"
STATUS="$PROJECT/status.json"
LOG="$ROOT/sota_gold_yolo_n_yolo11n_multiseed_e250.log"
PY="/root/miniconda3/envs/yolo26/bin/python"
SCRIPT="$ROOT/train_sota_gold_yolo_n_yolo11n_multiseed.py"
SELF="start_sota_gold_yolo_n_yolo11n_multiseed.sh"
mkdir -p "$PROJECT"

if [[ "${1:-}" != "--queue-worker" ]]; then
  nohup setsid bash "$0" --queue-worker > "$LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PROJECT/launcher.pid"
  echo "QUEUED pid=$pid log=$LOG status=$STATUS"
  exit 0
fi

write_status() {
  local phase="$1"
  local blockers_json="$2"
  cat > "$STATUS" <<JSON
{
  "status": "queued",
  "phase": "$phase",
  "model": "Gold-YOLO-n + yolo11n",
  "seeds": [0, 1, 2, 3],
  "datasets": ["neudet", "aluminum", "pcb"],
  "wait_for": $blockers_json,
  "updated_at": "$(date '+%F %T')",
  "log": "$LOG"
}
JSON
}

find_blockers() {
  ps -eo pid=,cmd= | awk -v self="$SELF" '
    /start_sota_|train_sota_/ && $0 !~ self && $0 !~ /awk/ {print $1":"substr($0, index($0,$2))}
  '
}

while true; do
  mapfile -t blockers < <(find_blockers || true)
  if (( ${#blockers[@]} == 0 )); then
    sleep 30
    mapfile -t blockers2 < <(find_blockers || true)
    if (( ${#blockers2[@]} == 0 )); then
      break
    fi
    blockers=("${blockers2[@]}")
  fi
  blockers_json=$(printf '%s\n' "${blockers[@]}" | "$PY" -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin if x.strip()], ensure_ascii=False))')
  write_status "wait_for_gpu" "$blockers_json"
  echo "[$(date '+%F %T')] WAITING blockers=$blockers_json"
  sleep 60
done

write_status "launching" "[]"
echo "[$(date '+%F %T')] START Gold-YOLO-n/yolo11n multiseed"
cd "$ROOT"
exec "$PY" "$SCRIPT"
