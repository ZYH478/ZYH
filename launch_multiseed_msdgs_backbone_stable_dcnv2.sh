#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/neu-det-yolo26
PY=/root/miniconda3/envs/yolo26/bin/python
LOG="$ROOT/msdgs_backbone_stable_dcnv2_e250.log"
cd "$ROOT"
if pgrep -af 'train_multiseed_msdgs_backbone_stable_dcnv2.py' | grep -v pgrep >/dev/null; then
  echo "ALREADY_RUNNING"
  pgrep -af 'train_multiseed_msdgs_backbone_stable_dcnv2.py'
  exit 0
fi
setsid -f bash -lc "cd '$ROOT' && exec '$PY' -u train_multiseed_msdgs_backbone_stable_dcnv2.py --epochs 250 --imgsz 640 --batch 32 --seeds 0 3 >> '$LOG' 2>&1"
sleep 3
pgrep -af 'train_multiseed_msdgs_backbone_stable_dcnv2.py' || true
echo "LOG=$LOG"
