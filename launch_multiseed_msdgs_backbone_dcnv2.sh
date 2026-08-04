#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/neu-det-yolo26
LOG="$ROOT/msdgs_backbone_dcnv2_multiseed_e250.log"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd "$ROOT"
if pgrep -af 'train_multiseed_msdgs_backbone_dcnv2.py' | grep -v pgrep >/dev/null; then
  echo "ALREADY_RUNNING"
  pgrep -af 'train_multiseed_msdgs_backbone_dcnv2.py'
  exit 0
fi
setsid -f bash -lc "source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26 && cd '$ROOT' && exec python -u train_multiseed_msdgs_backbone_dcnv2.py --epochs 250 --imgsz 640 --batch 32 --seeds 1 2 3 >> '$LOG' 2>&1"
sleep 2
pgrep -af 'train_multiseed_msdgs_backbone_dcnv2.py' || true
echo "LOG=$LOG"