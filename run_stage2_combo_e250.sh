#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26

cd /root/autodl-tmp/neu-det-yolo26
python install_yolo26_exp_modules.py
python -u train_yolo26_stage2_combo.py \
  --wait-stage1 \
  --poll-seconds 300 \
  --max-wait-hours 72 \
  --max-variants 10 \
  --epochs 250 \
  --batch 32 \
  --imgsz 640 \
  --seed 0
