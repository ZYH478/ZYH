#!/usr/bin/env bash
set -euo pipefail

ROOT="${YOLO26_EXP_ROOT:-/root/autodl-tmp/neu-det-yolo26}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd "$ROOT"

python -u install_gsconv_modules.py
python -u install_msdgs_module.py

# 数据与模型 gate 可重复执行；解压有 marker，不会重复展开大文件。
python -u prepare_generalization_datasets.py
python -u train_generalization_yolo26_msdgs.py --build-check

exec python -u train_generalization_yolo26_msdgs.py \
  --epochs 250 --batch 32 --imgsz 640 --seed 0
