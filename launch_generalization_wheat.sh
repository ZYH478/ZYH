#!/usr/bin/env bash
set -euo pipefail
ROOT="${YOLO26_EXP_ROOT:-/root/autodl-tmp/neu-det-yolo26}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd "$ROOT"
python -u install_gsconv_modules.py
python -u install_msdgs_module.py
python -u prepare_generalization_wheat.py --skip-extract
python -u train_generalization_wheat.py --build-check
exec python -u train_generalization_wheat.py --epochs 250 --batch 32 --imgsz 640 --seed 0
