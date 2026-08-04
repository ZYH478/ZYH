#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /root/autodl-tmp/neu-det-yolo26
python install_yolo26_exp_modules.py
export YOLO_CONFIG_DIR=/root/autodl-tmp/neu-det-yolo26/.yolo_config
export MPLCONFIGDIR=/root/autodl-tmp/neu-det-yolo26/.mpl_config
export TORCH_HOME=/root/autodl-tmp/neu-det-yolo26/.torch_cache
export XDG_CACHE_HOME=/root/autodl-tmp/neu-det-yolo26/.cache
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$XDG_CACHE_HOME"
nohup python -u train_yolo26_module_sweep.py --stage stage1 --epochs 250 --batch 32 --imgsz 640 --seed 0 > module_sweep_e250.log 2>&1 &
echo $! > module_sweep_e250.pid
echo "STARTED module_sweep_e250 pid=$(cat module_sweep_e250.pid) log=/root/autodl-tmp/neu-det-yolo26/module_sweep_e250.log"
