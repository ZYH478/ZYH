#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26

cd /root/autodl-tmp/neu-det-yolo26
python -m py_compile train_yolo26_module_sweep.py train_yolo26_stage2_combo.py
python install_yolo26_exp_modules.py

if ps -eo pid,args | grep '[t]rain_yolo26_module_sweep.py' >/dev/null; then
  echo "STAGE1_ALREADY_RUNNING"
  ps -eo pid,stat,etime,args | grep '[t]rain_yolo26_module_sweep.py'
else
  nohup python -u train_yolo26_module_sweep.py \
    --stage stage1 \
    --epochs 250 \
    --batch 32 \
    --imgsz 640 \
    --seed 0 >> module_sweep_e250.log 2>&1 &
  echo $! > module_sweep_e250.pid
  echo "STAGE1_RESTARTED pid=$(cat module_sweep_e250.pid)"
fi

if ps -eo pid,args | grep '[t]rain_yolo26_stage2_combo.py' >/dev/null; then
  echo "STAGE2_ALREADY_RUNNING"
  ps -eo pid,stat,etime,args | grep '[t]rain_yolo26_stage2_combo.py'
else
  nohup python -u train_yolo26_stage2_combo.py \
    --wait-stage1 \
    --poll-seconds 300 \
    --max-wait-hours 72 \
    --max-variants 10 \
    --epochs 250 \
    --batch 32 \
    --imgsz 640 \
    --seed 0 >> module_combo_e250.log 2>&1 &
  echo $! > module_combo_e250.pid
  echo "STAGE2_RESTARTED pid=$(cat module_combo_e250.pid)"
fi

sleep 5
echo "PROCS"
ps -eo pid,ppid,stat,etime,args | grep -E '[t]rain_yolo26_module_sweep.py|[t]rain_yolo26_stage2_combo.py' || true
echo "GPU"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo "STAGE1_STATUS"
cat runs_module_sweep_e250/status.json 2>/dev/null || true
echo
echo "STAGE2_STATUS"
cat runs_module_combo_e250/status.json 2>/dev/null || true
