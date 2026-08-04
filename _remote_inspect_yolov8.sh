#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/neu-det-yolo26
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26

echo '--- host/runtime ---'
date
python -c 'import sys, torch, ultralytics; print("python", sys.version.split()[0]); print("torch", torch.__version__); print("ultralytics", ultralytics.__version__); print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'

echo '--- GPU/process ---'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader
pgrep -af 'python.*(train|eval)' || true

echo '--- relevant files ---'
find . -maxdepth 3 -type f \( -name '*aluminum*.yaml' -o -name '*pcb*.yaml' -o -name 'neu-det.yaml' -o -name 'eval_generalization_aluminum_pcb.py' \) -print | sort

for file in \
  dataset/neu-det.yaml \
  runs_generalization_aluminum_pcb_yolo26_msdgs_e250/datasets/aluminum.yaml \
  runs_generalization_aluminum_pcb_yolo26_msdgs_e250/datasets/pcb.yaml; do
  echo "--- ${file} ---"
  if [[ -f "${file}" ]]; then
    cat "${file}"
  else
    echo MISSING
  fi
done

echo '--- weights ---'
ls -lh yolo26n.pt yolov8n.pt 2>/dev/null || true

echo '--- existing YOLOv8 dirs ---'
find . -maxdepth 2 -type d \( -iname '*yolo8*' -o -iname '*yolov8*' \) -print | sort
