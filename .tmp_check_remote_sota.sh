#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/neu-det-yolo26
echo "PWD=$PWD"
hostname || true
date '+%F %T %Z'
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
python - <<'PY'
import sys, torch, ultralytics
print('python', sys.executable)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('ultralytics', ultralytics.__version__, ultralytics.__file__)
PY
printf '\n=== GPU ===\n'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
printf '\n=== Train processes ===\n'
pgrep -af 'train_.*(sota|generalization|multiseed|yolo11|yolov8|rtdetr)' || true
printf '\n=== Dataset yamls ===\n'
ls -l dataset/neu-det.yaml runs_generalization_aluminum_pcb_yolo26_msdgs_e250/datasets/aluminum.yaml runs_generalization_aluminum_pcb_yolo26_msdgs_e250/datasets/pcb.yaml 2>&1
printf '\n=== Existing SOTA status ===\n'
for d in runs_sota_yolov8n_seed0_e250 runs_sota_yolo11n_seed0_e250 runs_sota_rtdetr_hgnetv2_l_seed0_e250; do
  echo "---$d"
  cat "$d/status.json" 2>/dev/null || echo NO_STATUS
  echo
done
