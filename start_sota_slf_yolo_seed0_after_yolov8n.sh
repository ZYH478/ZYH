#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/neu-det-yolo26
WAIT_LOG=/root/autodl-tmp/neu-det-yolo26/slf_yolo_seed0_wait.log
RUN_LOG=/root/autodl-tmp/neu-det-yolo26/slf_yolo_seed0_e250.log
{
  echo "[$(date '+%F %T')] WAIT_START: waiting for train_sota_yolov8n_seed0.py to exit before SLF-YOLO."
  while pgrep -f 'train_sota_yolov8n_seed0.py' >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] WAITING: YOLOv8n still running: $(pgrep -af 'train_sota_yolov8n_seed0.py' | tr '\n' ' ' | cut -c1-240)"
    sleep 300
  done
  echo "[$(date '+%F %T')] WAIT_DONE: YOLOv8n process absent; preparing SLF-YOLO."
  if [ -f "$RUN_LOG" ]; then
    mv "$RUN_LOG" "${RUN_LOG}.failed_cache_$(date '+%Y%m%d_%H%M%S')"
  fi
  # Remove only Ultralytics label caches for the three target datasets. These caches may be pickled by numpy 2.x
  # from the yolo26 env and fail in lightyolo/numpy 1.24 with ModuleNotFoundError: numpy._core.
  rm -f \
    /root/autodl-tmp/neu-det-yolo26/dataset/train/labels.cache \
    /root/autodl-tmp/neu-det-yolo26/dataset/valid/labels.cache \
    /root/autodl-tmp/neu-det-yolo26/dataset/test/labels.cache \
    /root/autodl-tmp/neu-det-yolo26/generalization_aluminum_pcb/raw/aluminum_yolo26/labels/train.cache \
    /root/autodl-tmp/neu-det-yolo26/generalization_aluminum_pcb/raw/aluminum_yolo26/labels/val.cache \
    /root/autodl-tmp/neu-det-yolo26/generalization_aluminum_pcb/raw/aluminum_yolo26/labels/test.cache \
    /root/autodl-tmp/neu-det-yolo26/generalization_aluminum_pcb/prepared/pcb/labels/train.cache \
    /root/autodl-tmp/neu-det-yolo26/generalization_aluminum_pcb/prepared/pcb/labels/val.cache \
    /root/autodl-tmp/neu-det-yolo26/generalization_aluminum_pcb/prepared/pcb/labels/test.cache || true
  rm -rf \
    /root/autodl-tmp/neu-det-yolo26/runs_sota_slf_yolo_seed0_e250/neudet \
    /root/autodl-tmp/neu-det-yolo26/runs_sota_slf_yolo_seed0_e250/aluminum \
    /root/autodl-tmp/neu-det-yolo26/runs_sota_slf_yolo_seed0_e250/pcb || true
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate lightyolo
  export PYTHONPATH=/root/autodl-tmp/neu-det-yolo26/SLF-YOLO/SLF-YOLO:${PYTHONPATH:-}
  export YOLO_CONFIG_DIR=/root/autodl-tmp/neu-det-yolo26/.yolo_config_slf
  export MPLCONFIGDIR=/root/autodl-tmp/neu-det-yolo26/.mpl_config_slf
  export TORCH_HOME=/root/autodl-tmp/neu-det-yolo26/.torch_cache
  echo "[$(date '+%F %T')] SLF_START: python -u train_sota_slf_yolo_seed0.py"
  exec python -u train_sota_slf_yolo_seed0.py >> "$RUN_LOG" 2>&1
} >> "$WAIT_LOG" 2>&1
