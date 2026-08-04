source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /root/autodl-tmp/neu-det-yolo26
python install_yolo26_exp_modules.py >/dev/null 2>&1
python install_backbone_modules.py >/dev/null 2>&1
mkdir -p runs_iter21_combo_e250
setsid nohup python -u train_iter21_combo.py > runs_iter21_combo_e250/train.log 2>&1 &
echo "PID=$!"
sleep 3
echo "=== nohup started, tail log ==="
tail -n 15 runs_iter21_combo_e250/train.log
