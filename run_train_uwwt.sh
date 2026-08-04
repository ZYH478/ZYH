source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /root/autodl-tmp/neu-det-yolo26
python install_yolo26_exp_modules.py >/dev/null 2>&1
python install_backbone_modules.py >/dev/null 2>&1
mkdir -p runs_uwwt_e250
setsid nohup python -u train_uwwt.py > runs_uwwt_e250/train.log 2>&1 < /dev/null &
echo "PID=$!"
sleep 8
echo "=== tail log ==="
tail -n 20 runs_uwwt_e250/train.log
