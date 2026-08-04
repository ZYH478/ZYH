source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /root/autodl-tmp/neu-det-yolo26
echo "=== install exp modules ==="
python install_yolo26_exp_modules.py 2>&1 | tail -n 3
echo "=== install backbone modules ==="
python install_backbone_modules.py 2>&1 | tail -n 3
echo "=== dry run ==="
python -u train_iter21_combo.py --dry 2>&1 | tail -n 40
