source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /root/autodl-tmp/neu-det-yolo26
python install_yolo26_exp_modules.py >/dev/null 2>&1
python install_backbone_modules.py >/dev/null 2>&1
echo "=== per_class truth (independent process reload best.pt) ==="
python per_class.py runs_gsdown_dwr_e250/y26n_gsdown_dwr_e250/weights/best.pt gsdown_dwr_truth 2>&1 | grep -vE 'torchvision|pip install|compatibility|^\s*$' | tail -n 40
