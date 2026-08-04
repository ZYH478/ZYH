source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /root/autodl-tmp/neu-det-yolo26
python install_yolo26_exp_modules.py >/dev/null 2>&1
python install_backbone_modules.py >/dev/null 2>&1
python -u bench_all.py 2>&1 | grep -vE 'torchvision|pip install|compatibility|Fast image|Scanning|^\s*$|it/s|ping:|WARNING'
