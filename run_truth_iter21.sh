source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /root/autodl-tmp/neu-det-yolo26
python install_yolo26_exp_modules.py >/dev/null 2>&1
python install_backbone_modules.py >/dev/null 2>&1
for name in y26n_i21_gsdown_pki_e250 y26n_i21_gsdown_pki_dwr_e250 y26n_i21_winner_dwr_e250 y26n_i21_winner_pki_dwr_e250; do
  echo "########## TRUTH $name ##########"
  python per_class.py runs_iter21_combo_e250/$name/weights/best.pt ${name}_truth 2>&1 | grep -vE 'torchvision|pip install|compatibility|Fast image|Scanning|^\s*$|it/s|ping:' | grep -E 'PER_CLASS|class|all|crazing|inclusion|patches|pitted|rolled|scratches|summary|parameters'
done
echo "ALL_TRUTH_DONE"
