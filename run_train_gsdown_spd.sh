#!/bin/bash
# iter22: 后台串行训练 gsdown_spd 两候选（setsid 脱离终端，挺过 SSH 断开）
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate yolo26
cd /root/autodl-tmp/neu-det-yolo26
# 幂等注入实验模块（SPDConv）与 SlimNeck 模块（GSConv/VoVGSCSP，gsdown head 需要）
python install_yolo26_exp_modules.py
python install_gsconv_modules.py
# 后台串行两候选，日志 gsdown_spd_e250.log
setsid nohup python -u train_gsdown_spd.py > gsdown_spd_e250.log 2>&1 &
echo "LAUNCHED pid=$!"
