#!/usr/bin/env python
"""诊断 winner(SPD_P3+DySample)在 UWWT 上失败根因：
是否为非方形 rect 输入导致 DySample 上采样尺寸不匹配。

方形 640x640 前向应成功；矩形 512x672 前向若复现 16 vs 21 报错，则确认根因。
"""
import torch
from ultralytics import YOLO

CFG = "/root/autodl-tmp/neu-det-yolo26/generated_models_module_combo2_e250/y26n_s2_spd_p3_dysample_e250.yaml"

ym = YOLO(CFG)
# 用 nc=5 重建，模拟 UWWT
try:
    ym.model.nc = 5
except Exception:
    pass
m = ym.model.eval()

def try_forward(h, w):
    x = torch.rand(1, 3, h, w)
    try:
        with torch.no_grad():
            y = m(x)
        print(f"OK  {h}x{w}")
        return True
    except Exception as e:
        print(f"FAIL {h}x{w}: {e!r}")
        return False

print("=== square ===")
try_forward(640, 640)
print("=== rect (non-square, 32-multiple) ===")
try_forward(512, 672)
try_forward(672, 512)
try_forward(640, 480)
