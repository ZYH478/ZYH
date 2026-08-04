#!/usr/bin/env python
"""build 验证 gsdown_dwr：确认能构建、end2end/reg_max/nl=3、fused params。"""
import torch
from ultralytics import YOLO

CFG = "generated_models_gsdown_dwr_e250/y26n_gsdown_dwr_e250.yaml"
m = YOLO(CFG)
det = m.model.model[-1]
n_unfused = sum(p.numel() for p in m.model.parameters())
x = torch.zeros(1, 3, 640, 640)
m.model.eval()
with torch.no_grad():
    y = m.model(x)
out = y[0].shape if hasattr(y[0], "shape") else [t.shape for t in y]
print(f"UNFUSED params={n_unfused}")
print(f"end2end={getattr(det, 'end2end', None)} reg_max={getattr(det, 'reg_max', None)} nl={det.nl}")
print(f"out={out}")

# fused
m2 = YOLO(CFG)
m2.model.fuse()
n_fused = sum(p.numel() for p in m2.model.parameters())
print(f"FUSED params={n_fused}")
print("BUILD_GSDOWN_DWR_OK")
