#!/usr/bin/env python
"""iter33 build check: MSDGS neck + backbone P2/4 后插 LCAE。
复用 train_lcae_gsdown.build_lcae_doc（含层索引偏移自检），构建 + forward + fuse + 数参。
不训练。核验：LCAE 在 index 2；Detect from=[17,20,23]；forward 通；fused 参数在红线内。
"""
import os
os.environ.setdefault("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26")

import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.yolo26_lcae import LCAE  # noqa: E402,F401

import train_lcae_gsdown as T  # noqa: E402

print("=== build LCAE doc (with index-shift self-checks) ===")
doc = T.build_lcae_doc()

GEN = os.path.join(os.environ["YOLO26_EXP_ROOT"], "generated_models_lcae_gsdown_e250")
os.makedirs(GEN, exist_ok=True)
cfg = os.path.join(GEN, "lcae.yaml")
with open(cfg, "w", encoding="utf-8") as f:
    f.write("# build_check\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
print("CFG", cfg)

# 打印 backbone 前几层 + head Detect，人工核对偏移
print("--- backbone[0:4] ---")
for i, b in enumerate(doc["backbone"][:4]):
    print(i, b[0], b[2])
print("--- head Detect ---")
print("detect_from", doc["head"][-1][0])

print("=== build model ===")
model = YOLO(cfg, task="detect")
m = model.model
head = m.model[-1]
print("head_class", type(head).__name__)
print("end2end", getattr(head, "end2end", None), "reg_max", getattr(head, "reg_max", None), "nl", getattr(head, "nl", None))

# 确认 LCAE 真在模型里
lcae_mods = [type(x).__name__ for x in m.model if type(x).__name__ == "LCAE"]
print("lcae_in_model", len(lcae_mods))

n_unfused = sum(p.numel() for p in m.parameters())
print("params_unfused", n_unfused)

m.eval()
x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    y = m(x)
print("forward_ok", type(y).__name__ if not isinstance(y, (list, tuple)) else "seq")

m.fuse()
n_fused = sum(p.numel() for p in m.parameters())
print("params_fused", n_fused, "(gsdown fused 1936000, redline)")
print("BUILD_LCAE_OK")
