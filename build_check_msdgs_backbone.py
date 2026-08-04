#!/usr/bin/env python
"""build 冒烟检查：MSDGS+backbone 两候选能否 build、结构是否正确、fused 参数是否 under 红线。
不训练。启动 250e 前先跑这个，别盲跑。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py
    python install_gsconv_modules.py
    python install_msdgs_module.py
    python install_backbone_modules.py
    python build_check_msdgs_backbone.py
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from ultralytics import YOLO

import train_msdgs_backbone_gsdown as M

GSDOWN_FUSED = 1_936_000  # gsdown 交付基线 fused 参数（红线）


def check(name: str, builder) -> None:
    doc = builder()
    M.GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = M.GEN_DIR / f"{name}.yaml"
    import yaml
    cfg.write_text("# build check\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    model = YOLO(str(cfg))
    m = model.model
    # 结构断言
    end2end = getattr(m, "end2end", None)
    detect = m.model[-1]
    reg_max = getattr(detect, "reg_max", None)
    nl = getattr(detect, "nl", None)
    # fused 参数（口径铁律：fuse 后 sum numel）
    mf = model.model.fuse() if hasattr(model.model, "fuse") else m
    fused_params = int(sum(p.numel() for p in mf.parameters()))
    # 前向输出形状
    x = torch.zeros(1, 3, 640, 640)
    mf.eval()
    with torch.no_grad():
        y = mf(x)
    shape = tuple(y[0].shape) if isinstance(y, (list, tuple)) else tuple(y.shape)
    status = "OK" if fused_params < GSDOWN_FUSED else "OVER_REDLINE"
    print(f"[{name}] end2end={end2end} reg_max={reg_max} nl={nl} "
          f"out={shape} fused_params={fused_params} "
          f"({fused_params/GSDOWN_FUSED*100:.1f}% of 1.936M) -> {status}")


if __name__ == "__main__":
    for name, builder in M.BUILDERS.items():
        check(name, builder)
