#!/usr/bin/env python
"""iter22 build 检查：确认 gsdown_spd 两候选 end2end/reg_max/nl/输出形状/fused params。

口径铁律：250e 训练前必须验证结构正确。SPDConv 只换 backbone 下采样 Conv，
head（含 end2end/reg_max）不动，理论上应保留，但仍显式核实。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python build_check_gsdown_spd.py
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
GEN_DIR = ROOT / "generated_models_gsdown_spd_e250"
CANDS = ["gsdown_spd_p3", "gsdown_spd_p3p4"]


def check_one(tag: str) -> None:
    cfg = GEN_DIR / f"y26n_{tag}_e250.yaml"
    print(f"==== BUILD_CHECK {tag} ({cfg.name}) ====")
    model = YOLO(str(cfg))
    m = model.model
    det = m.model[-1]
    end2end = getattr(det, "end2end", None)
    reg_max = getattr(det, "reg_max", None)
    nl = getattr(det, "nl", None)
    print(f"  end2end={end2end}  reg_max={reg_max}  nl={nl}")

    # unfused 参数
    unfused = sum(p.numel() for p in m.parameters())
    # fused 参数（口径铁律：真值取 fused）
    try:
        m.fuse()
    except Exception as e:
        print(f"  fuse warn: {e}")
    fused = sum(p.numel() for p in m.parameters())
    print(f"  params unfused={unfused:,}  fused={fused:,}")

    # forward 输出形状（end2end 推理路径应为 [1,300,6]）
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 640, 640))
    if isinstance(y, (list, tuple)):
        shapes = [tuple(t.shape) for t in y if hasattr(t, "shape")]
    else:
        shapes = [tuple(y.shape)]
    print(f"  forward out shapes={shapes}")

    ok = (end2end is True and reg_max == 1 and nl == 3)
    print(f"  VERDICT {'OK' if ok else 'CHECK_FAILED'} {tag}")


def main() -> int:
    for tag in CANDS:
        try:
            check_one(tag)
        except Exception:
            import traceback
            print(f"BUILD_CHECK_FAILED {tag}")
            traceback.print_exc()
    print("BUILD_CHECK_ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
