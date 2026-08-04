#!/usr/bin/env python
"""iter23 build 检查：三候选 end2end/reg_max/nl/输出形状/fused params + 逐层结构。

口径铁律：250e 训练前必须验证结构正确，尤其 FEM 双输入插层后的索引重映射。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py
    python install_gsconv_modules.py
    python install_msrdet_modules.py
    python build_check_msrdet.py
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from ultralytics import YOLO

import train_msrdet_gsdown as t

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))


def check_one(name: str) -> None:
    print(f"\n==== BUILD_CHECK {name} ====")
    doc = t.BUILDERS[name]()
    # 打印关键层，确认模块正确落位
    print("  backbone modules:", [(i, r[2]) for i, r in enumerate(doc["backbone"])])
    print("  head from/module:", [(r[0], r[2]) for r in doc["head"]])

    t.GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = t.GEN_DIR / f"{name}.yaml"
    import yaml
    cfg.write_text("# build_check\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    model = YOLO(str(cfg))
    m = model.model
    det = m.model[-1]
    end2end = getattr(det, "end2end", None)
    reg_max = getattr(det, "reg_max", None)
    nl = getattr(det, "nl", None)
    print(f"  end2end={end2end}  reg_max={reg_max}  nl={nl}")

    unfused = sum(p.numel() for p in m.parameters())
    try:
        m.fuse()
    except Exception as e:
        print(f"  fuse warn: {e}")
    fused = sum(p.numel() for p in m.parameters())
    print(f"  params unfused={unfused:,}  fused={fused:,}  (gsdown 对照 fused=1,936,xxx)")

    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 640, 640))
    if isinstance(y, (list, tuple)):
        shapes = [tuple(x.shape) for x in y if hasattr(x, "shape")]
    else:
        shapes = [tuple(y.shape)]
    print(f"  forward out shapes={shapes}")

    ok = (end2end is True and reg_max == 1 and nl == 3)
    print(f"  VERDICT {'OK' if ok else 'CHECK_FAILED'} {name}")


def main() -> int:
    for name in t.BUILDERS:
        try:
            check_one(name)
        except Exception:
            import traceback
            print(f"BUILD_CHECK_FAILED {name}")
            traceback.print_exc()
    print("\nBUILD_CHECK_ALL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
