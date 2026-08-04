#!/usr/bin/env python
"""iter38 build check：验证 MSDGS135eq + backbone 层6/8 A2C2f 组合能构建、
真实 fused 参数量、head 类型、forward+fuse 通过。不训练。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py
    python install_msdgs_module.py
    python build_check_a2c2f.py
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
OUT = ROOT / "generated_models_a2c2f_msdgs_e250" / "_buildcheck_a2c2f.yaml"


def build() -> dict:
    doc = yaml.safe_load(MSDGS_YAML.read_text(encoding="utf-8"))
    bb = doc["backbone"]
    swapped = []
    for i, block in enumerate(bb):
        if len(block) >= 4 and block[2] == "C3k2" and block[3] \
                and len(block[3]) >= 2 and block[3][1] is True:
            c2 = block[3][0]
            area = 4 if c2 == 512 else 1
            block[2] = "A2C2f"
            block[3] = [c2, True, area]
            swapped.append((i, c2, block[1], area))
    assert len(swapped) == 2, f"expected 2, got {swapped}"
    print(f"swapped (idx,c2,repeats,area)={swapped}")
    return doc


def main() -> int:
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# buildcheck\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    print(f"CFG {OUT}")

    model = YOLO(str(OUT), task="detect").load(str(OFFICIAL_WEIGHTS))
    m = model.model
    unfused = int(sum(p.numel() for p in m.parameters()))
    print(f"PARAMS_UNFUSED {unfused}")

    head = m.model[-1]
    print(f"HEAD {type(head).__name__} is_Detect={isinstance(head, Detect)} "
          f"end2end={getattr(head, 'end2end', None)} reg_max={getattr(head, 'reg_max', None)}")

    x = torch.zeros(1, 3, 640, 640)
    m.eval()
    with torch.no_grad():
        y = m(x)
    print(f"FORWARD_OK type={type(y).__name__}")

    try:
        m.fuse()
        fused = int(sum(p.numel() for p in m.parameters()))
        print(f"FUSE_OK PARAMS_FUSED {fused}")
    except Exception as exc:  # noqa: BLE001
        print(f"FUSE_FAIL {exc!r}")
        return 1

    print(f"VS_MSDGS_1777318 delta={fused - 1777318:+d}")
    print(f"VS_GSDOWN_1935814 delta={fused - 1935814:+d}")
    print(f"VS_REDLINE_1936000 {'OVER' if fused >= 1936000 else 'UNDER'} redline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
