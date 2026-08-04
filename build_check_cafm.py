#!/usr/bin/env python
"""iter39-A build check：验证 MSDGS135eq + backbone 层10 C2PSA→CAFM 能构建、fused 参数、head 类型。"""
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
GEN_DIR = ROOT / "generated_models_cafm_msdgs_e250"
OFFICIAL = ROOT / "yolo26n.pt"

MSDGS_FUSED = 1777318
GSDOWN_FUSED = 1935814
REDLINE = 1936000


def main() -> int:
    doc = yaml.safe_load(MSDGS_YAML.read_text(encoding="utf-8"))
    bb = doc["backbone"]
    swapped = []
    for i, block in enumerate(bb):
        if len(block) >= 4 and block[2] == "C2PSA":
            block[2] = "CAFM"
            swapped.append((i, block[3]))
    assert len(swapped) == 1, f"expected 1 C2PSA, swapped {swapped}"
    print(f"swapped (idx,args)={swapped}")

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GEN_DIR / "_buildcheck_cafm.yaml"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")

    model = YOLO(str(cfg), task="detect").load(str(OFFICIAL))
    m = model.model
    params_unfused = int(sum(p.numel() for p in m.parameters()))
    print(f"PARAMS_UNFUSED {params_unfused}")

    head = m.model[-1]
    is_detect = isinstance(head, Detect)
    print(f"HEAD {type(head).__name__} is_Detect={is_detect} "
          f"end2end={getattr(head, 'end2end', None)} reg_max={getattr(head, 'reg_max', None)}")

    x = torch.zeros(1, 3, 640, 640)
    m.eval()
    with torch.no_grad():
        y = m(x)
    print(f"FORWARD_OK type={type(y).__name__}")

    m.fuse()
    fused = int(sum(p.numel() for p in m.parameters()))
    print(f"FUSE_OK PARAMS_FUSED {fused}")
    print(f"VS_MSDGS_{MSDGS_FUSED} delta={fused - MSDGS_FUSED:+d}")
    print(f"VS_GSDOWN_{GSDOWN_FUSED} delta={fused - GSDOWN_FUSED:+d}")
    print(f"VS_REDLINE_{REDLINE} {'UNDER' if fused < REDLINE else 'OVER'} redline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
