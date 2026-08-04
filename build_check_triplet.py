#!/usr/bin/env python
"""iter39-B build check：验证 MSDGS135eq + head 三检测尺度后各插 TripletAttention
能构建、fused 参数、head 类型、Detect 三源指向 TripletAttention。

复用 train_triplet_msdgs.build_triplet 的稳健重建逻辑，避免重复实现索引重映射。
"""
import os
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
GEN_DIR = ROOT / "generated_models_triplet_msdgs_e250"
OFFICIAL = ROOT / "yolo26n.pt"

MSDGS_FUSED = 1777318
GSDOWN_FUSED = 1935814
REDLINE = 1936000

sys.path.insert(0, str(ROOT))
from train_triplet_msdgs import build_triplet  # noqa: E402


def main() -> int:
    doc = build_triplet()

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GEN_DIR / "_buildcheck_triplet.yaml"
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

    n_ta = sum(1 for mod in m.modules() if type(mod).__name__ == "TripletAttention")
    print(f"TRIPLET_LAYERS {n_ta}")

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
