#!/usr/bin/env python
"""逐个核实 iter19 候选的真实 fused params/GFLOPs（fuse 后再 info）。

model.info() 在某些版本返回聚合值可能取错位置，这里显式 fuse 后
用 sum(p.numel()) 直接数参数，并用 get_flops 取 GFLOPs。
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
PROJECT = ROOT / "runs_iter19_backbone_e250"

CANDIDATES = [
    "y26n_i19_pki_shallow_e250",
    "y26n_i19_dwr_deep_e250",
    "y26n_i19_pki_dwr_e250",
]


def main() -> None:
    for name in CANDIDATES:
        best = PROJECT / name / "weights" / "best.pt"
        if not best.exists():
            print(f"MISSING {name}")
            continue
        model = YOLO(str(best))
        model.model.fuse()
        params = sum(p.numel() for p in model.model.parameters())
        try:
            gflops = get_flops(model.model, imgsz=640)
        except Exception as exc:
            gflops = float("nan")
            print(f"  gflops err: {exc!r}")
        print(f"FUSED {name} params={params} gflops={gflops:.2f}")


if __name__ == "__main__":
    main()
