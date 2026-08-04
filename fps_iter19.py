#!/usr/bin/env python
"""补测 dwr_deep 的推理 FPS（fused，与 gsdown/winner 同口径 bs=1@640）。"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
PROJECT = ROOT / "runs_iter19_backbone_e250"

TARGETS = [
    "y26n_i19_dwr_deep_e250",
    "y26n_i19_pki_dwr_e250",
]


def bench(name: str) -> None:
    best = PROJECT / name / "weights" / "best.pt"
    if not best.exists():
        print(f"MISSING {name}")
        return
    m = YOLO(str(best))
    net = m.model.fuse().eval().cuda()
    x = torch.zeros(1, 3, 640, 640).cuda()
    with torch.no_grad():
        for _ in range(30):
            net(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(300):
            net(x)
        torch.cuda.synchronize()
        dt = (time.time() - t) / 300
    print(f"FPS {name} mean {dt*1000:.3f}ms FPS {1/dt:.1f}")


def main() -> None:
    for name in TARGETS:
        bench(name)


if __name__ == "__main__":
    main()
