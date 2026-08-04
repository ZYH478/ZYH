#!/usr/bin/env python
"""同口径实测 base / gsdown / winner_dwr 的推理 FPS（稳定版）。

针对首测出现的自相矛盾（gsdown 反比 base 慢、绝对值仅历史一半）做加固：
- 大 warmup（300）把 GPU 时钟拉满，规避冷启动/降频噪声。
- 每个模型连测 3 轮，每轮 500 iters，报每轮 + 取中位数，判断是否稳定。
- 交叉顺序无关：每模型独立重载 + fuse。
口径与 bench_all.py 一致：独立重载 best.pt -> fuse -> GPU bs=1@640。
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "base": "runs_module_sweep_e250/y26n_base_e250/weights/best.pt",
    "gsdown": "runs_module_stage3_e250/y26n_s3_vovgscsp_gsdown_e250/weights/best.pt",
    "winner_dwr": "runs_iter21_combo_e250/y26n_i21_winner_dwr_e250/weights/best.pt",
}


def bench(wp: str, warmup: int = 300, iters: int = 500, rounds: int = 3):
    full = ROOT / wp
    if not full.exists():
        print(f"MISSING {wp}")
        return None
    m = YOLO(str(full)).model
    m.fuse()
    m.eval().to(DEV)
    x = torch.rand(1, 3, 640, 640, device=DEV)
    fps_list = []
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(iters):
                m(x)
            if DEV == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / iters
            fps_list.append(1.0 / dt)
    params = sum(p.numel() for p in m.parameters())
    return fps_list, params


def main():
    print("DEVICE", DEV)
    if DEV == "cuda":
        print("GPU", torch.cuda.get_device_name(0))
    for name, wp in MODELS.items():
        r = bench(wp)
        if r is None:
            continue
        fps_list, params = r
        fps_list_sorted = sorted(fps_list)
        med = fps_list_sorted[len(fps_list_sorted) // 2]
        rounds_str = ", ".join(f"{f:.1f}" for f in fps_list)
        print(f"FPSRES {name} median={med:.1f} rounds=[{rounds_str}] "
              f"ms={1000.0/med:.3f} params={params} ({params/1e6:.4f}M)")


if __name__ == "__main__":
    main()
