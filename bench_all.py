#!/usr/bin/env python
"""统一同口径基准：对所有关键模型测 fused params / GFLOPs / val mAP50 / mAP50-95 / 6类 mAP50-95 / FPS。

口径铁律：每个模型独立 YOLO() 全新重载 best.pt。
- 精度：model.val(split='val')，取 results.box.map/map50 + per-class maps。
- FPS：重新加载 → model.fuse() → GPU batch=1 imgsz=640，warmup 100 + 计时 300 取平均。
输出 JSON 便于机读 + 打印汇总。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
OUT = ROOT / "runs_iter21_combo_e250" / "bench_all.json"

MODELS = {
    "base": "runs_module_sweep_e250/y26n_base_e250/weights/best.pt",
    "winner_spd_p3_dysample": "runs_module_combo2_e250/y26n_s2_spd_p3_dysample_e250/weights/best.pt",
    "gsdown": "runs_module_stage3_e250/y26n_s3_vovgscsp_gsdown_e250/weights/best.pt",
    "dwr_deep_i19": "runs_iter19_backbone_e250/y26n_i19_dwr_deep_e250/weights/best.pt",
    "gsdown_dwr_i20": "runs_gsdown_dwr_e250/y26n_gsdown_dwr_e250/weights/best.pt",
    "i21_gsdown_pki": "runs_iter21_combo_e250/y26n_i21_gsdown_pki_e250/weights/best.pt",
    "i21_gsdown_pki_dwr": "runs_iter21_combo_e250/y26n_i21_gsdown_pki_dwr_e250/weights/best.pt",
    "i21_winner_dwr": "runs_iter21_combo_e250/y26n_i21_winner_dwr_e250/weights/best.pt",
    "i21_winner_pki_dwr": "runs_iter21_combo_e250/y26n_i21_winner_pki_dwr_e250/weights/best.pt",
}

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def measure_fps(weights: str, warmup: int = 100, iters: int = 300) -> tuple[float, float]:
    ymodel = YOLO(str(ROOT / weights))
    m = ymodel.model
    m.fuse()
    m.eval().to(DEV)
    x = torch.rand(1, 3, 640, 640, device=DEV)
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    params = sum(p.numel() for p in m.parameters())
    return 1.0 / dt, params


results = {}
for name, wp in MODELS.items():
    full = ROOT / wp
    if not full.exists():
        print(f"SKIP {name}: missing {wp}")
        continue
    print(f"########## BENCH {name} ##########")
    # 精度（独立重载）
    ymodel = YOLO(str(full))
    metrics = ymodel.val(data=str(DATA), split="val", verbose=False)
    names = ymodel.model.names  # {idx: cls}
    maps = metrics.box.maps  # per-class mAP50-95, ndarray shape (nc,)
    per_class = {names[i]: float(maps[i]) for i in range(len(maps))}
    # FPS（重新加载 + fuse）
    fps, params = measure_fps(wp)
    rec = {
        "weights": wp,
        "fused_params": int(params),
        "val_map50": float(metrics.box.map50),
        "val_map50_95": float(metrics.box.map),
        "per_class_map50_95": per_class,
        "fps": round(fps, 1),
    }
    results[name] = rec
    print(f"  params={params/1e6:.4f}M val_map50={rec['val_map50']:.4f} "
          f"val_map50-95={rec['val_map50_95']:.4f} fps={rec['fps']}")
    print(f"  per_class={per_class}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"BENCH_ALL_DONE -> {OUT}")
