#!/usr/bin/env python
"""统一同口径基准 —— test split（299 张）。口径铁律：每模型独立 YOLO() 重载 best.pt。
只测精度：model.val(split='test')，取 map/map50 + per-class maps。FPS 与设备无关，val 版已测。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
OUT = ROOT / "runs_iter21_combo_e250" / "bench_all_test.json"

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

results = {}
for name, wp in MODELS.items():
    full = ROOT / wp
    if not full.exists():
        print(f"SKIP {name}: missing {wp}")
        continue
    print(f"########## TEST {name} ##########")
    ymodel = YOLO(str(full))
    metrics = ymodel.val(data=str(DATA), split="test", verbose=False)
    names = ymodel.model.names
    maps = metrics.box.maps
    per_class = {names[i]: float(maps[i]) for i in range(len(maps))}
    rec = {
        "weights": wp,
        "test_map50": float(metrics.box.map50),
        "test_map50_95": float(metrics.box.map),
        "per_class_map50_95": per_class,
    }
    results[name] = rec
    print(f"  test_map50={rec['test_map50']:.4f} test_map50-95={rec['test_map50_95']:.4f}")
    print(f"  per_class={per_class}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"BENCH_TEST_DONE -> {OUT}")
