#!/usr/bin/env python
"""iter19 三个完成候选的独立进程 fused 真值复核 + fused params/GFLOPs。

口径铁律：独立新进程重载 best.pt 的 val 才是真值。
用法：python truth_iter19.py
"""
from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
DATA = ROOT / "dataset" / "neu-det.yaml"
PROJECT = ROOT / "runs_iter19_backbone_e250"

CANDIDATES = [
    "y26n_i19_pki_shallow_e250",
    "y26n_i19_dwr_deep_e250",
    "y26n_i19_pki_dwr_e250",
]

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def main() -> None:
    results = {}
    for name in CANDIDATES:
        best = PROJECT / name / "weights" / "best.pt"
        if not best.exists():
            print(f"MISSING {name}: {best}")
            continue
        model = YOLO(str(best))
        # fused params/GFLOPs
        n_layers, params, grads, gflops = model.model.info(verbose=False)
        # 独立 val（fused）
        m = model.val(data=str(DATA), split="val", imgsz=640, batch=32, device=0,
                      project=str(PROJECT), name=f"{name}_truth", exist_ok=True, verbose=False)
        map50 = float(m.box.map50)
        map5095 = float(m.box.map)
        # per-class map50-95
        pc = {}
        try:
            for i, ci in enumerate(m.box.ap_class_index):
                pc[CLASSES[int(ci)]] = round(float(m.box.maps[int(ci)]), 4)
        except Exception:
            pass
        results[name] = {
            "val_map50": round(map50, 4),
            "val_map50_95": round(map5095, 4),
            "fused_params": int(params),
            "fused_gflops": round(float(gflops), 2),
            "per_class_map50_95": pc,
        }
        print(f"TRUTH {name} val_mAP50={map50:.4f} val_mAP50-95={map5095:.4f} "
              f"fused_params={int(params)} fused_gflops={float(gflops):.2f}")
        print(f"  per_class: {pc}")

    out = PROJECT / "truth_iter19.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAVED {out}")
    # 汇总排名
    print("=== RANKED by val_map50_95 ===")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["val_map50_95"], reverse=True):
        print(f"  {name}: {r['val_map50_95']} (params {r['fused_params']}, {r['fused_gflops']}G)")


if __name__ == "__main__":
    main()
