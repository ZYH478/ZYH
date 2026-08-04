#!/usr/bin/env python
"""独立进程重载 best.pt，在 NEU-DET test split 上评估逐类 AP + fused params（真值口径）。

用法: python per_class_test.py <weights.pt> <tag> [imgsz]
gsdown/SPD 系模型需先注入模块（install_yolo26_exp_modules.py + install_gsconv_modules.py）。
"""
import sys
import json
from ultralytics import YOLO

ROOT = "/root/autodl-tmp/neu-det-yolo26"
PROJECT = f"{ROOT}/runs"
DATA = f"{ROOT}/dataset/neu-det.yaml"
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface',
         'rolled-in_scale', 'scratches']


def main():
    weights = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "test_eval"
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    model = YOLO(weights)
    try:
        model.model.fuse()
    except Exception:
        pass
    params = sum(p.numel() for p in model.model.parameters())
    m = model.val(data=DATA, split='test', imgsz=imgsz, batch=32, device=0,
                  project=PROJECT, name=f"pctest_{tag}", exist_ok=True, verbose=False)
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[c] if c < len(NAMES) else str(c)
        per_class[name] = [float(m.box.ap50[i]), float(m.box.ap[i])]
    out = {
        "tag": tag, "fused_params": params,
        "test_map50": float(m.box.map50), "test_map50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_class": per_class,
    }
    print("PCTEST_JSON_START")
    print(json.dumps(out, ensure_ascii=False))
    print("PCTEST_JSON_END")


if __name__ == "__main__":
    sys.exit(main())
