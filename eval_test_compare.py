#!/usr/bin/env python
"""在 NEU-DET test split 上评估 base 与 vovgscsp_gsdown 两个模型，打印逐类 AP。

口径：独立进程重载 best.pt，model.val(split='test')。
gsdown 需 GSConv/VoVGSCSP 已注入 ultralytics（先跑 install_gsconv_modules.py）。

用法（远程）：python eval_test_compare.py
"""
import sys
from ultralytics import YOLO

ROOT = "/root/autodl-tmp/neu-det-yolo26"
PROJECT = f"{ROOT}/runs"
DATA = f"{ROOT}/dataset/neu-det.yaml"
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface',
         'rolled-in_scale', 'scratches']

MODELS = [
    ("base(yolo26n)",
     f"{ROOT}/runs_module_sweep_e250/y26n_base_e250/weights/best.pt"),
    ("vovgscsp_gsdown",
     f"{ROOT}/runs_module_stage3_e250/y26n_s3_vovgscsp_gsdown_e250/weights/best.pt"),
]


def eval_one(tag, weights):
    model = YOLO(weights)
    # fused params 真值
    try:
        model.model.fuse()
    except Exception:
        pass
    params = sum(p.numel() for p in model.model.parameters())
    m = model.val(data=DATA, split='test', imgsz=640, batch=32, device=0,
                  project=PROJECT, name=f"testcmp_{tag}", exist_ok=True,
                  verbose=False)
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[c] if c < len(NAMES) else str(c)
        per_class[name] = (float(m.box.ap50[i]), float(m.box.ap[i]))
    return {
        "tag": tag,
        "params": params,
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": per_class,
    }


def main():
    results = []
    for tag, w in MODELS:
        print(f"EVAL_START {tag}")
        results.append(eval_one(tag, w))
        print(f"EVAL_DONE {tag}")

    print("RESULT_JSON_START")
    import json
    print(json.dumps(results, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    sys.exit(main())
