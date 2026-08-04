#!/usr/bin/env python
"""对指定权重在 val 上评估并打印每类 AP，结果落到 stdout。

用法: python per_class.py <weights.pt> <run_name>
避免内联引号地狱：逻辑写成脚本文件上传执行。
"""
import sys
from ultralytics import YOLO

PROJECT = "/root/autodl-tmp/neu-det-yolo26/runs"
DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def main():
    weights = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) > 2 else "per_class_val"
    model = YOLO(weights)
    m = model.val(data=DATA, imgsz=640, batch=32, device=0,
                  project=PROJECT, name=run_name, exist_ok=True, verbose=False)
    print("PER_CLASS_START")
    print(f"{'class':<18}{'mAP50':>10}{'mAP50-95':>12}")
    print(f"{'all':<18}{m.box.map50:>10.4f}{m.box.map:>12.4f}")
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[c] if c < len(NAMES) else str(c)
        ap50 = m.box.ap50[i]
        ap = m.box.ap[i]
        print(f"{name:<18}{ap50:>10.4f}{ap:>12.4f}")
    print("PER_CLASS_END")


if __name__ == "__main__":
    sys.exit(main())
