#!/usr/bin/env python
"""在 test split 上评估指定权重并打印每类 AP。
用法: python test_eval.py <weights.pt> <run_name>
独立新进程重载 best.pt，唯一真值口径。
"""
import sys
from ultralytics import YOLO

DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def main():
    weights = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) > 2 else "test_eval"
    model = YOLO(weights)
    m = model.val(data=DATA, split="test", imgsz=640, batch=32, device=0,
                  project="/root/autodl-tmp/neu-det-yolo26/runs", name=run_name,
                  exist_ok=True, verbose=False)
    print("TEST_START")
    print(f"{'class':<18}{'mAP50':>10}{'mAP50-95':>12}")
    print(f"{'all':<18}{m.box.map50:>10.4f}{m.box.map:>12.4f}")
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[c] if c < len(NAMES) else str(c)
        print(f"{name:<18}{m.box.ap50[i]:>10.4f}{m.box.ap[i]:>12.4f}")
    print("TEST_END")


if __name__ == "__main__":
    sys.exit(main())
