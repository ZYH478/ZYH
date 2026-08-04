#!/usr/bin/env python
"""通用架构改进训练脚本，供多轮架构迭代复用。

用法: python train_arch.py <cfg.yaml> <run_name> [imgsz]
- cfg      : 模型结构 yaml（保留 end2end/reg_max 等 YOLO26 核心特性）
- run_name : runs/ 下的实验名
- imgsz    : 默认 640，与基线一致

超参与基线严格一致（seed=0 / 150 epoch / batch 32 / imgsz 640），保证公平对比。
从 yolo26n.pt 加载可复用 backbone 预训练权重（结构不匹配部分自动跳过）。
训练后用 best.pt 在 val 上评估，打印统一口径指标 + 每类 AP。
"""
import sys
from ultralytics import YOLO

PROJECT = "/root/autodl-tmp/neu-det-yolo26/runs"
DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"
WEIGHTS = "/root/autodl-tmp/neu-det-yolo26/yolo26n.pt"
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


def main():
    cfg = sys.argv[1]
    run_name = sys.argv[2]
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640

    model = YOLO(cfg).load(WEIGHTS)
    model.train(
        data=DATA,
        epochs=150,
        imgsz=imgsz,
        batch=32,
        workers=8,
        seed=0,
        device=0,
        project=PROJECT,
        name=run_name,
        exist_ok=True,
        patience=50,
        verbose=True,
    )
    best = YOLO(f"{PROJECT}/{run_name}/weights/best.pt")
    m = best.val(
        data=DATA,
        imgsz=imgsz,
        batch=32,
        device=0,
        project=PROJECT,
        name=f"{run_name}_val",
        exist_ok=True,
        verbose=False,
    )
    print("=" * 60)
    print(f"ARCH RESULTS [{run_name}]")
    print(f"mAP50    : {m.box.map50:.5f}")
    print(f"mAP50-95 : {m.box.map:.5f}")
    print(f"precision: {m.box.mp:.5f}")
    print(f"recall   : {m.box.mr:.5f}")
    speed = m.speed
    print(f"speed_ms : {speed}")
    infer_ms = speed.get("inference", 0)
    if infer_ms:
        print(f"FPS(inference-only): {1000.0/infer_ms:.2f}")
    print("--- per class ---")
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[c] if c < len(NAMES) else str(c)
        print(f"{name:<18}{m.box.ap50[i]:>10.4f}{m.box.ap[i]:>12.4f}")
    print("=" * 60)
    print(f"DONE_{run_name.upper()}")


if __name__ == "__main__":
    sys.exit(main())
