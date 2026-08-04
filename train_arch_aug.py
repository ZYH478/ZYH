#!/usr/bin/env python
"""基于最佳架构 + 训练增强的迭代脚本。

用法: python train_arch_aug.py <cfg.yaml> <run_name> [imgsz]
在 iter5 最佳架构（backbone A2C2f）基础上，加数据增强正则化 + 延长训练，
突破泛化瓶颈（NEU-DET 仅 1200 张）。评估仍以独立进程 per_class.py 重载为真值。

关键增强（相对基线默认）：
- mixup=0.15 / copy_paste=0.3：小数据集正则化，改善泛化
- close_mosaic=15：末段关闭 mosaic，稳定收敛
- epochs=200 + cos_lr：更充分训练 + 余弦退火
其余（seed=0 / batch 32 / imgsz 640）与基线一致。
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
        epochs=200,
        imgsz=imgsz,
        batch=32,
        workers=8,
        seed=0,
        device=0,
        project=PROJECT,
        name=run_name,
        exist_ok=True,
        patience=60,
        cos_lr=True,
        mixup=0.15,
        copy_paste=0.3,
        close_mosaic=15,
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
    print(f"ARCH+AUG RESULTS [{run_name}]")
    print(f"mAP50    : {m.box.map50:.5f}")
    print(f"mAP50-95 : {m.box.map:.5f}")
    print(f"precision: {m.box.mp:.5f}")
    print(f"recall   : {m.box.mr:.5f}")
    print("=" * 60)
    print(f"DONE_{run_name.upper()}")


if __name__ == "__main__":
    sys.exit(main())
