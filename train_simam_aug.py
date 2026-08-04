#!/usr/bin/env python
"""iter9：iter8 的 SimAM 架构 + 温和增强 + 延长训练，冲刺 mAP50>0.80。

iter7 崩溃根因是 copy_paste=0.3 破坏 end2end/NMS-free 的 one2one 标签分配
（epoch100 曾达 0.42 说明方向对）。本轮只保留温和 mixup=0.1，去掉致命的
copy_paste，加 cos_lr + 延长到 200ep，稳定改善泛化。

用法: python train_simam_aug.py <cfg.yaml> <run_name> [epochs]
评估仍以独立进程 per_class_simam.py 重载 best.pt 为唯一真值。
"""
import sys
import torch.nn as nn


class SimAM(nn.Module):
    """Simple parameter-free attention module (ICML 2021). 零参数。"""

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (
            4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5
        return x * self.act(y)


def inject_simam():
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    tasks.SimAM = SimAM
    modules.SimAM = SimAM
    try:
        import ultralytics.nn.modules.block as block
        block.SimAM = SimAM
    except Exception:
        pass


def main():
    cfg = sys.argv[1]
    run_name = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    inject_simam()
    from ultralytics import YOLO

    PROJECT = "/root/autodl-tmp/neu-det-yolo26/runs"
    DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"
    WEIGHTS = "/root/autodl-tmp/neu-det-yolo26/yolo26n.pt"

    model = YOLO(cfg).load(WEIGHTS)
    model.train(
        data=DATA,
        epochs=epochs,
        imgsz=640,
        batch=32,
        workers=8,
        seed=0,
        device=0,
        project=PROJECT,
        name=run_name,
        exist_ok=True,
        patience=60,
        cos_lr=True,
        mixup=0.1,
        close_mosaic=15,
        verbose=False,
    )
    best = YOLO(f"{PROJECT}/{run_name}/weights/best.pt")
    m = best.val(
        data=DATA, imgsz=640, batch=32, device=0,
        project=PROJECT, name=f"{run_name}_val", exist_ok=True, verbose=False,
    )
    print("=" * 60)
    print(f"SIMAM_AUG RESULTS [{run_name}]")
    print(f"mAP50    : {m.box.map50:.5f}")
    print(f"mAP50-95 : {m.box.map:.5f}")
    print("=" * 60)
    print(f"DONE_{run_name.upper()}")


if __name__ == "__main__":
    sys.exit(main())
