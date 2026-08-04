#!/usr/bin/env python
"""评估注入 SimAM 的模型：自包含 SimAM 定义 + 注入 + 重载 best.pt 评估。

用法: python per_class_simam.py <weights.pt> <run_name> [imgsz]
对普通模型也无害（注入未使用的类不影响）。评估口径与 per_class.py 一致，
是 SimAM 模型的唯一真值评估器（独立新进程重载）。
"""
import sys
import torch
import torch.nn as nn

PROJECT = "/root/autodl-tmp/neu-det-yolo26/runs"
DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"
NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


class SimAM(nn.Module):
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
    weights = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) > 2 else "per_class_simam"
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640

    inject_simam()
    from ultralytics import YOLO

    model = YOLO(weights)
    m = model.val(data=DATA, imgsz=imgsz, batch=32, device=0,
                  project=PROJECT, name=run_name, exist_ok=True, verbose=False)
    print("PER_CLASS_START")
    print(f"{'class':<18}{'mAP50':>10}{'mAP50-95':>12}")
    print(f"{'all':<18}{m.box.map50:>10.4f}{m.box.map:>12.4f}")
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[c] if c < len(NAMES) else str(c)
        print(f"{name:<18}{m.box.ap50[i]:>10.4f}{m.box.ap[i]:>12.4f}")
    print("PER_CLASS_END")


if __name__ == "__main__":
    sys.exit(main())
