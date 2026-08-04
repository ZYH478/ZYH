#!/usr/bin/env python
"""验证 SimAM 模型能否合法构建：注入 SimAM → 构建 YAML → 报告参数量/GFLOPs/end2end。

用法: python build_check_simam.py <cfg.yaml>
避免内联引号地狱：逻辑写成脚本文件。
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
    inject_simam()
    from ultralytics import YOLO
    m = YOLO(cfg)
    net = m.model
    print("BUILD_OK")
    print("params:", sum(p.numel() for p in net.parameters()))
    head = net.model[-1]
    print("head:", type(head).__name__)
    print("end2end:", getattr(net, "end2end", "NA"))
    print("head.end2end:", getattr(head, "end2end", "NA"))
    # 统计 SimAM 层数
    n_simam = sum(1 for mod in net.modules() if type(mod).__name__ == "SimAM")
    print("simam_layers:", n_simam)
    print("BUILD_DONE")


if __name__ == "__main__":
    sys.exit(main())
