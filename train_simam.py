#!/usr/bin/env python
"""iter8：iter5 最佳架构（主干 A2C2f）+ 检测头前 SimAM 无参注意力。

用 monkey-patch 注入 SimAM，不改 site-packages 源码（安全、易回滚）。
关键：parse_model 用 globals()[m] 解析模块名，故把 SimAM 注入 tasks 模块
全局命名空间；SimAM 通道不变，走 else 分支 c2=ch[f]，YAML args 写 []。

SimAM（Yang et al., ICML 2021）：基于神经元能量函数计算 3D 注意力权重，
零参数增量，放大信息量大的神经元、抑制冗余，对细小低对比度缺陷有效。
e_t 越小神经元越重要，注意力 = sigmoid(1/(4*(var+lambda)) + 0.5) 的倒数形式。

用法: python train_simam.py <cfg.yaml> <run_name> [epochs]
评估仍以独立进程 per_class.py 重载 best.pt 为唯一真值。
"""
import sys
import torch
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
    """把 SimAM 注入 ultralytics 模块命名空间，让 YAML 能按名解析。"""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    # parse_model 用 globals()[m]，故必须进 tasks 模块全局命名空间
    tasks.SimAM = SimAM
    modules.SimAM = SimAM
    # 保险：也放进 nn.modules.block（部分版本从此再导出）
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
        verbose=False,
    )
    best = YOLO(f"{PROJECT}/{run_name}/weights/best.pt")
    m = best.val(
        data=DATA, imgsz=640, batch=32, device=0,
        project=PROJECT, name=f"{run_name}_val", exist_ok=True, verbose=False,
    )
    print("=" * 60)
    print(f"SIMAM RESULTS [{run_name}]")
    print(f"mAP50    : {m.box.map50:.5f}")
    print(f"mAP50-95 : {m.box.map:.5f}")
    print("=" * 60)
    print(f"DONE_{run_name.upper()}")


if __name__ == "__main__":
    sys.exit(main())
