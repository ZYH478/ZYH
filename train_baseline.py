#!/usr/bin/env python
"""YOLO26n 在 NEU-DET 上的基线训练 + 验证。

固定 seed / 标准超参，作为后续改进的对比锚点。
训练结束后跑一次 val 记录 mAP 与推理速度，并打印模型参数量 / GFLOPs。
所有结果落在 runs/ 下，指标汇总打印到 stdout（也会被 nohup 日志捕获）。
"""
import sys
from ultralytics import YOLO

PROJECT = "/root/autodl-tmp/neu-det-yolo26/runs"
DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"
WEIGHTS = "/root/autodl-tmp/neu-det-yolo26/yolo26n.pt"


def main():
    model = YOLO(WEIGHTS)
    # 训练
    model.train(
        data=DATA,
        epochs=150,
        imgsz=640,
        batch=32,
        workers=8,
        seed=0,
        device=0,
        project=PROJECT,
        name="baseline",
        exist_ok=True,
        patience=50,
        verbose=True,
    )
    # 用 best.pt 在 val 上评估，记录 mAP 与速度
    best = YOLO(f"{PROJECT}/baseline/weights/best.pt")
    metrics = best.val(
        data=DATA,
        imgsz=640,
        batch=32,
        device=0,
        project=PROJECT,
        name="baseline_val",
        exist_ok=True,
    )
    # 汇总打印
    print("=" * 60)
    print("BASELINE RESULTS")
    print(f"mAP50    : {metrics.box.map50:.5f}")
    print(f"mAP50-95 : {metrics.box.map:.5f}")
    print(f"precision: {metrics.box.mp:.5f}")
    print(f"recall   : {metrics.box.mr:.5f}")
    speed = metrics.speed  # dict: preprocess/inference/postprocess/loss (ms/img)
    print(f"speed_ms : {speed}")
    infer_ms = speed.get("inference", 0)
    if infer_ms:
        print(f"FPS(inference-only): {1000.0/infer_ms:.2f}")
    print("=" * 60)
    print("DONE_BASELINE")


if __name__ == "__main__":
    sys.exit(main())
