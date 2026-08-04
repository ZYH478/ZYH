#!/usr/bin/env python
"""精度改进 iteration：YOLO26n + P2 小目标检测头。

针对 NEU-DET 细小纹理缺陷（crazing / rolled-in_scale）增加 P2/4 高分辨率
检测头。超参与基线严格一致（seed=0 / 150 epoch / imgsz 640 / batch 32），
保证与基线公平对比。从 yolo26n.pt 加载可复用的 backbone 预训练权重。
"""
import sys
from ultralytics import YOLO

PROJECT = "/root/autodl-tmp/neu-det-yolo26/runs"
DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"
CFG = "yolo26-p2.yaml"
WEIGHTS = "/root/autodl-tmp/neu-det-yolo26/yolo26n.pt"


def main():
    # 用 P2 结构配置构建 nano 模型，并加载基线预训练权重（结构不同部分自动跳过）
    model = YOLO(CFG).load(WEIGHTS)
    model.train(
        data=DATA,
        epochs=150,
        imgsz=640,
        batch=32,
        workers=8,
        seed=0,
        device=0,
        project=PROJECT,
        name="p2",
        exist_ok=True,
        patience=50,
        verbose=True,
    )
    best = YOLO(f"{PROJECT}/p2/weights/best.pt")
    metrics = best.val(
        data=DATA,
        imgsz=640,
        batch=32,
        device=0,
        project=PROJECT,
        name="p2_val",
        exist_ok=True,
    )
    print("=" * 60)
    print("P2 IMPROVED RESULTS")
    print(f"mAP50    : {metrics.box.map50:.5f}")
    print(f"mAP50-95 : {metrics.box.map:.5f}")
    print(f"precision: {metrics.box.mp:.5f}")
    print(f"recall   : {metrics.box.mr:.5f}")
    speed = metrics.speed
    print(f"speed_ms : {speed}")
    infer_ms = speed.get("inference", 0)
    if infer_ms:
        print(f"FPS(inference-only): {1000.0/infer_ms:.2f}")
    print("=" * 60)
    print("DONE_P2")


if __name__ == "__main__":
    sys.exit(main())
