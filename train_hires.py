#!/usr/bin/env python
"""精度改进 iteration（高分辨率）：YOLO26n + imgsz=800。

P2 头方案已验证回退。改用等价但更稳妥的思路：保持 yolo26n 原架构不变，
仅提高训练/推理分辨率到 800。NEU-DET 原生仅 200x200，升分辨率直接改善
细小纹理缺陷（crazing / rolled-in_scale）的特征表达，且不引入未收敛的新结构。
其余超参与基线严格一致（seed=0 / 150 epoch / batch 32），保证公平对比。
"""
import sys
from ultralytics import YOLO

PROJECT = "/root/autodl-tmp/neu-det-yolo26/runs"
DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"
WEIGHTS = "/root/autodl-tmp/neu-det-yolo26/yolo26n.pt"


def main():
    model = YOLO(WEIGHTS)
    model.train(
        data=DATA,
        epochs=150,
        imgsz=800,
        batch=32,
        workers=8,
        seed=0,
        device=0,
        project=PROJECT,
        name="hires",
        exist_ok=True,
        patience=50,
        verbose=True,
    )
    best = YOLO(f"{PROJECT}/hires/weights/best.pt")
    metrics = best.val(
        data=DATA,
        imgsz=800,
        batch=32,
        device=0,
        project=PROJECT,
        name="hires_val",
        exist_ok=True,
    )
    print("=" * 60)
    print("HIRES IMPROVED RESULTS")
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
    print("DONE_HIRES")


if __name__ == "__main__":
    sys.exit(main())
