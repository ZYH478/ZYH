#!/usr/bin/env python
"""一锤定音：用训练脚本原口径(split=test, batch=32, imgsz=640)重新 val
当前磁盘上的 a2c2f best.pt，与 report.json 记录的 test map50=0.7454 对比。

- 若得 ~0.7063 -> report.json 的 0.7454 是训练混乱期污染残留，best.pt 已是最终干净权重，
  统一 bench 可信，论文以 0.7063 为准。
- 若得 ~0.7454 -> bench 脚本口径有差异，需复查 bench。

同时打印 best.pt 的文件修改时间与训练 args，辅助判断 checkpoint 归属。
"""
import os
from pathlib import Path
import time

from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
W = ROOT / "runs_a2c2f_msdgs_e250" / "msdgs_a2c2f_bb" / "weights" / "best.pt"

print(f"BEST_PT {W}")
print(f"MTIME {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(W.stat().st_mtime))}")
print(f"SIZE {W.stat().st_size}")

m = YOLO(str(W))
r = m.val(data=str(DATA), split="test", imgsz=640, batch=32, device=0,
          project=str(ROOT / "runs_a2c2f_msdgs_e250"), name="msdgs_a2c2f_bb_reverify",
          exist_ok=True, verbose=False)
print(f"REVERIFY test map50={float(r.box.map50):.5f} map50-95={float(r.box.map):.5f}")
print(f"REPORT_JSON_SAID map50=0.74544 map50-95=0.39662")
print(f"BENCH_SAID map50=0.70626 map50-95=0.39035")
