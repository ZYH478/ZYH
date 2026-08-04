#!/usr/bin/env python
"""论文对比表统一评测：gsdown(baseline) / MSDGS135eq / A2C2f+MSDGS 背靠背。

目的：三个模型在**同一进程、同一 test 集、同一测速协议**下产出可比的
map50 / map50-95 / 逐类 / FPS。避免跨 run 拼表的 valid-vs-test、150e-vs-250e、
不同 GPU 负载 FPS 不可比等问题。

口径（owner 定）：baseline 行 = gsdown（250e / test 齐全），全部用 test split。

测速协议（保证 FPS 可比）：
- 固定 imgsz=640、batch=1、同一张 GPU、同一进程内顺序测；
- 每个模型先 warmup N 次，再计时 R 次取中位数（毫秒），FPS=1000/中位数；
- 纯推理时间（不含预处理/后处理），与 ultralytics speed['inference'] 同口径但自测更稳。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py     # GSConv/VoVGSCSP
    python install_msdgs_module.py       # MSDGS
    # A2C2f 原生，无需安装
    python -u bench_three_models.py

输出：bench_three_models.json（含三模型整机+逐类+FPS）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
OUT_JSON = ROOT / "bench_three_models.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

MODELS = [
    ("yolo26n_gsdown", ROOT / "runs_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250" / "weights" / "best.pt"),
    ("msdgs135eq", ROOT / "runs_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250" / "weights" / "best.pt"),
    ("a2c2f_msdgs", ROOT / "runs_a2c2f_msdgs_e250" / "msdgs_a2c2f_bb" / "weights" / "best.pt"),
]

IMGSZ = 640
WARMUP = 30
REPEAT = 200


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def eval_metrics(model: YOLO) -> dict:
    m = model.val(data=str(DATA), split="test", imgsz=IMGSZ, batch=32, device=0,
                  verbose=False, save_json=False, plots=False)
    out = {
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_class": {},
    }
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[int(c)] if int(c) < len(NAMES) else str(int(c))
        out["per_class"][name] = {
            "map50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i]),
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
        }
    return out


def bench_fps(model: YOLO) -> dict:
    """固定 batch=1、随机固定输入，warmup 后多次取中位数纯推理时延。"""
    net = model.model.eval().to("cuda:0")
    # fuse 后测（部署口径）
    try:
        net.fuse()
    except Exception:
        pass
    x = torch.rand(1, 3, IMGSZ, IMGSZ, device="cuda:0")
    with torch.no_grad():
        for _ in range(WARMUP):
            net(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPEAT):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    med = statistics.median(ts)
    mean = statistics.mean(ts)
    return {
        "infer_ms_median": med, "infer_ms_mean": mean,
        "fps_median": 1000.0 / med, "fps_mean": 1000.0 / mean,
        "warmup": WARMUP, "repeat": REPEAT, "batch": 1, "imgsz": IMGSZ,
    }


def fused_params(model: YOLO) -> int:
    return int(sum(p.numel() for p in model.model.parameters()))


def main() -> int:
    print(f"BENCH_START {now()} device={torch.cuda.get_device_name(0)}")
    report = {"schema_version": 1, "generated_at": now(), "split": "test",
              "imgsz": IMGSZ, "models": {}}
    for name, weights in MODELS:
        assert weights.exists(), f"missing weights: {weights}"
        print(f"--- {name} <- {weights}")
        # 指标用一个实例（val 内部会自行处理），FPS 用另一个 fresh 实例避免 val 状态干扰
        m_eval = YOLO(str(weights))
        metrics = eval_metrics(m_eval)
        m_fps = YOLO(str(weights))
        params = fused_params(m_fps)
        fps = bench_fps(m_fps)
        report["models"][name] = {
            "weights": str(weights), "params_fused": params,
            "metric": metrics, "speed": fps,
        }
        print(f"{name}: test_map50={metrics['map50']:.5f} "
              f"test_map50-95={metrics['map50_95']:.5f} "
              f"params={params} fps_med={fps['fps_median']:.1f}")
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"BENCH_DONE {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
