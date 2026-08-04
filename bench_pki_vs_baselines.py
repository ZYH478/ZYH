#!/usr/bin/env python
"""iter063 论文对比表统一评测：gsdown / MSDGS135eq / PKI(msdgs_pki_l4) 背靠背。

在 bench_three_models.py 基础上把第三个模型换成 PKI。目的与原脚本一致：三模型在
**同一进程、同一 test 集、同一测速协议**下产出可比的 map50 / map50-95 / 逐类 / FPS，
解决 iter37 PKI 的 FPS（560）与 gsdown/MSDGS（178/185）非同环境不可比的问题。

FPS/参数量由结构决定、与训练 seed 无关，故 PKI 用现存 seed0 best.pt 即可测出可比 FPS；
论文表里 PKI 的 mAP50 用多 seed 均值（来自 runs_pki_multiseed_e250/report.json），本 bench
产出的 PKI 单点指标应与 iter37 seed0 独立真值一致（自洽校验）。

口径（owner 定）：baseline 行 = gsdown（250e / test 齐全），全部用 test split。
测速：固定 imgsz=640、batch=1、同一张 GPU、同进程顺序测，warmup 后取中位数纯推理时延。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py     # GSConv/VoVGSCSP
    python install_msdgs_module.py       # MSDGS
    python install_pki_module.py         # PKIC3k2
    python -u bench_pki_vs_baselines.py

输出：bench_pki_vs_baselines.json（含三模型整机+逐类+FPS）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
OUT_JSON = ROOT / "bench_pki_vs_baselines.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# PKI 权重：默认用 iter37 seed0（FPS/参数与 seed 无关）；可用 PKI_BENCH_WEIGHTS 覆盖为多 seed 里
# mAP50 最高的一个，做指标自洽校验。
PKI_WEIGHTS = Path(
    os.environ.get(
        "PKI_BENCH_WEIGHTS",
        ROOT / "runs_pki_gsdown_e250" / "msdgs_pki_l4" / "weights" / "best.pt",
    )
)

MODELS = [
    ("yolo26n_gsdown", ROOT / "runs_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250" / "weights" / "best.pt"),
    ("msdgs135eq", ROOT / "runs_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250" / "weights" / "best.pt"),
    ("pki_msdgs_l4", PKI_WEIGHTS),
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
              "imgsz": IMGSZ, "pki_weights": str(PKI_WEIGHTS), "models": {}}
    for name, weights in MODELS:
        assert weights.exists(), f"missing weights: {weights}"
        print(f"--- {name} <- {weights}")
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
