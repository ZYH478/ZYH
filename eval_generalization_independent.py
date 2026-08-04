#!/usr/bin/env python
"""独立新进程重载 best.pt，生成 test 总体/逐类指标与统一 fused FPS。"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
PROJECT = ROOT / "runs_generalization_yolo26_msdgs_e250"


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def extract_metrics(result, names: list[str]) -> dict:
    box = result.box
    class_ap50 = {name: None for name in names}
    class_ap50_95 = {name: None for name in names}
    for i, class_index in enumerate(box.ap_class_index):
        ci = int(class_index)
        name = names[ci] if 0 <= ci < len(names) else str(ci)
        class_ap50[name] = float(box.ap50[i])
        class_ap50_95[name] = float(box.ap[i])
    missing = [name for name, value in class_ap50.items() if value is None]
    if missing:
        raise RuntimeError(f"test evaluation missing classes: {missing}")
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "class_ap50": class_ap50,
        "class_ap50_95": class_ap50_95,
    }


def benchmark_forward(net: torch.nn.Module, imgsz: int, batch: int, warmup: int, iterations: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the canonical FPS benchmark")
    device = torch.device("cuda:0")
    net = net.to(device).eval()
    param = next(net.parameters())
    dtype = param.dtype
    x = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
    torch.cuda.empty_cache()
    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            net(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    images = batch * iterations
    fps = images / elapsed
    return {
        "fps": float(fps),
        "infer_ms_per_image": float(1000.0 / fps),
        "elapsed_seconds": float(elapsed),
        "images": images,
        "batch": batch,
        "imgsz": imgsz,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "dtype": str(dtype).replace("torch.", ""),
        "scope": "fused model forward only; excludes preprocess and postprocess",
        "device": torch.cuda.get_device_name(0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=("yolo26n", "msdgs"))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=200)
    args = ap.parse_args()

    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    output = Path(args.output).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not data.is_file():
        raise FileNotFoundError(data)
    data_doc = yaml.safe_load(data.read_text(encoding="utf-8"))
    raw_names = data_doc["names"]
    names = [str(raw_names[i]) for i in sorted(raw_names)] if isinstance(raw_names, dict) else [str(x) for x in raw_names]

    print(f"INDEPENDENT_EVAL_START dataset={args.dataset} model={args.model} weights={weights}", flush=True)
    model = YOLO(str(weights))
    fused = model.model.fuse() if hasattr(model.model, "fuse") else model.model
    model.model = fused
    fused_params = int(sum(p.numel() for p in fused.parameters()))

    result = model.val(
        data=str(data), split="test", imgsz=args.imgsz, batch=args.batch, device=0,
        plots=False, verbose=False, rect=False, project=str(PROJECT / "independent_val_runs"),
        name=f"{args.dataset}_{args.model}", exist_ok=True,
    )
    metrics = extract_metrics(result, names)
    fps = benchmark_forward(model.model, args.imgsz, args.batch, args.warmup, args.iterations)
    det = model.model.model[-1]
    report = {
        "status": "done",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "model": args.model,
        "weights": str(weights),
        "data": str(data),
        "protocol": {
            "truth": "fresh Python process reloads best.pt from disk and evaluates test split",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": 0,
            "rect": False,
        },
        "structure": {
            "fused_params": fused_params,
            "end2end": bool(getattr(model.model, "end2end", False)),
            "reg_max": int(getattr(det, "reg_max", -1)),
            "nl": int(getattr(det, "nl", -1)),
        },
        "test": metrics,
        "benchmark": fps,
    }
    atomic_json(output, report)
    print(
        f"INDEPENDENT_EVAL_DONE dataset={args.dataset} model={args.model} "
        f"map50={metrics['map50']:.6f} map50_95={metrics['map50_95']:.6f} "
        f"fps={fps['fps']:.2f} fused_params={fused_params} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
