#!/usr/bin/env python
"""Independent-process val/test evaluator for MSDGS experiment checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

# torchvision 0.18's deform_conv2d has a CPU-only kernel that SIGSEGVs inside
# thop's profiling forward (uncatchable). ultralytics calls get_flops during
# model load / val setup on CPU before the model reaches the GPU, which crashes
# any DCNv2 checkpoint. FLOPs here are cosmetic (the authoritative GPU value is
# measured in the build gate), so neuter the profiler before importing YOLO.
import ultralytics.utils.torch_utils as _tu  # noqa: E402

_tu.get_flops = lambda *a, **k: 0.0
_tu.get_flops_with_torch_profiler = lambda *a, **k: 0.0

from ultralytics import YOLO  # noqa: E402

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def metric_dict(metrics) -> dict:
    infer = float(metrics.speed.get("inference", 0.0) or 0.0)
    out = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "infer_ms": infer,
        "fps_infer_only": 1000.0 / infer if infer else None,
        "per_class": {},
    }
    for i, cls_idx in enumerate(metrics.ap_class_index):
        idx = int(cls_idx)
        name = NAMES[idx] if idx < len(NAMES) else str(idx)
        out["per_class"][name] = {
            "map50": float(metrics.box.ap50[i]),
            "map50_95": float(metrics.box.ap[i]),
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    print(f"INDEPENDENT_EVAL_PROCESS weights={weights}")
    model = YOLO(str(weights))
    params_unfused = int(sum(p.numel() for p in model.model.parameters()))
    val = model.val(
        data=args.data, split="val", imgsz=args.imgsz, batch=args.batch, device=0,
        project=args.project, name=f"{args.name}_val_independent", exist_ok=True, verbose=False,
    )
    test = model.val(
        data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, device=0,
        project=args.project, name=f"{args.name}_test_independent", exist_ok=True, verbose=False,
    )

    fused_model = YOLO(str(weights))
    try:
        fused_model.model.fuse()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN fuse failed: {exc!r}")
    params_fused = int(sum(p.numel() for p in fused_model.model.parameters()))
    # info()/get_flops runs a thop profiling forward. torchvision 0.18's
    # deform_conv2d has a CPU-only kernel that SIGSEGVs there (uncatchable), so
    # move the model to GPU when available before profiling. FLOPs are numerically
    # identical on GPU and this is a no-op for non-deform checkpoints.
    import torch as _torch

    if _torch.cuda.is_available():
        try:
            fused_model.model = fused_model.model.cuda()
        except Exception as exc:  # noqa: BLE001
            print(f"WARN move-to-cuda failed: {exc!r}")
    try:
        info = fused_model.info(verbose=False, imgsz=args.imgsz)
        gflops = float(info[-1]) if isinstance(info, (tuple, list)) and len(info) >= 4 else None
    except Exception as exc:  # noqa: BLE001
        print(f"WARN info/GFLOPs failed: {exc!r}")
        gflops = None

    result = {
        "weights": str(weights),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator_pid": __import__("os").getpid(),
        "params_unfused": params_unfused,
        "params_fused": params_fused,
        "gflops": gflops,
        "val": metric_dict(val),
        "test": metric_dict(test),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("INDEPENDENT_RESULT", f"val_mAP50-95={result['val']['map50_95']:.6f}",
          f"test_mAP50={result['test']['map50']:.6f}",
          f"test_mAP50-95={result['test']['map50_95']:.6f}",
          f"fused_params={params_fused}", f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
