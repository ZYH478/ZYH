#!/usr/bin/env python
# Train vovgscsp_gsdown with swappable box regression loss (CIoU / Focaler-CIoU / WIoU v3).
# Base model fixed = vovgscsp_gsdown (BASE_BACKBONE + VoVGSCSP neck + GSConv down),
# only the IoU term of BboxLoss is swapped via runtime monkey-patch (iou_patch.py).
# Single-variable comparison: 3 runs, same structure, same official-weight start line.
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time
from typing import Any

import train_yolo26_module_sweep as sweep
import train_yolo26_stage3_gsconv as stage3
import iou_patch
from ultralytics.utils.loss import BboxLoss


ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
STAGE1_REPORT_JSON = ROOT / "runs_module_sweep_e250" / "sweep_report.json"
PROJECT = Path(os.environ.get("YOLO26_IOU_PROJECT", ROOT / "runs_iou_e250"))
GEN_DIR = Path(os.environ.get("YOLO26_IOU_GEN_DIR", ROOT / "generated_models_iou_e250"))
REPORT_JSON = PROJECT / "combo_report.json"
REPORT_CSV = PROJECT / "combo_results.csv"
STATUS_JSON = PROJECT / "status.json"
BASELINE_NAME = "y26n_base_e250"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))

# Keep the pristine official forward so each run resets before applying its own patch.
_ORIG_FORWARD = BboxLoss.forward


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def gsdown_doc() -> dict[str, Any]:
    # vovgscsp_gsdown: pure base backbone + VoVGSCSP neck + GSConv downsample (no SPD, no DySample).
    return sweep.base_doc(sweep.BASE_BACKBONE, stage3.make_gsconv_head(dysample=False, gsconv_down=True))


def iou_specs() -> dict[str, dict[str, Any]]:
    return {
        "y26n_gsdown_ciou_e250": {
            "doc": copy.deepcopy(gsdown_doc()),
            "iou": "ciou",
            "note": "vovgscsp_gsdown baseline CIoU (control, must reproduce ~0.4018)",
            "components": ["VoVGSCSP_neck", "GSConv_down", "CIoU"],
        },
        "y26n_gsdown_focaler_e250": {
            "doc": copy.deepcopy(gsdown_doc()),
            "iou": "focaler_ciou",
            "note": "vovgscsp_gsdown + Focaler-CIoU (focus hard samples)",
            "components": ["VoVGSCSP_neck", "GSConv_down", "FocalerCIoU"],
        },
        "y26n_gsdown_wiou_e250": {
            "doc": copy.deepcopy(gsdown_doc()),
            "iou": "wiou",
            "note": "vovgscsp_gsdown + WIoU v3 (dynamic non-monotonic focusing)",
            "components": ["VoVGSCSP_neck", "GSConv_down", "WIoUv3"],
        },
    }


def enrich(name: str, exp: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    def m(e: dict[str, Any], split: str, key: str) -> float:
        try:
            return float(e[split][key])
        except Exception:
            return float("nan")

    return {
        "name": name,
        "source": "iou",
        "val_map50": m(exp, "val", "map50"),
        "val_map50_95": m(exp, "val", "map50_95"),
        "test_map50": m(exp, "test", "map50"),
        "test_map50_95": m(exp, "test", "map50_95"),
        "delta_val_map50_95": m(exp, "val", "map50_95") - m(base, "val", "map50_95"),
        "delta_val_map50": m(exp, "val", "map50") - m(base, "val", "map50"),
        "params": exp.get("params"),
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    stage1 = load_json(STAGE1_REPORT_JSON)
    base = stage1.get("experiments", {}).get(BASELINE_NAME, {})
    rows = [
        enrich(name, exp, base)
        for name, exp in report.get("experiments", {}).items()
        if exp.get("status") == "done"
    ]
    rows = sorted(rows, key=lambda r: (r["val_map50_95"], r["val_map50"]), reverse=True)
    return {"updated_at": now(), "baseline": BASELINE_NAME, "ranked": rows, "best": rows[0] if rows else None}


def set_sweep_outputs() -> None:
    sweep.PROJECT = PROJECT
    sweep.GEN_DIR = GEN_DIR
    sweep.REPORT_JSON = REPORT_JSON
    sweep.REPORT_CSV = REPORT_CSV
    sweep.STATUS_JSON = STATUS_JSON


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=sweep.DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=sweep.DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=sweep.DEFAULT_IMGSZ)
    parser.add_argument("--seed", type=int, default=sweep.DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument("--only", default="", help="comma separated model names")
    args = parser.parse_args()

    PROJECT.mkdir(parents=True, exist_ok=True)
    specs = iou_specs()
    selected = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else list(specs)
    unknown = [x for x in selected if x not in specs]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")

    if args.dry_plan:
        for name in selected:
            print(name, "::", specs[name]["iou"], "::", specs[name]["note"])
        return 0

    set_sweep_outputs()
    sweep.ensure_modules()

    report = load_json(REPORT_JSON) or {"schema_version": 1, "created_at": now(), "experiments": {}}
    report["stage1_report_json"] = str(STAGE1_REPORT_JSON)
    report.setdefault("experiments", {})

    completed = {k for k, v in report.get("experiments", {}).items() if v.get("status") == "done"}
    pending = [x for x in selected if args.force or x not in completed]
    sweep.save_status("running", pending[0] if pending else None, pending, {"stage": "iou"})

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        spec = specs[name]
        # Reset to pristine official forward, then apply this run's IoU patch (ciou = no patch).
        BboxLoss.forward = _ORIG_FORWARD
        active = iou_patch.patch_bbox_loss(spec["iou"])
        sweep.WEIGHTS = OFFICIAL_WEIGHTS
        print(f"IOU_TYPE {name} <- {active}  weights={sweep.WEIGHTS}")
        sweep.save_status("running", name, selected[pos:], {"stage": "iou", "iou": spec["iou"]})
        try:
            result = sweep.run_one(name, spec, args.epochs, args.batch, args.imgsz, args.seed, args.force)
        except Exception as exc:  # noqa: BLE001
            report["experiments"][name] = {
                "status": "failed",
                "note": spec.get("note", ""),
                "components": spec.get("components", []),
                "error": repr(exc),
                "failed_at": now(),
            }
            sweep.save_report(report)
            sweep.save_status("failed", name, selected[pos:], {"stage": "iou", "error": repr(exc)})
            raise
        result["components"] = spec.get("components", [])
        result["iou"] = spec["iou"]
        report["experiments"][name] = result
        report["best_summary"] = summarize(report)
        sweep.save_report(report)
        print(
            "IOU_RESULT", name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50={result['test']['map50']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
            f"params={result.get('params')}",
        )

    # Restore official forward at the end so nothing lingers patched.
    BboxLoss.forward = _ORIG_FORWARD
    report["best_summary"] = summarize(report)
    sweep.save_report(report)
    sweep.save_status("done", None, [], {"stage": "iou"})
    print(f"IOU_DONE report={REPORT_JSON} csv={REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
