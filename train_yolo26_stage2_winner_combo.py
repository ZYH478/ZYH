#!/usr/bin/env python
"""YOLO26 250e stage2 winner-combo search for NEU-DET.

区别于 `train_yolo26_stage2_combo.py` 的自动 best_attn 逻辑：本脚本用**显式**候选列表，
只聚焦 stage1 真正超过 250e baseline 的三个 winner —— SPD_P2 / SPD_P3 / DySample ——
的正交组合。stage1 已证明所有注意力（SimAM/EMA/CoordAtt/LSKA）都拉低 mAP50-95，
因此不再铺任何纯注意力位置变体。

winner（stage1 fresh 250e，vs y26n_base_e250 的 val mAP50-95）：
- SPD_P2   +0.76pp
- DySample +0.74pp
- SPD_P3   +0.66pp

组合逻辑：SPD 改 backbone 下采样、DySample 改 head 上采样，两者正交，最可能叠加增益。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python -u train_yolo26_stage2_winner_combo.py

输出：
- runs_module_combo2_e250/combo_report.json
- runs_module_combo2_e250/combo_results.csv
- runs_module_combo2_e250/status.json
- generated_models_module_combo2_e250/*.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import train_yolo26_module_sweep as sweep


ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
STAGE1_PROJECT = Path(os.environ.get("YOLO26_STAGE1_PROJECT", ROOT / "runs_module_sweep_e250"))
STAGE1_REPORT_JSON = STAGE1_PROJECT / "sweep_report.json"

PROJECT = Path(os.environ.get("YOLO26_STAGE2W_PROJECT", ROOT / "runs_module_combo2_e250"))
GEN_DIR = Path(os.environ.get("YOLO26_STAGE2W_GEN_DIR", ROOT / "generated_models_module_combo2_e250"))
REPORT_JSON = PROJECT / "combo_report.json"
REPORT_CSV = PROJECT / "combo_results.csv"
STATUS_JSON = PROJECT / "status.json"

BASELINE_NAME = "y26n_base_e250"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def winner_combo_specs() -> dict[str, dict[str, Any]]:
    """显式 winner 组合，由稳到叠加，全部保持 end2end/reg_max=1。"""
    return {
        # 两处下采样都换成信息保真的 SPDConv（backbone layer1 P2/4 + layer3 P3/8）。
        "y26n_s2_spd_p2p3_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(1, 3), sweep.make_head()),
            "note": "SPDConv(P2/4 + P3/8) 双下采样保真组合",
            "components": ["SPDConv=p2", "SPDConv=p3"],
        },
        # backbone SPD + head DySample，两侧正交。
        "y26n_s2_spd_p2_dysample_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(1), sweep.make_head(dysample=True)),
            "note": "SPDConv(P2/4) + DySample 上采样",
            "components": ["SPDConv=p2", "DySample"],
        },
        "y26n_s2_spd_p3_dysample_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(3), sweep.make_head(dysample=True)),
            "note": "SPDConv(P3/8) + DySample 上采样",
            "components": ["SPDConv=p3", "DySample"],
        },
        # 三个 winner 全叠。
        "y26n_s2_spd_p2p3_dysample_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(1, 3), sweep.make_head(dysample=True)),
            "note": "SPDConv(P2/4 + P3/8) + DySample 三 winner 全叠",
            "components": ["SPDConv=p2", "SPDConv=p3", "DySample"],
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
        "source": "stage2w",
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
    args = parser.parse_args()

    PROJECT.mkdir(parents=True, exist_ok=True)
    specs = winner_combo_specs()
    selected = list(specs)

    if args.dry_plan:
        for name in selected:
            print(name, "::", specs[name]["note"], "::", specs[name]["components"])
        return 0

    set_sweep_outputs()
    sweep.ensure_modules()

    report = load_json(REPORT_JSON) or {"schema_version": 1, "created_at": now(), "experiments": {}}
    report["stage1_report_json"] = str(STAGE1_REPORT_JSON)
    report.setdefault("experiments", {})

    completed = {k for k, v in report.get("experiments", {}).items() if v.get("status") == "done"}
    pending = [x for x in selected if args.force or x not in completed]
    sweep.save_status("running", pending[0] if pending else None, pending, {"stage": "stage2_winner_combo"})

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        sweep.save_status("running", name, selected[pos:], {"stage": "stage2_winner_combo"})
        try:
            result = sweep.run_one(name, specs[name], args.epochs, args.batch, args.imgsz, args.seed, args.force)
        except Exception as exc:  # noqa: BLE001
            report["experiments"][name] = {
                "status": "failed",
                "note": specs[name].get("note", ""),
                "components": specs[name].get("components", []),
                "error": repr(exc),
                "failed_at": now(),
            }
            sweep.save_report(report)
            sweep.save_status("failed", name, selected[pos:], {"stage": "stage2_winner_combo", "error": repr(exc)})
            raise
        result["components"] = specs[name].get("components", [])
        report["experiments"][name] = result
        report["best_summary"] = summarize(report)
        sweep.save_report(report)
        print(
            "STAGE2W_RESULT", name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50={result['test']['map50']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
        )

    report["best_summary"] = summarize(report)
    sweep.save_report(report)
    sweep.save_status("done", None, [], {"stage": "stage2_winner_combo"})
    print(f"STAGE2W_DONE report={REPORT_JSON} csv={REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
