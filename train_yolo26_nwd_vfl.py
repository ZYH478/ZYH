#!/usr/bin/env python
"""在 vovgscsp_gsdown 上做零参数精度改进搜索：NWD 标签分配 / Varifocal 分类损失 / 组合。

基础模型固定 = vovgscsp_gsdown（BASE_BACKBONE + VoVGSCSP neck + GSConv down，
无 SPD / DySample / AuxDetect），结构完全不变（零新增参数/FLOPs），只在训练期换
「标签分配相似度」和「分类损失」两处逻辑。全部从官方 yolo26n.pt 迁移（同起跑线），
改进纯靠算法。设计对齐 train_yolo26_iou.py：单进程循环，每候选前 reset 再 patch。

候选（含 CIoU/BCE 官方对照，用于复现 gsdown ~0.4077 基准）：
- y26n_gsdown_ctrl_e250     : 官方（对照，理应复现 gsdown baseline）
- y26n_gsdown_nwd_e250      : + NWD 标签分配（ratio=0.5）
- y26n_gsdown_vfl_e250      : + Varifocal 分类损失
- y26n_gsdown_nwd_vfl_e250  : + NWD + VFL 组合

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py   # SPDConv/DySample（gsdown 用 GSConv，也需 exp 基础）
    python install_gsconv_modules.py        # GSConv/VoVGSCSP
    python -u train_yolo26_nwd_vfl.py

输出：
- runs_nwd_vfl_e250/combo_report.json / combo_results.csv / status.json
- generated_models_nwd_vfl_e250/*.yaml
"""
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
import nwd_vfl_patch as nv


ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
STAGE1_REPORT_JSON = ROOT / "runs_module_sweep_e250" / "sweep_report.json"
PROJECT = Path(os.environ.get("YOLO26_NWD_PROJECT", ROOT / "runs_nwd_vfl_e250"))
GEN_DIR = Path(os.environ.get("YOLO26_NWD_GEN_DIR", ROOT / "generated_models_nwd_vfl_e250"))
REPORT_JSON = PROJECT / "combo_report.json"
REPORT_CSV = PROJECT / "combo_results.csv"
STATUS_JSON = PROJECT / "status.json"
BASELINE_NAME = "y26n_base_e250"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def gsdown_doc() -> dict[str, Any]:
    # vovgscsp_gsdown: 纯 base backbone + VoVGSCSP neck + GSConv 下采样（无 SPD / DySample）。
    return sweep.base_doc(sweep.BASE_BACKBONE, stage3.make_gsconv_head(dysample=False, gsconv_down=True))


def specs() -> dict[str, dict[str, Any]]:
    return {
        "y26n_gsdown_ctrl_e250": {
            "doc": copy.deepcopy(gsdown_doc()),
            "patch": {},
            "note": "vovgscsp_gsdown 官方对照（CIoU+BCE，须复现 ~0.4077）",
            "components": ["VoVGSCSP_neck", "GSConv_down"],
        },
        "y26n_gsdown_nwd_e250": {
            "doc": copy.deepcopy(gsdown_doc()),
            "patch": {"nwd": True, "nwd_ratio": 0.5, "nwd_const": 0.1},
            "note": "vovgscsp_gsdown + NWD 标签分配（overlaps=0.5*CIoU+0.5*NWD）",
            "components": ["VoVGSCSP_neck", "GSConv_down", "NWD"],
        },
        "y26n_gsdown_vfl_e250": {
            "doc": copy.deepcopy(gsdown_doc()),
            "patch": {"vfl": True, "vfl_gamma": 2.0, "vfl_alpha": 0.75},
            "note": "vovgscsp_gsdown + Varifocal 分类损失",
            "components": ["VoVGSCSP_neck", "GSConv_down", "VFL"],
        },
        "y26n_gsdown_nwd_vfl_e250": {
            "doc": copy.deepcopy(gsdown_doc()),
            "patch": {"nwd": True, "nwd_ratio": 0.5, "nwd_const": 0.1, "vfl": True, "vfl_gamma": 2.0, "vfl_alpha": 0.75},
            "note": "vovgscsp_gsdown + NWD + VFL 组合",
            "components": ["VoVGSCSP_neck", "GSConv_down", "NWD", "VFL"],
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
        "source": "nwd_vfl",
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
    parser.add_argument("--only", default="", help="逗号分隔，只跑指定模型")
    args = parser.parse_args()

    all_specs = specs()
    selected = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else list(all_specs)
    unknown = [x for x in selected if x not in all_specs]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")

    if args.dry_plan:
        for name in selected:
            print(name, "::", all_specs[name]["patch"], "::", all_specs[name]["note"])
        return 0

    PROJECT.mkdir(parents=True, exist_ok=True)
    set_sweep_outputs()
    sweep.ensure_modules()

    report = load_json(REPORT_JSON) or {"schema_version": 1, "created_at": now(), "experiments": {}}
    report["stage1_report_json"] = str(STAGE1_REPORT_JSON)
    report.setdefault("experiments", {})

    completed = {k for k, v in report.get("experiments", {}).items() if v.get("status") == "done"}
    pending = [x for x in selected if args.force or x not in completed]
    sweep.save_status("running", pending[0] if pending else None, pending, {"stage": "nwd_vfl"})

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        spec = all_specs[name]
        # 每候选前 reset 到官方，再按 spec 打补丁（空 patch = 官方对照）。
        nv.reset_all()
        active = nv.patch_from_spec(spec["patch"])
        sweep.WEIGHTS = OFFICIAL_WEIGHTS
        print(f"PATCH {name} <- {active}  weights={sweep.WEIGHTS}")
        sweep.save_status("running", name, selected[pos:], {"stage": "nwd_vfl", "patch": spec["patch"]})
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
            sweep.save_status("failed", name, selected[pos:], {"stage": "nwd_vfl", "error": repr(exc)})
            raise
        result["components"] = spec.get("components", [])
        result["patch"] = spec["patch"]
        result["patch_active"] = active
        report["experiments"][name] = result
        report["best_summary"] = summarize(report)
        sweep.save_report(report)
        print(
            "NWD_RESULT", name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50={result['test']['map50']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
            f"params={result.get('params')}",
        )

    # 收尾 reset，避免残留补丁污染后续进程。
    nv.reset_all()
    report["best_summary"] = summarize(report)
    sweep.save_report(report)
    sweep.save_status("done", None, [], {"stage": "nwd_vfl"})
    print(f"NWD_DONE report={REPORT_JSON} csv={REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
