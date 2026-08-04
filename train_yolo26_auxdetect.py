#!/usr/bin/env python
"""YOLO26 AuxDetect 消融：把 6 个已定型模型的 Detect 换成 AuxDetect 重训 250e。

背景：stage1/2/3 已产出 6 个模型（base 基线 / 赢家 SPD_P3+DySample / 4 个 stage3
轻量化候选）。owner 要求把每个模型的检测头 Detect 换成轻量 AuxDetect（继承 Detect，
训练时加 aux_cv2/aux_cv3 辅助分支，total = o2m*L_o2m + o2o*L_o2o + 0.25*L_aux，
推理/fuse 时丢弃辅助分支，零额外成本），观察辅助监督是否提升精度。

6 个模型结构（与非 aux 版本逐层一致，仅 head 最后一行 Detect->AuxDetect）：
- base_aux             : BASE_BACKBONE            + make_head()
- spd_p3_dysample_aux  : spd_backbone(3)          + make_head(dysample=True)
- gsconv_neck_aux      : spd_backbone(3)          + make_gsconv_head(dysample=True,  gsconv_down=False)
- gsconv_full_aux      : spd_backbone(3)          + make_gsconv_head(dysample=True,  gsconv_down=True)
- vovgscsp_only_aux    : BASE_BACKBONE            + make_gsconv_head(dysample=False, gsconv_down=False)
- vovgscsp_gsdown_aux  : BASE_BACKBONE            + make_gsconv_head(dysample=False, gsconv_down=True)

全部从官方 yolo26n.pt 迁移（同起跑线），保持 end2end/reg_max=1。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py   # SPDConv/DySample
    python install_gsconv_modules.py        # GSConv/VoVGSCSP
    python install_auxdetect.py             # AuxDetect + AuxE2ELoss
    python -u train_yolo26_auxdetect.py

输出：
- runs_auxdetect_e250/combo_report.json / combo_results.csv / status.json
- generated_models_auxdetect_e250/*.yaml
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


ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
STAGE1_PROJECT = Path(os.environ.get("YOLO26_STAGE1_PROJECT", ROOT / "runs_module_sweep_e250"))
STAGE1_REPORT_JSON = STAGE1_PROJECT / "sweep_report.json"

PROJECT = Path(os.environ.get("YOLO26_AUX_PROJECT", ROOT / "runs_auxdetect_e250"))
GEN_DIR = Path(os.environ.get("YOLO26_AUX_GEN_DIR", ROOT / "generated_models_auxdetect_e250"))
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


def to_aux(head: list[list[Any]]) -> list[list[Any]]:
    """把 head 里的 Detect 换成 AuxDetect（深拷贝，不改原 head）。"""
    out = copy.deepcopy(head)
    for row in out:
        if row[2] == "Detect":
            row[2] = "AuxDetect"
    return out


def aux_specs() -> dict[str, dict[str, Any]]:
    """6 个模型的 AuxDetect 版本，均保持 end2end/reg_max=1，从官方权重迁移。"""
    return {
        "y26n_base_aux_e250": {
            "doc": sweep.base_doc(sweep.BASE_BACKBONE, to_aux(sweep.make_head())),
            "note": "基线 base + AuxDetect",
            "components": ["AuxDetect"],
        },
        "y26n_spd_p3_dysample_aux_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(3), to_aux(sweep.make_head(dysample=True))),
            "note": "赢家 SPD_P3+DySample + AuxDetect",
            "components": ["SPDConv=p3", "DySample", "AuxDetect"],
        },
        "y26n_gsconv_neck_aux_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(3), to_aux(stage3.make_gsconv_head(dysample=True, gsconv_down=False))),
            "note": "gsconv_neck + AuxDetect",
            "components": ["SPDConv=p3", "DySample", "VoVGSCSP_neck", "AuxDetect"],
        },
        "y26n_gsconv_full_aux_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(3), to_aux(stage3.make_gsconv_head(dysample=True, gsconv_down=True))),
            "note": "gsconv_full + AuxDetect",
            "components": ["SPDConv=p3", "DySample", "VoVGSCSP_neck", "GSConv_down", "AuxDetect"],
        },
        "y26n_vovgscsp_only_aux_e250": {
            "doc": sweep.base_doc(sweep.BASE_BACKBONE, to_aux(stage3.make_gsconv_head(dysample=False, gsconv_down=False))),
            "note": "vovgscsp_only + AuxDetect",
            "components": ["VoVGSCSP_neck", "AuxDetect"],
        },
        "y26n_vovgscsp_gsdown_aux_e250": {
            "doc": sweep.base_doc(sweep.BASE_BACKBONE, to_aux(stage3.make_gsconv_head(dysample=False, gsconv_down=True))),
            "note": "vovgscsp_gsdown + AuxDetect",
            "components": ["VoVGSCSP_neck", "GSConv_down", "AuxDetect"],
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
        "source": "auxdetect",
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

    PROJECT.mkdir(parents=True, exist_ok=True)
    specs = aux_specs()
    selected = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else list(specs)
    unknown = [x for x in selected if x not in specs]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")

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
    sweep.save_status("running", pending[0] if pending else None, pending, {"stage": "auxdetect"})

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        sweep.save_status("running", name, selected[pos:], {"stage": "auxdetect"})
        # 全部从官方 yolo26n.pt 迁移，同起跑线；改进纯靠 AuxDetect 辅助监督。
        sweep.WEIGHTS = OFFICIAL_WEIGHTS
        print(f"INIT_WEIGHTS {name} <- {sweep.WEIGHTS}")
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
            sweep.save_status("failed", name, selected[pos:], {"stage": "auxdetect", "error": repr(exc)})
            raise
        result["components"] = specs[name].get("components", [])
        report["experiments"][name] = result
        report["best_summary"] = summarize(report)
        sweep.save_report(report)
        print(
            "AUX_RESULT", name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50={result['test']['map50']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
            f"params={result.get('params')}",
        )

    report["best_summary"] = summarize(report)
    sweep.save_report(report)
    sweep.save_status("done", None, [], {"stage": "auxdetect"})
    print(f"AUX_DONE report={REPORT_JSON} csv={REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
