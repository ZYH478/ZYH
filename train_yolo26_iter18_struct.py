#!/usr/bin/env python
"""iter18 结构改进搜索：在两个起点上叠加信息保真类结构模块提精度。

两个起点（owner 定：都跑，对照择优）：
- gsdown  = 轻量交付赢家 vovgscsp_gsdown（1.936M/-18.5%，val 0.4077）
            = 纯 base backbone + VoVGSCSP neck + GSConv 下采样
- winner  = 精度赢家 SPD_P3+DySample（2.499M，val 0.4246/+2.56pp）
            = spd_backbone(3) + make_head(dysample=True)

三个结构改进（契合本 goal 唯一有效规律「信息保真类有效」，均保住/接近起点轻量）：
- HWD    : 把下采样换成 Haar 小波下采样（信息保真，保留高频纹理子带，直击 crazing）
- CARAFE : 把上采样换成内容感知上采样（DySample 同族）
- RepConv: 把 neck 关键 3x3 换成 RepConv（训练多分支，fuse 后推理零成本）

全部从官方 yolo26n.pt 迁移（同起跑线，改进纯靠网络结构），保持 end2end/reg_max=1。

对照 ctrl（gsdown 0.4077 / winner 0.4246）已有真值，本脚本默认不重训，用 --with-ctrl 可补。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py   # SPDConv/DySample
    python install_gsconv_modules.py        # GSConv/VoVGSCSP
    python install_hwd_carafe.py            # HWD/CARAFE
    python -u train_yolo26_iter18_struct.py

输出：runs_iter18_struct_e250/ + generated_models_iter18_struct_e250/
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
STAGE1_REPORT_JSON = ROOT / "runs_module_sweep_e250" / "sweep_report.json"
PROJECT = Path(os.environ.get("YOLO26_I18_PROJECT", ROOT / "runs_iter18_struct_e250"))
GEN_DIR = Path(os.environ.get("YOLO26_I18_GEN_DIR", ROOT / "generated_models_iter18_struct_e250"))
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


# ---------- head builders：在 stage3.make_gsconv_head / sweep.make_head 基础上做结构替换 ----------

def gsdown_head(down: str = "GSConv", up: str | None = None, repconv: bool = False) -> list[list[Any]]:
    """gsdown 起点的 head，可把下采样/上采样/block 类型替换成结构改进模块。

    起点默认：VoVGSCSP neck + GSConv 下采样 + nn.Upsample（= vovgscsp_gsdown）。
    - down='HWD'   : 下采样换 HWD（信息保真）
    - up='CARAFE'  : 上采样换 CARAFE
    - repconv=True : neck 下采样 3x3 换 RepConv（仍 stride-2，训练多分支）
    """
    head: list[list[Any]] = []
    idx = 10

    def add(row: list[Any]) -> int:
        nonlocal idx
        head.append(row)
        idx += 1
        return idx

    def up_row():
        if up == "CARAFE":
            return ["CARAFE", [2]]
        return ["nn.Upsample", [None, 2, "nearest"]]

    def down_mod():
        if down == "HWD":
            return "HWD"
        if repconv:
            return "RepConv"
        return down  # GSConv (默认) 或 Conv

    add(list([-1, 1]) + up_row())
    add([[-1, 6], 1, "Concat", [1]])
    p4_lat = add([-1, 2, "VoVGSCSP", [512]])

    add(list([-1, 1]) + up_row())
    add([[-1, 4], 1, "Concat", [1]])
    p3 = add([-1, 2, "VoVGSCSP", [256]])

    add([-1, 1, down_mod(), [256, 3, 2]])
    add([[-1, p4_lat], 1, "Concat", [1]])
    p4 = add([-1, 2, "VoVGSCSP", [512]])

    add([-1, 1, down_mod(), [512, 3, 2]])
    add([[-1, 10], 1, "Concat", [1]])
    p5 = add([-1, 1, "VoVGSCSP", [1024]])

    add([[p3, p4, p5], 1, "Detect", ["nc"]])
    return head


def winner_head(down: str = "Conv", up: str = "DySample", repconv: bool = False) -> list[list[Any]]:
    """winner(SPD_P3+DySample) 起点的 head：标准 C3k2 neck + DySample 上采样 + Conv 下采样。

    - down='HWD'   : 下采样换 HWD
    - up='CARAFE'  : 上采样 DySample 换 CARAFE（同族对照）
    - repconv=True : 下采样换 RepConv
    """
    head: list[list[Any]] = []
    idx = 10

    def add(row: list[Any]) -> int:
        nonlocal idx
        head.append(row)
        idx += 1
        return idx

    def up_row():
        if up == "CARAFE":
            return ["CARAFE", [2]]
        if up == "DySample":
            return ["DySample", [2, "lp", 4, False]]
        return ["nn.Upsample", [None, 2, "nearest"]]

    def down_mod():
        if down == "HWD":
            return "HWD"
        if repconv:
            return "RepConv"
        return "Conv"

    add(list([-1, 1]) + up_row())
    add([[-1, 6], 1, "Concat", [1]])
    p4_lat = add([-1, 2, "C3k2", [512, False]])

    add(list([-1, 1]) + up_row())
    add([[-1, 4], 1, "Concat", [1]])
    p3 = add([-1, 2, "C3k2", [256, False]])

    add([-1, 1, down_mod(), [256, 3, 2]])
    add([[-1, p4_lat], 1, "Concat", [1]])
    p4 = add([-1, 2, "C3k2", [512, False]])

    add([-1, 1, down_mod(), [512, 3, 2]])
    add([[-1, 10], 1, "Concat", [1]])
    p5 = add([-1, 1, "C3k2", [1024, True]])

    add([[p3, p4, p5], 1, "Detect", ["nc"]])
    return head


def specs() -> dict[str, dict[str, Any]]:
    G = sweep.BASE_BACKBONE
    W = lambda: sweep.spd_backbone(3)
    return {
        # ---- gsdown 起点（1.936M，冲『轻量前提下提精度』）----
        "y26n_i18_gsdown_hwd_e250": {
            "doc": sweep.base_doc(G, gsdown_head(down="HWD")),
            "note": "gsdown + HWD 下采样（Haar 小波信息保真，保高频纹理）",
            "components": ["VoVGSCSP_neck", "HWD_down"],
        },
        "y26n_i18_gsdown_carafe_e250": {
            "doc": sweep.base_doc(G, gsdown_head(up="CARAFE")),
            "note": "gsdown + CARAFE 上采样（内容感知，DySample 同族）",
            "components": ["VoVGSCSP_neck", "GSConv_down", "CARAFE_up"],
        },
        "y26n_i18_gsdown_repconv_e250": {
            "doc": sweep.base_doc(G, gsdown_head(repconv=True)),
            "note": "gsdown + RepConv 下采样（训练多分支，fuse 后零成本）",
            "components": ["VoVGSCSP_neck", "RepConv_down"],
        },
        # ---- winner 起点（2.499M，纯冲精度新高）----
        "y26n_i18_winner_hwd_e250": {
            "doc": sweep.base_doc(W(), winner_head(down="HWD")),
            "note": "winner(SPD_P3+DySample) + HWD 下采样",
            "components": ["SPDConv=p3", "DySample", "HWD_down"],
        },
        "y26n_i18_winner_carafe_e250": {
            "doc": sweep.base_doc(W(), winner_head(up="CARAFE")),
            "note": "winner + 上采样 DySample->CARAFE（同族对照）",
            "components": ["SPDConv=p3", "CARAFE_up"],
        },
        "y26n_i18_winner_repconv_e250": {
            "doc": sweep.base_doc(W(), winner_head(repconv=True)),
            "note": "winner(SPD_P3+DySample) + RepConv 下采样",
            "components": ["SPDConv=p3", "DySample", "RepConv_down"],
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
        "source": "iter18_struct",
        "val_map50": m(exp, "val", "map50"),
        "val_map50_95": m(exp, "val", "map50_95"),
        "test_map50": m(exp, "test", "map50"),
        "test_map50_95": m(exp, "test", "map50_95"),
        "delta_val_map50_95": m(exp, "val", "map50_95") - m(base, "val", "map50_95"),
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


def ensure_all_modules() -> None:
    """确保 SPDConv/DySample + GSConv/VoVGSCSP + HWD/CARAFE 全部注册。"""
    import subprocess
    import sys
    here = Path(__file__).resolve().parent
    for installer in ("install_yolo26_exp_modules.py", "install_gsconv_modules.py", "install_hwd_carafe.py"):
        subprocess.run([sys.executable, str(here / installer)], check=True)


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
            print(name, "::", all_specs[name]["note"], "::", all_specs[name]["components"])
        return 0

    PROJECT.mkdir(parents=True, exist_ok=True)
    set_sweep_outputs()
    ensure_all_modules()

    report = load_json(REPORT_JSON) or {"schema_version": 1, "created_at": now(), "experiments": {}}
    report["stage1_report_json"] = str(STAGE1_REPORT_JSON)
    report.setdefault("experiments", {})

    completed = {k for k, v in report.get("experiments", {}).items() if v.get("status") == "done"}
    pending = [x for x in selected if args.force or x not in completed]
    sweep.save_status("running", pending[0] if pending else None, pending, {"stage": "iter18_struct"})

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        spec = all_specs[name]
        sweep.WEIGHTS = OFFICIAL_WEIGHTS
        print(f"INIT_WEIGHTS {name} <- {sweep.WEIGHTS}")
        sweep.save_status("running", name, selected[pos:], {"stage": "iter18_struct"})
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
            sweep.save_status("failed", name, selected[pos:], {"stage": "iter18_struct", "error": repr(exc)})
            raise
        result["components"] = spec.get("components", [])
        report["experiments"][name] = result
        report["best_summary"] = summarize(report)
        sweep.save_report(report)
        print(
            "I18_RESULT", name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
            f"params={result.get('params')}",
        )

    report["best_summary"] = summarize(report)
    sweep.save_report(report)
    sweep.save_status("done", None, [], {"stage": "iter18_struct"})
    print(f"I18_DONE report={REPORT_JSON} csv={REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
