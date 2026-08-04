#!/usr/bin/env python
"""YOLO26 250e stage3 轻量化搜索：在赢家 SPD_P3+DySample 上叠加 SlimNeck。

背景：stage2 赢家 = y26n_s2_spd_p3_dysample_e250
  val mAP50 0.7529 / mAP50-95 0.4246（+2.56pp）/ 推理 1.04ms，已过双阈值。
  但参数 2.499M 比基线 2.376M +5.2%，尚未命中"参数下降 ≥10%"的轻量化硬轴。

策略：把 neck 的 C3k2 换成 VoVGSCSP、下采样 Conv 换成 GSConv（SlimNeck），
在守住精度的前提下把参数压回基线以下。

候选（全部保持 end2end/reg_max=1）：
- y26n_s3_gsconv_neck_e250       : 赢家(SPD_P3+DySample) + neck C3k2->VoVGSCSP
- y26n_s3_gsconv_full_e250       : 赢家 + neck C3k2->VoVGSCSP + 下采样 Conv->GSConv
- y26n_s3_vovgscsp_only_e250     : 纯 base + neck C3k2->VoVGSCSP（隔离降参贡献对照）

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py   # 先确保 GSConv/VoVGSCSP 已注册
    python -u train_yolo26_stage3_gsconv.py

输出：
- runs_module_stage3_e250/combo_report.json / combo_results.csv / status.json
- generated_models_module_stage3_e250/*.yaml
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

PROJECT = Path(os.environ.get("YOLO26_STAGE3_PROJECT", ROOT / "runs_module_stage3_e250"))
GEN_DIR = Path(os.environ.get("YOLO26_STAGE3_GEN_DIR", ROOT / "generated_models_module_stage3_e250"))
REPORT_JSON = PROJECT / "combo_report.json"
REPORT_CSV = PROJECT / "combo_results.csv"
STATUS_JSON = PROJECT / "status.json"

BASELINE_NAME = "y26n_base_e250"

# stage3 改进口径：结构上继承赢家(SPD_P3+DySample)的候选，从赢家 best.pt 迁移，
# 让 SPD backbone + DySample 权重完美继承，只有换成 VoVGSCSP 的 neck 层重训——
# 这才是"在赢家基础上改进、观察是否更好"。而 vovgscsp_only 是纯 base backbone，
# 结构与赢家对不上，从赢家迁移反而丢预训练 backbone，故它保持从官方权重迁移。
WINNER_WEIGHTS = Path(os.environ.get(
    "YOLO26_WINNER_WEIGHTS",
    ROOT / "runs_module_combo2_e250" / "y26n_s2_spd_p3_dysample_e250" / "weights" / "best.pt",
))
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_gsconv_head(dysample: bool = False, gsconv_down: bool = False) -> list[list[Any]]:
    """YOLO26 P3/P4/P5 head，neck 的 C3k2 换成 VoVGSCSP，可选下采样换 GSConv。

    结构与 sweep.make_head 完全对齐（层号/Concat/Detect 输入一致），只替换 block 类型：
      C3k2 -> VoVGSCSP（进 base+repeat frozenset，n 会被 parse_model insert）
      下采样 Conv[c,3,2] -> GSConv[c,3,2]（仅 gsconv_down=True 时）
    """
    head: list[list[Any]] = []
    idx = 10

    def add(row: list[Any]) -> int:
        nonlocal idx
        head.append(row)
        idx += 1
        return idx

    up1 = ["DySample", [2, "lp", 4, False]] if dysample else ["nn.Upsample", [None, 2, "nearest"]]
    up2 = ["DySample", [2, "lp", 4, False]] if dysample else ["nn.Upsample", [None, 2, "nearest"]]
    down = "GSConv" if gsconv_down else "Conv"

    add([-1, 1, up1[0], up1[1]])
    add([[-1, 6], 1, "Concat", [1]])
    p4_lat = add([-1, 2, "VoVGSCSP", [512]])

    add([-1, 1, up2[0], up2[1]])
    add([[-1, 4], 1, "Concat", [1]])
    p3 = add([-1, 2, "VoVGSCSP", [256]])

    # 下采样：GSConv 用 [c,k,s]，普通 Conv 也用 [c,3,2]
    add([-1, 1, down, [256, 3, 2]])
    add([[-1, p4_lat], 1, "Concat", [1]])
    p4 = add([-1, 2, "VoVGSCSP", [512]])

    add([-1, 1, down, [512, 3, 2]])
    add([[-1, 10], 1, "Concat", [1]])
    p5 = add([-1, 1, "VoVGSCSP", [1024]])

    add([[p3, p4, p5], 1, "Detect", ["nc"]])
    return head


def stage3_specs() -> dict[str, dict[str, Any]]:
    """显式轻量化候选，均保持 end2end/reg_max=1。"""
    return {
        # 赢家 SPD_P3 + DySample，再把 neck C3k2 换成 VoVGSCSP。
        "y26n_s3_gsconv_neck_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(3), make_gsconv_head(dysample=True, gsconv_down=False)),
            "note": "赢家(SPD_P3+DySample) + neck C3k2->VoVGSCSP 降参",
            "components": ["SPDConv=p3", "DySample", "VoVGSCSP_neck"],
            # 结构上继承赢家 SPD_P3+DySample 并叠 SlimNeck；权重从官方 yolo26n.pt 迁移，
            # 与其余候选同起跑线，检验"结构改进"本身能否使结果变好。
        },
        # 更激进：连下采样 Conv 也换成 GSConv。
        "y26n_s3_gsconv_full_e250": {
            "doc": sweep.base_doc(sweep.spd_backbone(3), make_gsconv_head(dysample=True, gsconv_down=True)),
            "note": "赢家 + neck C3k2->VoVGSCSP + 下采样 Conv->GSConv",
            "components": ["SPDConv=p3", "DySample", "VoVGSCSP_neck", "GSConv_down"],
        },
        # 隔离对照：纯 base + VoVGSCSP neck，衡量 SlimNeck 单独的降参/精度影响。
        "y26n_s3_vovgscsp_only_e250": {
            "doc": sweep.base_doc(sweep.BASE_BACKBONE, make_gsconv_head(dysample=False, gsconv_down=False)),
            "note": "纯 base + neck C3k2->VoVGSCSP（隔离 SlimNeck 贡献对照）",
            "components": ["VoVGSCSP_neck"],
        },
        # 最激进纯轻量：纯 base + VoVGSCSP neck + 下采样 Conv->GSConv（无 SPD/DySample）。
        # 补齐 2x2 矩阵的缺格：= vovgscsp_only 再叠 GSConv 下采样 = gsconv_full 去掉 SPD+DySample。
        "y26n_s3_vovgscsp_gsdown_e250": {
            "doc": sweep.base_doc(sweep.BASE_BACKBONE, make_gsconv_head(dysample=False, gsconv_down=True)),
            "note": "纯 base + neck C3k2->VoVGSCSP + 下采样 Conv->GSConv（最激进纯轻量）",
            "components": ["VoVGSCSP_neck", "GSConv_down"],
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
        "source": "stage3",
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
    specs = stage3_specs()
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
    sweep.save_status("running", pending[0] if pending else None, pending, {"stage": "stage3_gsconv"})

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        sweep.save_status("running", name, selected[pos:], {"stage": "stage3_gsconv"})
        # 按候选设置迁移起点：gsconv_neck/full 从赢家 best.pt 继承 SPD+DySample 权重，
        # vovgscsp_only 结构与赢家不兼容（纯 base backbone），保持从官方 stock 权重迁移。
        # 注意：init_weights 显式为 None 时回落官方权重，且每轮都从 OFFICIAL 起算，
        # 避免上一候选把 sweep.WEIGHTS 改成赢家后污染本候选的 fallback。
        init_w = specs[name].get("init_weights") or OFFICIAL_WEIGHTS
        sweep.WEIGHTS = Path(init_w)
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
            sweep.save_status("failed", name, selected[pos:], {"stage": "stage3_gsconv", "error": repr(exc)})
            raise
        result["components"] = specs[name].get("components", [])
        result["init_weights"] = str(sweep.WEIGHTS)
        report["experiments"][name] = result
        report["best_summary"] = summarize(report)
        sweep.save_report(report)
        print(
            "STAGE3_RESULT", name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50={result['test']['map50']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
            f"params={result.get('params')}",
        )

    report["best_summary"] = summarize(report)
    sweep.save_report(report)
    sweep.save_status("done", None, [], {"stage": "stage3_gsconv"})
    print(f"STAGE3_DONE report={REPORT_JSON} csv={REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
