#!/usr/bin/env python
"""iter19：在 base backbone 上做特征提取增强，从头对照择优（不叠已优化赢家）。

设计依据（本 goal 铁律 + 次级规律）：
- 有效规律：几何自适应/信息保真类模块有效（SPDConv +2.56pp、DySample），
  重加权类无效（注意力全家、VFL、NWD、IoU损失、iter18 HWD/CARAFE/RepConv 均掉点）。
- 次级规律：iter14/16/17/18 反复证明「在已优化的紧局部最优上再叠模块 → 破坏原增益」。
  故本轮回到 base backbone 从第一性重做特征提取，给结构改进留边际空间。

三个 backbone 特征提取模块（纯 PyTorch、免编译，见 install_backbone_modules.py）：
- DCNv2Conv : 可变形卷积下采样（采样点几何自适应），放深层下采样，对 scratches/inclusion 形变对症
- PKIC3k2   : 多核 Inception C3k2（并联多尺度核，无注意力），放浅中层，对 pitted/patches 尺度混杂对症
- DWRC3k2   : 空洞残差 C3k2（多路空洞聚合大上下文），放深层，对 crazing/rolled-in 弱纹理对症

候选矩阵（全部 base backbone + 标准 head，从 yolo26n.pt 迁移，保 end2end/reg_max=1）：
- y26n_i19_pki_shallow  : 浅中层 C3k2(层2/4) -> PKIC3k2
- y26n_i19_dwr_deep     : 深层 C3k2(层6/8)  -> DWRC3k2
- y26n_i19_pki_dwr      : 浅层PKI(2/4) + 深层DWR(6/8) 正交组合
- y26n_i19_dcnv2_deep   : 深层下采样 Conv(层5/7) -> DCNv2Conv

对照 = base（stage1 已有真值 val mAP50-95 0.3991），不重训。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py      # SPDConv/DySample（DCNv2 head 不用，但保持环境一致）
    python install_backbone_modules.py         # DCNv2Conv/PKIC3k2/DWRC3k2
    python -u train_yolo26_iter19_backbone.py

输出：runs_iter19_backbone_e250/ + generated_models_iter19_backbone_e250/
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
STAGE1_REPORT_JSON = ROOT / "runs_module_sweep_e250" / "sweep_report.json"
PROJECT = Path(os.environ.get("YOLO26_I19_PROJECT", ROOT / "runs_iter19_backbone_e250"))
GEN_DIR = Path(os.environ.get("YOLO26_I19_GEN_DIR", ROOT / "generated_models_iter19_backbone_e250"))
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


# ---------- backbone builders：在 BASE_BACKBONE 基础上做模块替换 ----------
def _clone_backbone() -> list[list[Any]]:
    out = []
    for row in sweep.BASE_BACKBONE:
        out.append([row[0], row[1], row[2], list(row[3])])
    return out


def bb_pki(layers=(2, 4)) -> list[list[Any]]:
    """把指定层的 C3k2 换成 PKIC3k2（多核 Inception）。"""
    bb = _clone_backbone()
    for i in layers:
        if bb[i][2] == "C3k2":
            bb[i][2] = "PKIC3k2"
    return bb


def bb_dwr(layers=(6, 8)) -> list[list[Any]]:
    """把指定层的 C3k2 换成 DWRC3k2（空洞残差）。"""
    bb = _clone_backbone()
    for i in layers:
        if bb[i][2] == "C3k2":
            bb[i][2] = "DWRC3k2"
    return bb


def bb_pki_dwr(pki_layers=(2, 4), dwr_layers=(6, 8)) -> list[list[Any]]:
    """浅层 PKI + 深层 DWR 正交组合。"""
    bb = _clone_backbone()
    for i in pki_layers:
        if bb[i][2] == "C3k2":
            bb[i][2] = "PKIC3k2"
    for i in dwr_layers:
        if bb[i][2] == "C3k2":
            bb[i][2] = "DWRC3k2"
    return bb


def bb_dcnv2(layers=(5, 7)) -> list[list[Any]]:
    """把指定层的 stride-2 下采样 Conv 换成 DCNv2Conv（可变形）。"""
    bb = _clone_backbone()
    for i in layers:
        if bb[i][2] == "Conv" and len(bb[i][3]) >= 3 and bb[i][3][2] == 2:
            bb[i][2] = "DCNv2Conv"
    return bb


def specs() -> dict[str, dict[str, Any]]:
    head = sweep.make_head()  # 标准 P3/P4/P5 head，不改
    return {
        "y26n_i19_pki_shallow_e250": {
            "doc": sweep.base_doc(bb_pki((2, 4)), head),
            "note": "base + 浅中层 C3k2(2/4)->PKIC3k2（多核 Inception 多尺度，无注意力）",
            "components": ["PKIC3k2@2,4"],
        },
        "y26n_i19_dwr_deep_e250": {
            "doc": sweep.base_doc(bb_dwr((6, 8)), head),
            "note": "base + 深层 C3k2(6/8)->DWRC3k2（空洞残差大上下文，直击弱纹理）",
            "components": ["DWRC3k2@6,8"],
        },
        "y26n_i19_pki_dwr_e250": {
            "doc": sweep.base_doc(bb_pki_dwr((2, 4), (6, 8)), head),
            "note": "base + 浅层PKI(2/4)+深层DWR(6/8) 正交组合",
            "components": ["PKIC3k2@2,4", "DWRC3k2@6,8"],
        },
        "y26n_i19_dcnv2_deep_e250": {
            "doc": sweep.base_doc(bb_dcnv2((5, 7)), head),
            "note": "base + 深层下采样 Conv(5/7)->DCNv2Conv（几何自适应，形变对症）",
            "components": ["DCNv2Conv@5,7"],
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
        "source": "iter19_backbone",
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
    import subprocess
    import sys
    here = Path(__file__).resolve().parent
    for installer in ("install_yolo26_exp_modules.py", "install_backbone_modules.py"):
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
    sweep.save_status("running", None, selected, {"stage": "iter19_backbone"})

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        spec = all_specs[name]
        sweep.WEIGHTS = OFFICIAL_WEIGHTS
        print(f"INIT_WEIGHTS {name} <- {sweep.WEIGHTS}")
        sweep.save_status("running", name, selected[pos:], {"stage": "iter19_backbone"})
        try:
            result = sweep.run_one(name, spec, args.epochs, args.batch, args.imgsz, args.seed, args.force)
        except Exception as exc:  # noqa: BLE001
            # 单候选崩溃只记录并跳过，不再拖垮整批（iter18 raise 拖死后续候选的教训）。
            report["experiments"][name] = {
                "status": "failed",
                "note": spec.get("note", ""),
                "components": spec.get("components", []),
                "error": repr(exc),
                "failed_at": now(),
            }
            sweep.save_report(report)
            print(f"CANDIDATE_FAILED {name}: {exc!r} -- continue to next")
            continue
        result["components"] = spec.get("components", [])
        report["experiments"][name] = result
        report["best_summary"] = summarize(report)
        sweep.save_report(report)
        print(
            "I19_RESULT", name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
            f"params={result.get('params')}",
        )

    report["best_summary"] = summarize(report)
    sweep.save_report(report)
    sweep.save_status("done", None, [], {"stage": "iter19_backbone"})
    print(f"I19_DONE report={REPORT_JSON} csv={REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
