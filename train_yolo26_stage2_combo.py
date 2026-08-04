#!/usr/bin/env python
"""YOLO26 250e stage2 combination search for NEU-DET.

本脚本接在 `train_yolo26_module_sweep.py --stage stage1` 后面运行：

1. 等待 stage1 完成（可选）。
2. 读取 stage1 的 fresh val/test 指标，以 `y26n_base_e250` 为唯一基线。
3. 只基于“优于基线”或“几乎不回退”的模块生成有限组合与位置/数量细化候选。
4. 继续按 owner 固定口径 `epochs=250 / batch=32 / imgsz=640 / seed=0` 串行训练。

远程用法：

    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python -u train_yolo26_stage2_combo.py --wait-stage1 --max-variants 10

输出：

- `runs_module_combo_e250/combo_report.json`
- `runs_module_combo_e250/combo_results.csv`
- `runs_module_combo_e250/status.json`
- `generated_models_module_combo_e250/*.yaml`
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import train_yolo26_module_sweep as sweep


# =========================
# Top-level experiment config
# =========================
ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
STAGE1_PROJECT = Path(os.environ.get("YOLO26_STAGE1_PROJECT", ROOT / "runs_module_sweep_e250"))
STAGE1_REPORT_JSON = STAGE1_PROJECT / "sweep_report.json"
STAGE1_STATUS_JSON = STAGE1_PROJECT / "status.json"

STAGE2_PROJECT = Path(os.environ.get("YOLO26_STAGE2_PROJECT", ROOT / "runs_module_combo_e250"))
STAGE2_GEN_DIR = Path(os.environ.get("YOLO26_STAGE2_GEN_DIR", ROOT / "generated_models_module_combo_e250"))
STAGE2_REPORT_JSON = STAGE2_PROJECT / "combo_report.json"
STAGE2_REPORT_CSV = STAGE2_PROJECT / "combo_results.csv"
STAGE2_STATUS_JSON = STAGE2_PROJECT / "status.json"

BASELINE_NAME = "y26n_base_e250"
EPS = 1e-6


STAGE1_ATTRS: dict[str, dict[str, Any]] = {
    "y26n_dysample_e250": {"family": "dysample", "dysample": True},
    "y26n_simam_p3_e250": {"family": "attn", "attn": "SimAM", "positions": ("P3",)},
    "y26n_simam_p3p4_e250": {"family": "attn", "attn": "SimAM", "positions": ("P3", "P4")},
    "y26n_simam_p3p4p5_e250": {"family": "attn", "attn": "SimAM", "positions": ("P3", "P4", "P5")},
    "y26n_ema_p3p4p5_e250": {"family": "attn", "attn": "EMA", "positions": ("P3", "P4", "P5")},
    "y26n_ca_p3p4p5_e250": {"family": "attn", "attn": "CoordAtt", "positions": ("P3", "P4", "P5")},
    "y26n_lska_p4p5_e250": {"family": "attn", "attn": "LSKA", "positions": ("P4", "P5")},
    "y26n_spd_p2_e250": {"family": "spd", "spd_layers": (1,), "spd_tag": "p2"},
    "y26n_spd_p3_e250": {"family": "spd", "spd_layers": (3,), "spd_tag": "p3"},
    "y26n_p2_e250": {"family": "p2head", "p2head": True},
    "y26n_a2c2fbb_simam_p3p4p5_e250": {
        "family": "a2c2fbb_attn",
        "backbone": "a2c2f",
        "attn": "SimAM",
        "positions": ("P3", "P4", "P5"),
    },
}

ATTN_TAG = {"SimAM": "simam", "EMA": "ema", "CoordAtt": "ca", "LSKA": "lska"}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def stage2_status(status: str, current: str | None, pending: list[str], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "status": status,
        "current": current,
        "pending": pending,
        "updated_at": now(),
        "stage1_report_json": str(STAGE1_REPORT_JSON),
        "report_json": str(STAGE2_REPORT_JSON),
        "report_csv": str(STAGE2_REPORT_CSV),
    }
    if extra:
        payload.update(extra)
    write_json(STAGE2_STATUS_JSON, payload)


def wait_for_stage1(poll_seconds: int, max_wait_hours: float) -> None:
    deadline = time.time() + max_wait_hours * 3600
    while True:
        status = load_json(STAGE1_STATUS_JSON)
        report = load_json(STAGE1_REPORT_JSON)
        state = status.get("status", "missing")
        done = state == "done" and BASELINE_NAME in report.get("experiments", {})
        stage2_status(
            "waiting_stage1",
            status.get("current"),
            status.get("pending", []),
            {"stage1_status": state, "stage1_updated_at": status.get("updated_at")},
        )
        if done:
            return
        if state == "failed":
            raise RuntimeError(f"stage1 failed: {status}")
        if time.time() >= deadline:
            raise TimeoutError(f"stage1 not done after {max_wait_hours}h: {status}")
        time.sleep(poll_seconds)


def metric(exp: dict[str, Any], split: str, key: str) -> float:
    try:
        return float(exp[split][key])
    except Exception:
        return float("nan")


def enrich_metrics(name: str, exp: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    val95 = metric(exp, "val", "map50_95")
    val50 = metric(exp, "val", "map50")
    test95 = metric(exp, "test", "map50_95")
    test50 = metric(exp, "test", "map50")
    b_val95 = metric(base, "val", "map50_95")
    b_val50 = metric(base, "val", "map50")
    b_test95 = metric(base, "test", "map50_95")
    b_test50 = metric(base, "test", "map50")
    return {
        "name": name,
        "note": exp.get("note", ""),
        "status": exp.get("status"),
        "val_map50_95": val95,
        "val_map50": val50,
        "test_map50_95": test95,
        "test_map50": test50,
        "delta_val_map50_95": val95 - b_val95,
        "delta_val_map50": val50 - b_val50,
        "delta_test_map50_95": test95 - b_test95,
        "delta_test_map50": test50 - b_test50,
        # validation first, test as robustness tie-breaker
        "score": (val95 - b_val95) + 0.40 * (val50 - b_val50) + 0.50 * (test95 - b_test95) + 0.20 * (test50 - b_test50),
        "attrs": STAGE1_ATTRS.get(name, {}),
    }


def is_better(row: dict[str, Any]) -> bool:
    return row["delta_val_map50_95"] > EPS or row["delta_val_map50"] > EPS


def is_near(row: dict[str, Any]) -> bool:
    # 单模块没有明显超过 baseline 时，允许非常接近的模块进入少量组合尝试；
    # 这保留“组合协同”的可能，但不会让明显回退模块污染搜索。
    return row["delta_val_map50_95"] >= -0.005 or row["delta_val_map50"] >= -0.005


def pos_tag(positions: tuple[str, ...]) -> str:
    return "".join(p.lower() for p in positions)


def make_doc(
    *,
    backbone: str = "base",
    p2head: bool = False,
    dysample: bool = False,
    attn: str | None = None,
    positions: tuple[str, ...] = (),
    spd_layers: tuple[int, ...] = (),
) -> dict[str, Any]:
    if backbone == "a2c2f":
        bb = sweep.A2C2F_BACKBONE
    elif spd_layers:
        bb = sweep.spd_backbone(*spd_layers)
    else:
        bb = sweep.BASE_BACKBONE

    if p2head:
        head = sweep.make_p2_head(attn, positions)
    else:
        head = sweep.make_head(attn, positions, dysample=dysample)
    return sweep.base_doc(bb, head)


def set_stage2_sweep_outputs() -> None:
    sweep.PROJECT = STAGE2_PROJECT
    sweep.GEN_DIR = STAGE2_GEN_DIR
    sweep.REPORT_JSON = STAGE2_REPORT_JSON
    sweep.REPORT_CSV = STAGE2_REPORT_CSV
    sweep.STATUS_JSON = STAGE2_STATUS_JSON


def add_candidate(
    out: dict[str, dict[str, Any]],
    name: str,
    *,
    note: str,
    components: list[str],
    doc: dict[str, Any],
    source_reason: str,
) -> None:
    if name in out:
        return
    out[name] = {"doc": doc, "note": note, "components": components, "source_reason": source_reason}


def best_row(rows: list[dict[str, Any]], family: str, require_usable: bool = True) -> dict[str, Any] | None:
    subset = [r for r in rows if r["attrs"].get("family") == family and (not require_usable or (is_better(r) or is_near(r)))]
    if not subset:
        return None
    return sorted(subset, key=lambda r: (is_better(r), r["score"], r["delta_val_map50_95"], r["delta_val_map50"]), reverse=True)[0]


def existing_stage1_names(report: dict[str, Any]) -> set[str]:
    return {k for k, v in report.get("experiments", {}).items() if v.get("status") == "done"}


def build_stage2_specs(max_variants: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = load_json(STAGE1_REPORT_JSON)
    experiments = report.get("experiments", {})
    if BASELINE_NAME not in experiments or experiments[BASELINE_NAME].get("status") != "done":
        raise RuntimeError(f"stage1 baseline missing or incomplete in {STAGE1_REPORT_JSON}")

    base = experiments[BASELINE_NAME]
    rows = [
        enrich_metrics(name, exp, base)
        for name, exp in experiments.items()
        if name != BASELINE_NAME and exp.get("status") == "done"
    ]
    rows = sorted(rows, key=lambda r: (is_better(r), r["score"], r["delta_val_map50_95"]), reverse=True)
    winners = [r for r in rows if is_better(r)]
    usable = winners if winners else [r for r in rows if is_near(r)]
    used_stage1 = existing_stage1_names(report)

    candidates: dict[str, dict[str, Any]] = {}
    plan_reason = "winners" if winners else "near_misses"

    attn_rows = [r for r in usable if r["attrs"].get("family") == "attn"]
    if not attn_rows:
        # 如果直接 attention 没进 usable，但 A2C2f+SimAM 是赢家，仍用 SimAM 做位置细化。
        attn_rows = [r for r in usable if r["attrs"].get("family") == "a2c2fbb_attn"]
    best_attn = sorted(attn_rows, key=lambda r: (is_better(r), r["score"], r["delta_val_map50_95"]), reverse=True)[0] if attn_rows else None
    best_dysample = best_row(rows, "dysample", require_usable=True)
    best_spd = best_row(rows, "spd", require_usable=True)
    best_p2 = best_row(rows, "p2head", require_usable=False)
    best_a2c2f = best_row(rows, "a2c2fbb_attn", require_usable=True)

    # 1) 注意力位置/数量细化：先补 stage1 没跑的子集，用于找最优位置和数量。
    if best_attn:
        attn = best_attn["attrs"]["attn"]
        tag = ATTN_TAG[attn]
        if attn == "LSKA":
            pos_sets = [("P4",), ("P5",), ("P3", "P4", "P5"), ("P3", "P5")]
        else:
            pos_sets = [("P4",), ("P5",), ("P3", "P5"), ("P4", "P5"), ("P3", "P4", "P5")]
        for ps in pos_sets:
            ps = tuple(ps)
            name = f"y26n_s2_{tag}_{pos_tag(ps)}_e250"
            equivalent_stage1 = f"y26n_{tag}_{pos_tag(ps)}_e250"
            if equivalent_stage1 in used_stage1:
                continue
            add_candidate(
                candidates,
                name,
                note=f"stage2 attention 位置/数量细化：{attn} at {','.join(ps)}",
                components=[attn, f"positions={'+'.join(ps)}"],
                doc=make_doc(attn=attn, positions=ps),
                source_reason=f"{plan_reason}: best_attn={best_attn['name']}",
            )

    # 2) DySample + 最优注意力：低成本组合，通常最值得优先测。
    if best_attn and best_dysample:
        attn = best_attn["attrs"]["attn"]
        positions = tuple(best_attn["attrs"].get("positions", ()))
        tag = ATTN_TAG[attn]
        name = f"y26n_s2_dysample_{tag}_{pos_tag(positions)}_e250"
        add_candidate(
            candidates,
            name,
            note=f"DySample + {attn}({'+'.join(positions)}) 低成本组合",
            components=["DySample", attn, f"positions={'+'.join(positions)}"],
            doc=make_doc(dysample=True, attn=attn, positions=positions),
            source_reason=f"{plan_reason}: {best_dysample['name']} + {best_attn['name']}",
        )

    # 3) A2C2f backbone 与注意力/DySample 的小组合；历史上 backbone A2C2f 是有效方向。
    if best_attn and (best_a2c2f or is_better(best_attn)):
        attn = best_attn["attrs"]["attn"]
        positions = tuple(best_attn["attrs"].get("positions", ()))
        tag = ATTN_TAG[attn]
        name = f"y26n_s2_a2c2fbb_{tag}_{pos_tag(positions)}_e250"
        # stage1 已跑过 a2c2fbb+SimAM P3P4P5，则不要重复。
        if not (attn == "SimAM" and positions == ("P3", "P4", "P5") and "y26n_a2c2fbb_simam_p3p4p5_e250" in used_stage1):
            add_candidate(
                candidates,
                name,
                note=f"A2C2f backbone + {attn}({'+'.join(positions)})",
                components=["A2C2fBackbone", attn, f"positions={'+'.join(positions)}"],
                doc=make_doc(backbone="a2c2f", attn=attn, positions=positions),
                source_reason=f"{plan_reason}: A2C2f history/stage1 + {best_attn['name']}",
            )
        if best_dysample:
            add_candidate(
                candidates,
                f"y26n_s2_a2c2fbb_dysample_{tag}_{pos_tag(positions)}_e250",
                note=f"A2C2f backbone + DySample + {attn}({'+'.join(positions)})",
                components=["A2C2fBackbone", "DySample", attn, f"positions={'+'.join(positions)}"],
                doc=make_doc(backbone="a2c2f", dysample=True, attn=attn, positions=positions),
                source_reason=f"{plan_reason}: A2C2f history/stage1 + DySample + {best_attn['name']}",
            )

    # 4) SPD 与注意力/DySample：只有 SPD 单模块不明显回退时才尝试。
    if best_spd:
        spd_layers = tuple(best_spd["attrs"]["spd_layers"])
        spd_tag = best_spd["attrs"]["spd_tag"]
        if best_attn:
            attn = best_attn["attrs"]["attn"]
            positions = tuple(best_attn["attrs"].get("positions", ()))
            tag = ATTN_TAG[attn]
            add_candidate(
                candidates,
                f"y26n_s2_spd_{spd_tag}_{tag}_{pos_tag(positions)}_e250",
                note=f"SPDConv({spd_tag}) + {attn}({'+'.join(positions)})",
                components=[f"SPDConv={spd_tag}", attn, f"positions={'+'.join(positions)}"],
                doc=make_doc(spd_layers=spd_layers, attn=attn, positions=positions),
                source_reason=f"{plan_reason}: {best_spd['name']} + {best_attn['name']}",
            )
        if best_dysample:
            add_candidate(
                candidates,
                f"y26n_s2_spd_{spd_tag}_dysample_e250",
                note=f"SPDConv({spd_tag}) + DySample",
                components=[f"SPDConv={spd_tag}", "DySample"],
                doc=make_doc(spd_layers=spd_layers, dysample=True),
                source_reason=f"{plan_reason}: {best_spd['name']} + {best_dysample['name']}",
            )
            if best_attn and is_better(best_spd) and is_better(best_dysample):
                attn = best_attn["attrs"]["attn"]
                positions = tuple(best_attn["attrs"].get("positions", ()))
                tag = ATTN_TAG[attn]
                add_candidate(
                    candidates,
                    f"y26n_s2_spd_{spd_tag}_dysample_{tag}_{pos_tag(positions)}_e250",
                    note=f"SPDConv({spd_tag}) + DySample + {attn}({'+'.join(positions)})",
                    components=[f"SPDConv={spd_tag}", "DySample", attn, f"positions={'+'.join(positions)}"],
                    doc=make_doc(spd_layers=spd_layers, dysample=True, attn=attn, positions=positions),
                    source_reason=f"strict winners combo: {best_spd['name']} + {best_dysample['name']} + {best_attn['name']}",
                )

    # 5) P2 head 很容易增开销/回退，只在 stage1 不严重回退时加一个注意力组合。
    if best_attn and best_p2 and (best_p2["delta_val_map50_95"] >= -0.02 or best_p2["delta_val_map50"] >= -0.02):
        attn = best_attn["attrs"]["attn"]
        positions = tuple(best_attn["attrs"].get("positions", ()))
        tag = ATTN_TAG[attn]
        p2_positions = tuple(["P2", *positions]) if "P2" not in positions else positions
        add_candidate(
            candidates,
            f"y26n_s2_p2head_{tag}_{pos_tag(p2_positions)}_e250",
            note=f"P2 head + {attn}({'+'.join(p2_positions)})，小目标头条件组合",
            components=["P2Head", attn, f"positions={'+'.join(p2_positions)}"],
            doc=make_doc(p2head=True, attn=attn, positions=p2_positions),
            source_reason=f"P2 not severe regression + {best_attn['name']}",
        )

    selected = dict(list(candidates.items())[:max_variants])
    plan = {
        "created_at": now(),
        "baseline": {
            "name": BASELINE_NAME,
            "val_map50_95": metric(base, "val", "map50_95"),
            "val_map50": metric(base, "val", "map50"),
            "test_map50_95": metric(base, "test", "map50_95"),
            "test_map50": metric(base, "test", "map50"),
        },
        "selection_policy": {
            "primary": "only modules better than 250e baseline on val mAP50-95 or val mAP50",
            "fallback": "if no winners, allow near-miss modules within -0.5pp mAP50-95 or mAP50 for limited synergy tests",
            "max_variants": max_variants,
            "reason": plan_reason,
        },
        "stage1_ranked": rows,
        "stage1_winners": winners,
        "selected_candidates": [
            {"name": name, "note": spec["note"], "components": spec["components"], "source_reason": spec["source_reason"]}
            for name, spec in selected.items()
        ],
    }
    return selected, plan


def summarize_best(report: dict[str, Any], stage1_report: dict[str, Any]) -> dict[str, Any]:
    base = stage1_report["experiments"][BASELINE_NAME]
    combined: list[dict[str, Any]] = []
    for source, reps in [("stage1", stage1_report.get("experiments", {})), ("stage2", report.get("experiments", {}))]:
        for name, exp in reps.items():
            if exp.get("status") != "done":
                continue
            row = enrich_metrics(name, exp, base) if name != BASELINE_NAME else {
                "name": name,
                "source": source,
                "val_map50_95": metric(exp, "val", "map50_95"),
                "val_map50": metric(exp, "val", "map50"),
                "test_map50_95": metric(exp, "test", "map50_95"),
                "test_map50": metric(exp, "test", "map50"),
                "delta_val_map50_95": 0.0,
                "delta_val_map50": 0.0,
                "delta_test_map50_95": 0.0,
                "delta_test_map50": 0.0,
                "score": 0.0,
            }
            row["source"] = source
            combined.append(row)
    ranked = sorted(combined, key=lambda r: (r["val_map50_95"], r["val_map50"], r["test_map50_95"]), reverse=True)
    return {
        "updated_at": now(),
        "best_by_val_map50_95": ranked[0] if ranked else None,
        "ranked_top10": ranked[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-stage1", action="store_true", help="Poll stage1 status until it is done before planning stage2.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-wait-hours", type=float, default=72.0)
    parser.add_argument("--max-variants", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=sweep.DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=sweep.DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=sweep.DEFAULT_IMGSZ)
    parser.add_argument("--seed", type=int, default=sweep.DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-plan", action="store_true")
    args = parser.parse_args()

    STAGE2_PROJECT.mkdir(parents=True, exist_ok=True)
    if args.wait_stage1:
        wait_for_stage1(args.poll_seconds, args.max_wait_hours)

    specs, plan = build_stage2_specs(args.max_variants)
    report = load_json(STAGE2_REPORT_JSON) or {
        "schema_version": 1,
        "created_at": now(),
        "experiments": {},
    }
    report["stage1_report_json"] = str(STAGE1_REPORT_JSON)
    report["stage2_plan"] = plan
    report.setdefault("experiments", {})
    write_json(STAGE2_REPORT_JSON, report)

    selected = list(specs)
    if not selected:
        stage2_status("no_candidates", None, [], {"reason": "No stage1 winner or near-miss module suitable for combo search."})
        print("NO_STAGE2_CANDIDATES")
        return 0

    if args.dry_plan:
        stage2_status("planned", None, selected, {"dry_plan": True})
        print(json.dumps(plan["selected_candidates"], indent=2, ensure_ascii=False))
        return 0

    set_stage2_sweep_outputs()
    sweep.ensure_modules()

    completed = {k for k, v in report.get("experiments", {}).items() if v.get("status") == "done"}
    pending = [x for x in selected if args.force or x not in completed]
    sweep.save_status("running", pending[0] if pending else None, pending, {"stage": "stage2_combo"})

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        sweep.save_status("running", name, selected[pos:], {"stage": "stage2_combo"})
        try:
            result = sweep.run_one(name, specs[name], args.epochs, args.batch, args.imgsz, args.seed, args.force)
        except Exception as exc:  # noqa: BLE001
            report.setdefault("experiments", {})[name] = {
                "status": "failed",
                "note": specs[name].get("note", ""),
                "components": specs[name].get("components", []),
                "source_reason": specs[name].get("source_reason", ""),
                "error": repr(exc),
                "failed_at": now(),
            }
            sweep.save_report(report)
            sweep.save_status("failed", name, selected[pos:], {"stage": "stage2_combo", "error": repr(exc)})
            raise
        result["components"] = specs[name].get("components", [])
        result["source_reason"] = specs[name].get("source_reason", "")
        report.setdefault("experiments", {})[name] = result
        stage1_report = load_json(STAGE1_REPORT_JSON)
        report["best_summary"] = summarize_best(report, stage1_report)
        sweep.save_report(report)
        print(
            "STAGE2_RESULT",
            name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50={result['test']['map50']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
        )

    stage1_report = load_json(STAGE1_REPORT_JSON)
    report["best_summary"] = summarize_best(report, stage1_report)
    sweep.save_report(report)
    sweep.save_status("done", None, [], {"stage": "stage2_combo"})
    print(f"STAGE2_DONE report={STAGE2_REPORT_JSON} csv={STAGE2_REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
