#!/usr/bin/env python
"""YOLO26 250e module ablation runner for NEU-DET.

目标：
- 基线固定为 YOLO26n，epochs=250，batch=32，imgsz=640，seed=0。
- 自定义模块先做单变量消融，再按结果继续做位置/数量/组合搜索。
- 每个实验训练后用 fresh `YOLO(best.pt).val(...)` 分别评估 val/test，并写 CSV/JSON。

用法（远程 4090）：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py
    python train_yolo26_module_sweep.py --stage stage1

可用 `--variants name1,name2` 只跑指定实验；重复运行会跳过已完成实验。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


# =========================
# Top-level experiment config
# =========================
ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = Path(os.environ.get("YOLO26_EXP_DATA", ROOT / "dataset" / "neu-det.yaml"))
WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = Path(os.environ.get("YOLO26_EXP_PROJECT", ROOT / "runs_module_sweep_e250"))
GEN_DIR = Path(os.environ.get("YOLO26_EXP_GEN_DIR", ROOT / "generated_models_module_sweep_e250"))
REPORT_JSON = PROJECT / "sweep_report.json"
REPORT_CSV = PROJECT / "sweep_results.csv"
STATUS_JSON = PROJECT / "status.json"

DEFAULT_EPOCHS = 250
DEFAULT_BATCH = 32
DEFAULT_IMGSZ = 640
DEFAULT_SEED = 0
ALLOW_RESUME = os.environ.get("YOLO26_ALLOW_RESUME", "0").strip().lower() in {"1", "true", "yes", "y"}
NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


SCALES = {
    "n": [0.50, 0.25, 1024],
    "s": [0.50, 0.50, 1024],
    "m": [0.50, 1.00, 512],
    "l": [1.00, 1.00, 512],
    "x": [1.00, 1.50, 512],
}


BASE_BACKBONE = [
    [-1, 1, "Conv", [64, 3, 2]],
    [-1, 1, "Conv", [128, 3, 2]],
    [-1, 2, "C3k2", [256, False, 0.25]],
    [-1, 1, "Conv", [256, 3, 2]],
    [-1, 2, "C3k2", [512, False, 0.25]],
    [-1, 1, "Conv", [512, 3, 2]],
    [-1, 2, "C3k2", [512, True]],
    [-1, 1, "Conv", [1024, 3, 2]],
    [-1, 2, "C3k2", [1024, True]],
    [-1, 1, "SPPF", [1024, 5, 3, True]],
    [-1, 2, "C2PSA", [1024]],
]


A2C2F_BACKBONE = [
    [-1, 1, "Conv", [64, 3, 2]],
    [-1, 1, "Conv", [128, 3, 2]],
    [-1, 2, "C3k2", [256, False, 0.25]],
    [-1, 1, "Conv", [256, 3, 2]],
    [-1, 2, "C3k2", [512, False, 0.25]],
    [-1, 1, "Conv", [512, 3, 2]],
    [-1, 2, "A2C2f", [512, True, 4]],
    [-1, 1, "Conv", [1024, 3, 2]],
    [-1, 2, "A2C2f", [1024, True, 1]],
    [-1, 1, "SPPF", [1024, 5, 3, True]],
    [-1, 2, "C2PSA", [1024]],
]


def base_doc(backbone: list[list[Any]], head: list[list[Any]]) -> dict[str, Any]:
    return {"nc": 6, "end2end": True, "reg_max": 1, "scales": SCALES, "backbone": backbone, "head": head}


def make_head(attn: str | None = None, positions: tuple[str, ...] = (), dysample: bool = False) -> list[list[Any]]:
    """Build YOLO26 P3/P4/P5 head with optional c-preserving modules."""
    head: list[list[Any]] = []
    idx = 10

    def add(row: list[Any]) -> int:
        nonlocal idx
        head.append(row)
        idx += 1
        return idx

    def attn_args(stage: str) -> list[Any]:
        # c-preserving modules are not base_modules; parse_model keeps c2=ch[f].
        if attn == "EMA":
            return [32]
        if attn == "CoordAtt":
            return [32]
        if attn == "LSKA":
            return [5, 7, 3]
        return []  # SimAM

    up1 = ["DySample", [2, "lp", 4, False]] if dysample else ["nn.Upsample", [None, 2, "nearest"]]
    up2 = ["DySample", [2, "lp", 4, False]] if dysample else ["nn.Upsample", [None, 2, "nearest"]]

    add([-1, 1, up1[0], up1[1]])
    add([[-1, 6], 1, "Concat", [1]])
    p4_lat = add([-1, 2, "C3k2", [512, True]])

    add([-1, 1, up2[0], up2[1]])
    add([[-1, 4], 1, "Concat", [1]])
    p3 = add([-1, 2, "C3k2", [256, True]])
    if attn and "P3" in positions:
        p3 = add([-1, 1, attn, attn_args("P3")])

    add([-1, 1, "Conv", [256, 3, 2]])
    add([[-1, p4_lat], 1, "Concat", [1]])
    p4 = add([-1, 2, "C3k2", [512, True]])
    if attn and "P4" in positions:
        p4 = add([-1, 1, attn, attn_args("P4")])

    add([-1, 1, "Conv", [512, 3, 2]])
    add([[-1, 10], 1, "Concat", [1]])
    p5 = add([-1, 1, "C3k2", [1024, True, 0.5, True]])
    if attn and "P5" in positions:
        p5 = add([-1, 1, attn, attn_args("P5")])

    add([[p3, p4, p5], 1, "Detect", ["nc"]])
    return head


def make_p2_head(attn: str | None = None, positions: tuple[str, ...] = ()) -> list[list[Any]]:
    """Official-style P2/P3/P4/P5 head, with optional c-preserving attention at Detect inputs."""
    head: list[list[Any]] = []
    idx = 10

    def add(row: list[Any]) -> int:
        nonlocal idx
        head.append(row)
        idx += 1
        return idx

    def attn_args(stage: str) -> list[Any]:
        if attn == "EMA":
            return [32]
        if attn == "CoordAtt":
            return [32]
        if attn == "LSKA":
            return [5, 7, 3]
        return []

    add([-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    add([[-1, 6], 1, "Concat", [1]])
    p4_lat = add([-1, 2, "C3k2", [512, True]])
    add([-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    add([[-1, 4], 1, "Concat", [1]])
    p3_lat = add([-1, 2, "C3k2", [256, True]])
    add([-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    add([[-1, 2], 1, "Concat", [1]])
    p2 = add([-1, 2, "C3k2", [128, True]])
    if attn and "P2" in positions:
        p2 = add([-1, 1, attn, attn_args("P2")])
    add([-1, 1, "Conv", [128, 3, 2]])
    add([[-1, p3_lat], 1, "Concat", [1]])
    p3 = add([-1, 2, "C3k2", [256, True]])
    if attn and "P3" in positions:
        p3 = add([-1, 1, attn, attn_args("P3")])
    add([-1, 1, "Conv", [256, 3, 2]])
    add([[-1, p4_lat], 1, "Concat", [1]])
    p4 = add([-1, 2, "C3k2", [512, True]])
    if attn and "P4" in positions:
        p4 = add([-1, 1, attn, attn_args("P4")])
    add([-1, 1, "Conv", [512, 3, 2]])
    add([[-1, 10], 1, "Concat", [1]])
    p5 = add([-1, 1, "C3k2", [1024, True, 0.5, True]])
    if attn and "P5" in positions:
        p5 = add([-1, 1, attn, attn_args("P5")])
    add([[p2, p3, p4, p5], 1, "Detect", ["nc"]])
    return head


def spd_backbone(*layers: int) -> list[list[Any]]:
    """Replace selected stride-2 Conv layers by SPDConv. Layer ids follow BASE_BACKBONE."""
    out = []
    for i, row in enumerate(BASE_BACKBONE):
        row = [row[0], row[1], row[2], list(row[3])]
        if i in layers and row[2] == "Conv" and row[3][2] == 2:
            row[2] = "SPDConv"
        out.append(row)
    return out


def variant_specs() -> dict[str, dict[str, Any]]:
    """Variant matrix, ordered from stable/cheap to risky/heavier."""
    specs: dict[str, dict[str, Any]] = {
        # Fresh owner-requested anchor.
        "y26n_base_e250": {"kind": "weights", "note": "YOLO26n 官方权重微调 250e/b32 基线"},
        # Direct official small-object architecture; previous 150e was weak but 250e requested fresh check.
        "y26n_p2_e250": {"doc": base_doc(BASE_BACKBONE, make_p2_head()), "note": "官方 P2 小目标头，P2/P3/P4/P5 Detect"},
        # Upsampling and attention are c-preserving and low-risk.
        "y26n_dysample_e250": {"doc": base_doc(BASE_BACKBONE, make_head(dysample=True)), "note": "DySample 替换两处 nearest upsample"},
        "y26n_simam_p3_e250": {"doc": base_doc(BASE_BACKBONE, make_head("SimAM", ("P3",))), "note": "SimAM 位置 sweep: P3 only"},
        "y26n_simam_p3p4_e250": {"doc": base_doc(BASE_BACKBONE, make_head("SimAM", ("P3", "P4"))), "note": "SimAM 数量 sweep: P3+P4"},
        "y26n_simam_p3p4p5_e250": {"doc": base_doc(BASE_BACKBONE, make_head("SimAM", ("P3", "P4", "P5"))), "note": "SimAM 数量 sweep: P3+P4+P5"},
        "y26n_ema_p3p4p5_e250": {"doc": base_doc(BASE_BACKBONE, make_head("EMA", ("P3", "P4", "P5"))), "note": "EMA 多尺度注意力 at Detect inputs"},
        "y26n_ca_p3p4p5_e250": {"doc": base_doc(BASE_BACKBONE, make_head("CoordAtt", ("P3", "P4", "P5"))), "note": "Coordinate Attention at Detect inputs"},
        "y26n_lska_p4p5_e250": {"doc": base_doc(BASE_BACKBONE, make_head("LSKA", ("P4", "P5"))), "note": "LSKA 大核注意力 at high-level P4/P5"},
        # SPD position sweep; changes downsampling and partial pretrained weight transfer.
        "y26n_spd_p2_e250": {"doc": base_doc(spd_backbone(1), make_head()), "note": "SPDConv 替换 backbone layer1 P2/4 downsample"},
        "y26n_spd_p3_e250": {"doc": base_doc(spd_backbone(3), make_head()), "note": "SPDConv 替换 backbone layer3 P3/8 downsample"},
        # Re-check previous best family under the new 250e rule.
        "y26n_a2c2fbb_simam_p3p4p5_e250": {
            "doc": base_doc(A2C2F_BACKBONE, make_head("SimAM", ("P3", "P4", "P5"))),
            "note": "旧最佳方向：backbone P4/P5 A2C2f + SimAM×3，按 250e 重跑",
        },
    }
    return specs


def stage_variants(stage: str) -> list[str]:
    if stage == "baseline":
        return ["y26n_base_e250"]
    if stage == "stage1":
        return [
            "y26n_base_e250",
            "y26n_dysample_e250",
            "y26n_simam_p3_e250",
            "y26n_simam_p3p4_e250",
            "y26n_simam_p3p4p5_e250",
            "y26n_ema_p3p4p5_e250",
            "y26n_ca_p3p4p5_e250",
            "y26n_lska_p4p5_e250",
            "y26n_spd_p2_e250",
            "y26n_spd_p3_e250",
            "y26n_p2_e250",
            "y26n_a2c2fbb_simam_p3p4p5_e250",
        ]
    if stage == "attention":
        return ["y26n_simam_p3_e250", "y26n_simam_p3p4_e250", "y26n_simam_p3p4p5_e250", "y26n_ema_p3p4p5_e250", "y26n_ca_p3p4p5_e250", "y26n_lska_p4p5_e250"]
    if stage == "spd":
        return ["y26n_spd_p2_e250", "y26n_spd_p3_e250"]
    if stage == "all":
        return list(variant_specs())
    raise ValueError(f"unknown stage: {stage}")


def ensure_modules() -> None:
    here = Path(__file__).resolve().parent
    installer = here / "install_yolo26_exp_modules.py"
    subprocess.run([sys.executable, str(installer)], check=True)


def dump_yaml(name: str, doc: dict[str, Any]) -> Path:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GEN_DIR / f"{name}.yaml"
    path.write_text("# generated by train_yolo26_module_sweep.py\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def load_report() -> dict[str, Any]:
    if REPORT_JSON.exists():
        return json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    return {"schema_version": 1, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments": {}}


def save_report(report: dict[str, Any]) -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    report["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for name, r in report.get("experiments", {}).items():
        if r.get("status") == "done":
            rows.append({
                "name": name,
                "note": r.get("note", ""),
                "val_map50": r["val"].get("map50"),
                "val_map50_95": r["val"].get("map50_95"),
                "val_precision": r["val"].get("precision"),
                "val_recall": r["val"].get("recall"),
                "test_map50": r["test"].get("map50"),
                "test_map50_95": r["test"].get("map50_95"),
                "params": r.get("params"),
                "gflops": r.get("gflops"),
                "infer_ms_val": r["val"].get("infer_ms"),
                "weights": r.get("weights"),
                "cfg": r.get("cfg"),
            })
    if rows:
        with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def save_status(status: str, current: str | None, pending: list[str], extra: dict[str, Any] | None = None) -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "current": current,
        "pending": pending,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report_json": str(REPORT_JSON),
        "report_csv": str(REPORT_CSV),
    }
    if extra:
        payload.update(extra)
    STATUS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def metric_dict(m: Any) -> dict[str, Any]:
    infer = float(m.speed.get("inference", 0.0) or 0.0)
    out = {
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "speed": {k: float(v) for k, v in m.speed.items()},
        "infer_ms": infer,
        "fps_infer_only": (1000.0 / infer) if infer else None,
        "per_class": {},
    }
    for i, cls in enumerate(m.ap_class_index):
        name = NAMES[int(cls)] if int(cls) < len(NAMES) else str(int(cls))
        out["per_class"][name] = {"map50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i])}
    return out


def model_size(model: Any) -> tuple[int, float | None]:
    params = int(sum(p.numel() for p in model.model.parameters()))
    gflops = None
    try:
        info = model.model.info(detailed=False, verbose=False)
        if isinstance(info, tuple) and len(info) >= 4:
            gflops = float(info[3])
    except Exception:
        gflops = None
    return params, gflops


def training_rows(run_dir: Path) -> int:
    """Return completed result rows in Ultralytics results.csv, ignoring header."""
    results = run_dir / "results.csv"
    if not results.exists():
        return 0
    try:
        with results.open(newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def run_one(name: str, spec: dict[str, Any], epochs: int, batch: int, imgsz: int, seed: int, force: bool = False) -> dict[str, Any]:
    from ultralytics import YOLO

    run_dir = PROJECT / name
    weights = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    cfg_path = None
    if "doc" in spec:
        cfg_path = dump_yaml(name, spec["doc"])
        model_arg: str | Path = cfg_path
        model = YOLO(str(model_arg)).load(str(WEIGHTS))
    else:
        model_arg = WEIGHTS
        model = YOLO(str(model_arg))

    rows = training_rows(run_dir)
    if weights.exists() and rows >= epochs and not force:
        print(f"SKIP_TRAIN existing weights: {name}")
    elif last.exists() and rows > 0 and ALLOW_RESUME and not force:
        print(f"RESUME_TRAIN {name} rows={rows}/{epochs} last={last}")
        YOLO(str(last)).train(resume=True)
        print(f"TRAIN_DONE {name}")
    else:
        if rows > 0 and run_dir.exists() and not force:
            backup = run_dir.with_name(f"{run_dir.name}_interrupted_{time.strftime('%Y%m%d_%H%M%S')}")
            print(f"BACKUP_PARTIAL_RUN {name} rows={rows}/{epochs} backup={backup}")
            run_dir.rename(backup)
        print(f"TRAIN_START {name} epochs={epochs} batch={batch} imgsz={imgsz}")
        model.train(
            data=str(DATA),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=8,
            seed=seed,
            device=0,
            project=str(PROJECT),
            name=name,
            exist_ok=True,
            patience=max(epochs, 250),  # effectively no early stop under the requested 250e regime
            cache=False,
            verbose=True,
        )
        print(f"TRAIN_DONE {name}")

    best = YOLO(str(weights))
    val = best.val(data=str(DATA), split="val", imgsz=imgsz, batch=batch, device=0, project=str(PROJECT), name=f"{name}_val", exist_ok=True, verbose=False)
    test = best.val(data=str(DATA), split="test", imgsz=imgsz, batch=batch, device=0, project=str(PROJECT), name=f"{name}_test", exist_ok=True, verbose=False)
    params, gflops = model_size(best)
    return {
        "status": "done",
        "note": spec.get("note", ""),
        "cfg": str(cfg_path) if cfg_path else str(WEIGHTS),
        "weights": str(weights),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "seed": seed,
        "params": params,
        "gflops": gflops,
        "val": metric_dict(val),
        "test": metric_dict(test),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="stage1", choices=["baseline", "stage1", "attention", "spd", "all"])
    parser.add_argument("--variants", default="", help="Comma-separated variant names. Overrides --stage when set.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    PROJECT.mkdir(parents=True, exist_ok=True)
    ensure_modules()

    specs = variant_specs()
    selected = [x.strip() for x in args.variants.split(",") if x.strip()] if args.variants else stage_variants(args.stage)
    unknown = [x for x in selected if x not in specs]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")

    report = load_report()
    completed = set(k for k, v in report.get("experiments", {}).items() if v.get("status") == "done")
    pending = [x for x in selected if args.force or x not in completed]
    save_status("running", pending[0] if pending else None, pending)

    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        save_status("running", name, selected[pos:])
        try:
            result = run_one(name, specs[name], args.epochs, args.batch, args.imgsz, args.seed, args.force)
        except Exception as exc:  # noqa: BLE001
            report.setdefault("experiments", {})[name] = {
                "status": "failed",
                "note": specs[name].get("note", ""),
                "error": repr(exc),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_report(report)
            save_status("failed", name, selected[pos:], {"error": repr(exc)})
            raise
        report.setdefault("experiments", {})[name] = result
        save_report(report)
        print(
            "RESULT",
            name,
            f"val_mAP50={result['val']['map50']:.5f}",
            f"val_mAP50-95={result['val']['map50_95']:.5f}",
            f"test_mAP50={result['test']['map50']:.5f}",
            f"test_mAP50-95={result['test']['map50_95']:.5f}",
        )

    save_status("done", None, [])
    print(f"SWEEP_DONE report={REPORT_JSON} csv={REPORT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
