#!/usr/bin/env python
"""Aluminum 上 YOLO26n/MSDGS seed1/2/3 稳定性训练与四 seed 汇总。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
PROJECT = ROOT / "runs_aluminum_multiseed_e250"
SOURCE_PROJECT = ROOT / "runs_generalization_aluminum_pcb_yolo26_msdgs_e250"
DATA = SOURCE_PROJECT / "datasets" / "aluminum.yaml"
OFFICIAL_WEIGHTS = ROOT / "yolo26n.pt"
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
EVAL_SCRIPT = ROOT / "eval_generalization_aluminum_pcb.py"
MODEL_CFG = {"yolo26n": "yolo26n.yaml", "msdgs": str(MSDGS_YAML)}
MODELS = ("yolo26n", "msdgs")
SEEDS = (1, 2, 3)
ALL_SEEDS = (0, 1, 2, 3)
DEFAULT_EPOCHS = 250
DEFAULT_BATCH = 32
DEFAULT_IMGSZ = 640


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def training_report(model: str, seed: int) -> Path:
    return PROJECT / f"training_aluminum_{model}_seed{seed}_report.json"


def independent_report(model: str, seed: int) -> Path:
    if seed == 0:
        return SOURCE_PROJECT / f"independent_aluminum_{model}_report.json"
    return PROJECT / f"independent_aluminum_{model}_seed{seed}_report.json"


def build_check() -> int:
    from ultralytics.data.utils import check_det_dataset

    required = [DATA, OFFICIAL_WEIGHTS, MSDGS_YAML, EVAL_SCRIPT]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    checked = check_det_dataset(str(DATA), autodownload=False)
    if int(checked["nc"]) != 4:
        raise RuntimeError(f"expected 4 classes, got {checked['nc']}")
    seed0 = {}
    models = {}
    for model_name, cfg in MODEL_CFG.items():
        rep0 = read_json(independent_report(model_name, 0), {})
        if rep0.get("status") != "done":
            raise RuntimeError(f"missing completed seed0 independent report for {model_name}")
        seed0[model_name] = rep0
        yolo = YOLO(cfg)
        yolo.load(str(OFFICIAL_WEIGHTS))
        net = yolo.model.to("cuda:0").eval()
        with torch.inference_mode():
            out = net(torch.randn(1, 3, DEFAULT_IMGSZ, DEFAULT_IMGSZ, device="cuda:0"))
        det = net.model[-1]
        custom = sum(1 for m in net.modules() if m.__class__.__name__ == "MSDGS")
        fused = net.fuse() if hasattr(net, "fuse") else net
        params = int(sum(p.numel() for p in fused.parameters()))
        models[model_name] = {
            "cfg": cfg,
            "end2end": bool(getattr(fused, "end2end", False)),
            "reg_max": int(getattr(det, "reg_max", -1)),
            "nl": int(getattr(det, "nl", -1)),
            "msdgs_modules": custom,
            "fused_params": params,
            "forward_type": type(out).__name__,
        }
        if not models[model_name]["end2end"] or models[model_name]["reg_max"] != 1 or models[model_name]["nl"] != 3:
            raise RuntimeError(f"bad YOLO26 structure: {models[model_name]}")
        if model_name == "msdgs" and custom != 4:
            raise RuntimeError(f"expected 4 MSDGS modules, got {custom}")
        print(f"MODEL_BUILD_OK model={model_name} params={params} msdgs={custom}", flush=True)
    report = {
        "status": "pass",
        "created_at": now(),
        "data": str(DATA),
        "nc": int(checked["nc"]),
        "names": checked["names"],
        "seeds_to_train": list(SEEDS),
        "seed0_anchor": {m: str(independent_report(m, 0)) for m in MODELS},
        "models": models,
        "cuda": torch.cuda.get_device_name(0),
    }
    atomic_json(PROJECT / "build_check_report.json", report)
    print("ALUMINUM_MULTISEED_BUILD_GATE_PASS", flush=True)
    return 0


def worker(model_name: str, seed: int, epochs: int, batch: int, imgsz: int) -> int:
    cfg = MODEL_CFG[model_name]
    out = training_report(model_name, seed)
    report = {
        "status": "running",
        "started_at": now(),
        "dataset": "aluminum",
        "model": model_name,
        "seed": seed,
        "pid": os.getpid(),
        "cfg": cfg,
        "data": str(DATA),
        "recipe": {
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "seed": seed,
            "device": 0,
            "cache": False,
            "resume": False,
            "patience": 0,
            "deterministic": True,
            "pretrained": str(OFFICIAL_WEIGHTS),
        },
    }
    atomic_json(out, report)
    print(f"TRAIN_START aluminum/{model_name}/seed{seed}", flush=True)
    try:
        yolo = YOLO(cfg)
        yolo.train(
            data=str(DATA), epochs=epochs, batch=batch, imgsz=imgsz, device=0,
            seed=seed, deterministic=True, pretrained=str(OFFICIAL_WEIGHTS),
            cache=False, resume=False, patience=0,
            project=str(PROJECT / f"seed{seed}"), name=model_name, exist_ok=False,
            plots=False, save=True, verbose=False, rect=False,
        )
        run_dir = Path(str(yolo.trainer.save_dir)).resolve()
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        results = run_dir / "results.csv"
        if not best.is_file() or not last.is_file() or not results.is_file():
            raise RuntimeError(f"missing artifacts in {run_dir}")
        completed_epochs = len(results.read_text(encoding="utf-8").splitlines()) - 1
        if completed_epochs != epochs:
            raise RuntimeError(f"expected {epochs} epochs, got {completed_epochs}")
        report.update({
            "status": "done", "finished_at": now(), "run_dir": str(run_dir),
            "best": str(best), "last": str(last), "results_csv": str(results),
            "completed_epochs": completed_epochs,
        })
        atomic_json(out, report)
        print(f"TRAIN_DONE aluminum/{model_name}/seed{seed} best={best}", flush=True)
        return 0
    except Exception as exc:
        report.update({"status": "failed", "finished_at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        atomic_json(out, report)
        print(f"TRAIN_FAILED aluminum/{model_name}/seed{seed} error={exc!r}", flush=True)
        return 1


def experiment_done(model: str, seed: int) -> bool:
    rep = read_json(independent_report(model, seed), {})
    return rep.get("status") == "done" and Path(rep.get("weights", "")).is_file()


def metric_stats(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def summarize() -> dict:
    experiments = {}
    rows = []
    for model in MODELS:
        experiments[model] = {}
        for seed in ALL_SEEDS:
            rep = read_json(independent_report(model, seed), {})
            if rep.get("status") != "done":
                continue
            if seed != 0 and rep.get("seed") != seed:
                rep["seed"] = seed
                atomic_json(independent_report(model, seed), rep)
            experiments[model][str(seed)] = rep
            row = {
                "model": model,
                "seed": seed,
                "test_map50": rep["test"]["map50"],
                "test_map50_95": rep["test"]["map50_95"],
                "precision": rep["test"]["precision"],
                "recall": rep["test"]["recall"],
                "fps": rep["benchmark"]["fps"],
                "fused_params": rep["structure"]["fused_params"],
                "weights": rep["weights"],
            }
            for cls, value in rep["test"]["class_ap50"].items():
                row[f"class_ap50::{cls}"] = value
            for cls, value in rep["test"]["class_ap50_95"].items():
                row[f"class_ap50_95::{cls}"] = value
            rows.append(row)

    stats = {}
    for model in MODELS:
        reps = experiments[model]
        if len(reps) != len(ALL_SEEDS):
            continue
        stats[model] = {
            "test_map50": metric_stats([reps[str(s)]["test"]["map50"] for s in ALL_SEEDS]),
            "test_map50_95": metric_stats([reps[str(s)]["test"]["map50_95"] for s in ALL_SEEDS]),
            "precision": metric_stats([reps[str(s)]["test"]["precision"] for s in ALL_SEEDS]),
            "recall": metric_stats([reps[str(s)]["test"]["recall"] for s in ALL_SEEDS]),
            "fps": metric_stats([reps[str(s)]["benchmark"]["fps"] for s in ALL_SEEDS]),
            "fused_params": reps["0"]["structure"]["fused_params"],
            "class_ap50": {},
            "class_ap50_95": {},
        }
        classes = list(reps["0"]["test"]["class_ap50"])
        for cls in classes:
            stats[model]["class_ap50"][cls] = metric_stats([reps[str(s)]["test"]["class_ap50"][cls] for s in ALL_SEEDS])
            stats[model]["class_ap50_95"][cls] = metric_stats([reps[str(s)]["test"]["class_ap50_95"][cls] for s in ALL_SEEDS])

    paired = {}
    if all(len(experiments[m]) == len(ALL_SEEDS) for m in MODELS):
        paired = {
            str(seed): {
                "map50_delta_msdgs_minus_yolo26n": experiments["msdgs"][str(seed)]["test"]["map50"] - experiments["yolo26n"][str(seed)]["test"]["map50"],
                "map50_95_delta_msdgs_minus_yolo26n": experiments["msdgs"][str(seed)]["test"]["map50_95"] - experiments["yolo26n"][str(seed)]["test"]["map50_95"],
            }
            for seed in ALL_SEEDS
        }

    report = {
        "status": "done" if len(rows) == len(MODELS) * len(ALL_SEEDS) else "partial",
        "created_at": now(),
        "dataset": "aluminum",
        "protocol": "250e/imgsz640/batch32/deterministic/cacheFalse/patience0/yolo26n.pt; independent fresh-process test evaluation",
        "seed0_source": str(SOURCE_PROJECT),
        "new_seeds": list(SEEDS),
        "experiments": experiments,
        "statistics": stats,
        "paired_deltas": paired,
        "completed_total": len(rows),
        "expected_total": len(MODELS) * len(ALL_SEEDS),
    }
    atomic_json(PROJECT / "report.json", report)

    class_cols = sorted({k for r in rows for k in r if k.startswith("class_")})
    fields = ["model", "seed", "test_map50", "test_map50_95", "precision", "recall", "fps", "fused_params", *class_cols, "weights"]
    with (PROJECT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["model"], r["seed"])))

    lines = [
        "# Aluminum YOLO26n / MSDGS 四 Seed 稳定性对比",
        "",
        "> seed0 复用既有独立 test 真值；seed1/2/3 使用完全相同训练协议补跑。所有指标均由独立新 Python 进程从磁盘重载 best.pt 后在 test split 评测。",
        "",
        "## 各 Seed 总体结果",
        "",
        "| 模型 | Seed | test mAP50 | test mAP50-95 | Precision | Recall | FPS | Fused Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for seed in ALL_SEEDS:
            rep = experiments.get(model, {}).get(str(seed))
            if rep:
                lines.append(f"| {model} | {seed} | {rep['test']['map50']:.6f} | {rep['test']['map50_95']:.6f} | {rep['test']['precision']:.6f} | {rep['test']['recall']:.6f} | {rep['benchmark']['fps']:.1f} | {rep['structure']['fused_params']:,} |")
            else:
                lines.append(f"| {model} | {seed} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
    lines += ["", "## 四 Seed 稳定性统计", "", "| 模型 | mAP50 mean±std | min | range | mAP50-95 mean±std | min | range |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        st = stats.get(model)
        if st:
            a, b = st["test_map50"], st["test_map50_95"]
            lines.append(f"| {model} | {a['mean']:.6f} ± {a['std_population']:.6f} | {a['min']:.6f} | {a['range']:.6f} | {b['mean']:.6f} ± {b['std_population']:.6f} | {b['min']:.6f} | {b['range']:.6f} |")
        else:
            lines.append(f"| {model} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
    lines += ["", "## 各类别四 Seed 均值", "", "| 类别 | YOLO26n AP50 mean±std | MSDGS AP50 mean±std | MSDGS-YOLO | YOLO26n AP50-95 mean±std | MSDGS AP50-95 mean±std | MSDGS-YOLO |", "|---|---:|---:|---:|---:|---:|---:|"]
    if all(m in stats for m in MODELS):
        for cls in stats["yolo26n"]["class_ap50"]:
            ya = stats["yolo26n"]["class_ap50"][cls]
            ma = stats["msdgs"]["class_ap50"][cls]
            yb = stats["yolo26n"]["class_ap50_95"][cls]
            mb = stats["msdgs"]["class_ap50_95"][cls]
            lines.append(f"| {cls} | {ya['mean']:.6f} ± {ya['std_population']:.6f} | {ma['mean']:.6f} ± {ma['std_population']:.6f} | {ma['mean']-ya['mean']:+.6f} | {yb['mean']:.6f} ± {yb['std_population']:.6f} | {mb['mean']:.6f} ± {mb['std_population']:.6f} | {mb['mean']-yb['mean']:+.6f} |")
    else:
        lines.append("| 待训练完成 | - | - | - | - | - | - |")
    (PROJECT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SUMMARY_WRITTEN status={report['status']} completed={len(rows)}/{len(MODELS)*len(ALL_SEEDS)}", flush=True)
    return report


def run_orchestrator(epochs: int, batch: int, imgsz: int) -> int:
    matrix = [(seed, model) for seed in SEEDS for model in MODELS]
    completed = [f"aluminum/{model}/seed{seed}" for seed, model in matrix if experiment_done(model, seed)]
    failures = []
    status_path = PROJECT / "status.json"
    atomic_json(status_path, {
        "status": "running", "phase": "orchestrating", "started_at": now(), "pid": os.getpid(),
        "current": None, "completed": completed, "failures": failures,
        "pending": [f"aluminum/{m}/seed{s}" for s, m in matrix if not experiment_done(m, s)],
    })
    for seed, model in matrix:
        tag = f"aluminum/{model}/seed{seed}"
        if experiment_done(model, seed):
            print(f"EXPERIMENT_SKIP {tag}", flush=True)
            continue
        tr = read_json(training_report(model, seed), {})
        if tr.get("status") == "done" and tr.get("completed_epochs") == epochs and Path(tr.get("best", "")).is_file():
            best = Path(tr["best"])
            print(f"TRAIN_REUSE {tag} best={best}", flush=True)
        else:
            atomic_json(status_path, {
                "status": "running", "phase": "training", "updated_at": now(), "pid": os.getpid(),
                "current": tag, "completed": completed, "failures": failures,
                "pending": [f"aluminum/{m}/seed{s}" for s, m in matrix if f"aluminum/{m}/seed{s}" != tag and not experiment_done(m, s)],
            })
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model", model, "--seed", str(seed), "--epochs", str(epochs), "--batch", str(batch), "--imgsz", str(imgsz)]
            rc = subprocess.run(cmd).returncode
            tr = read_json(training_report(model, seed), {})
            if rc != 0 or tr.get("status") != "done":
                failures.append({"experiment": tag, "stage": "training", "rc": rc, "error": tr.get("error")})
                summarize()
                continue
            best = Path(tr["best"])
        atomic_json(status_path, {
            "status": "running", "phase": "independent_eval", "updated_at": now(), "pid": os.getpid(),
            "current": tag, "completed": completed, "failures": failures,
            "pending": [f"aluminum/{m}/seed{s}" for s, m in matrix if f"aluminum/{m}/seed{s}" != tag and not experiment_done(m, s)],
        })
        out = independent_report(model, seed)
        cmd = [sys.executable, str(EVAL_SCRIPT), "--dataset", "aluminum", "--model", model, "--weights", str(best), "--data", str(DATA), "--output", str(out), "--imgsz", str(imgsz), "--batch", str(batch)]
        rc = subprocess.run(cmd).returncode
        ev = read_json(out, {})
        if rc != 0 or ev.get("status") != "done":
            failures.append({"experiment": tag, "stage": "independent_eval", "rc": rc})
            summarize()
            continue
        ev["seed"] = seed
        atomic_json(out, ev)
        completed.append(tag)
        summarize()
    final = summarize()
    status = "done" if not failures and final["status"] == "done" else "done_with_failures"
    atomic_json(status_path, {
        "status": status, "phase": "done", "finished_at": now(), "pid": os.getpid(),
        "current": None, "completed": completed, "failures": failures, "pending": [],
    })
    print(f"ALUMINUM_MULTISEED_DONE status={status} completed={len(completed)}/{len(matrix)} failures={len(failures)}", flush=True)
    return 0 if status == "done" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-check", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.build_check:
        return build_check()
    if args.summarize:
        summarize()
        return 0
    if args.worker:
        if args.model is None or args.seed not in SEEDS:
            raise ValueError("worker requires --model and --seed in 1,2,3")
        return worker(args.model, args.seed, args.epochs, args.batch, args.imgsz)
    return run_orchestrator(args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    raise SystemExit(main())
