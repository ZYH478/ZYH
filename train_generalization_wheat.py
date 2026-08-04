#!/usr/bin/env python
"""YOLO26n 与 MSDGS 在 Wheat 上的四组串行训练编排。

默认模式按固定矩阵逐组启动独立训练 worker；每组完成后再启动独立评测进程。
`--build-check` 只运行数据/模型 gate，不训练；`--worker` 仅供编排器内部调用。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
PROJECT = ROOT / "runs_generalization_wheat_yolo26_msdgs_e250"
DATA_DIR = PROJECT / "datasets"
STATUS = PROJECT / "status.json"
REPORT = PROJECT / "report.json"
REPORT_MD = PROJECT / "report.md"
COMPARISON_CSV = PROJECT / "comparison.csv"
BUILD_REPORT = PROJECT / "build_check_report.json"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
DATASETS = ("wheat",)
MODELS = ("yolo26n", "msdgs")
MODEL_CFG = {"yolo26n": "yolo26n.yaml", "msdgs": str(MSDGS_YAML)}
DEFAULT_EPOCHS = 250
DEFAULT_BATCH = 32
DEFAULT_IMGSZ = 640
DEFAULT_SEED = 0
DEFAULT_PATIENCE = 0


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


def training_report(dataset: str, model: str) -> Path:
    return PROJECT / f"training_{dataset}_{model}_report.json"


def independent_report(dataset: str, model: str) -> Path:
    return PROJECT / f"independent_{dataset}_{model}_report.json"


def shape_tree(value):
    if isinstance(value, dict):
        return {k: shape_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [shape_tree(v) for v in value]
    return list(value.shape) if hasattr(value, "shape") else type(value).__name__


def build_check() -> int:
    from ultralytics.data.utils import check_det_dataset

    if not OFFICIAL_WEIGHTS.is_file():
        raise FileNotFoundError(OFFICIAL_WEIGHTS)
    if not MSDGS_YAML.is_file():
        raise FileNotFoundError(MSDGS_YAML)
    profile = read_json(PROJECT / "dataset_profile.json", {})
    if not profile.get("gate_pass"):
        raise RuntimeError("dataset_profile.json gate_pass is not true")

    datasets = {}
    for dataset in DATASETS:
        data = DATA_DIR / f"{dataset}.yaml"
        checked = check_det_dataset(str(data), autodownload=False)
        datasets[dataset] = {
            "yaml": str(data),
            "nc": int(checked["nc"]),
            "names": checked["names"],
            "train": str(checked["train"]),
            "val": str(checked["val"]),
            "test": str(checked["test"]),
        }
        print(f"DATA_YAML_BUILD_OK {dataset} nc={checked['nc']}")

    models = {}
    for name, cfg in MODEL_CFG.items():
        yolo = YOLO(cfg)
        yolo.load(str(OFFICIAL_WEIGHTS))
        net = yolo.model.to("cuda:0").eval()
        with torch.inference_mode():
            out = net(torch.randn(1, 3, DEFAULT_IMGSZ, DEFAULT_IMGSZ, device="cuda:0"))
        det = net.model[-1]
        custom_count = sum(1 for m in net.modules() if m.__class__.__name__ == "MSDGS")
        fused = net.fuse() if hasattr(net, "fuse") else net
        params = int(sum(p.numel() for p in fused.parameters()))
        models[name] = {
            "cfg": cfg,
            "weights": str(OFFICIAL_WEIGHTS),
            "end2end": bool(getattr(fused, "end2end", False)),
            "reg_max": int(getattr(det, "reg_max", -1)),
            "nl": int(getattr(det, "nl", -1)),
            "msdgs_modules": custom_count,
            "fused_params": params,
            "forward_output": shape_tree(out),
        }
        if not models[name]["end2end"] or models[name]["reg_max"] != 1 or models[name]["nl"] != 3:
            raise RuntimeError(f"bad YOLO26 structure for {name}: {models[name]}")
        if name == "msdgs" and custom_count != 4:
            raise RuntimeError(f"expected four MSDGS neck blocks, got {custom_count}")
        print(f"MODEL_BUILD_OK {name} end2end={models[name]['end2end']} reg_max={models[name]['reg_max']} nl={models[name]['nl']} params={params} msdgs={custom_count}")
    if not models["msdgs"]["fused_params"] < models["yolo26n"]["fused_params"]:
        raise RuntimeError("MSDGS is not lighter than YOLO26n in build gate")

    report = {"status": "pass", "created_at": now(), "cuda": torch.cuda.get_device_name(0), "datasets": datasets, "models": models}
    atomic_json(BUILD_REPORT, report)
    print(f"GENERALIZATION_BUILD_GATE_PASS {BUILD_REPORT}")
    return 0


def worker(dataset: str, model_name: str, epochs: int, batch: int, imgsz: int, seed: int) -> int:
    data = DATA_DIR / f"{dataset}.yaml"
    cfg = MODEL_CFG[model_name]
    report_path = training_report(dataset, model_name)
    started = now()
    report = {
        "status": "running", "started_at": started, "dataset": dataset, "model": model_name,
        "cfg": cfg, "data": str(data), "pid": os.getpid(),
        "recipe": {"epochs": epochs, "batch": batch, "imgsz": imgsz, "seed": seed, "device": 0,
                   "cache": False, "resume": False, "patience": DEFAULT_PATIENCE, "deterministic": True, "pretrained": str(OFFICIAL_WEIGHTS)},
    }
    atomic_json(report_path, report)
    print(f"TRAIN_WORKER_START dataset={dataset} model={model_name} cfg={cfg} epochs={epochs} batch={batch} seed={seed}", flush=True)
    try:
        yolo = YOLO(cfg)
        yolo.train(
            data=str(data), epochs=epochs, batch=batch, imgsz=imgsz, device=0, seed=seed,
            deterministic=True, pretrained=str(OFFICIAL_WEIGHTS), cache=False, resume=False, patience=DEFAULT_PATIENCE,
            project=str(PROJECT / dataset), name=model_name, exist_ok=False,
            plots=False, save=True, verbose=False, rect=False,
        )
        run_dir = Path(str(yolo.trainer.save_dir)).resolve()
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        csv_path = run_dir / "results.csv"
        if not best.is_file() or not last.is_file() or not csv_path.is_file():
            raise RuntimeError(f"missing final artifacts in {run_dir}")
        rows = max(0, len(csv_path.read_text(encoding="utf-8").splitlines()) - 1)
        if rows != epochs:
            raise RuntimeError(f"incomplete results.csv: expected {epochs}, got {rows}")
        report.update({"status": "done", "finished_at": now(), "run_dir": str(run_dir), "best": str(best), "last": str(last), "results_csv": str(csv_path), "completed_epochs": rows})
        atomic_json(report_path, report)
        print(f"TRAIN_WORKER_DONE dataset={dataset} model={model_name} best={best} epochs={rows}", flush=True)
        return 0
    except Exception as exc:
        report.update({"status": "failed", "finished_at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        atomic_json(report_path, report)
        print(f"TRAIN_WORKER_FAILED dataset={dataset} model={model_name} error={exc!r}", flush=True)
        return 1


def summarize() -> dict:
    aggregate = {
        "created_at": now(),
        "wording": "同一目标数据集重新训练后的迁移性与鲁棒性验证，不是 zero-shot 泛化。",
        "protocol": {
            "training": "250 epochs / imgsz640 / batch32 / seed0 / deterministic=True / cache=False / resume=False / patience=0 / yolo26n.pt initialization",
            "truth": "independent fresh Python process reloads best.pt and evaluates test split",
            "fps": "RTX 4090 / fused / imgsz640 / batch32 / fixed warmup and iterations / forward only",
        },
        "experiments": {},
    }
    rows = []
    for dataset in DATASETS:
        aggregate["experiments"][dataset] = {}
        for model in MODELS:
            path = independent_report(dataset, model)
            if not path.exists():
                continue
            rep = read_json(path, {})
            if rep.get("status") != "done":
                continue
            aggregate["experiments"][dataset][model] = rep
            row = {
                "dataset": dataset,
                "model": model,
                "test_map50": rep["test"]["map50"],
                "test_map50_95": rep["test"]["map50_95"],
                "fps": rep["benchmark"]["fps"],
                "infer_ms_per_image": rep["benchmark"]["infer_ms_per_image"],
                "fused_params": rep["structure"]["fused_params"],
                "weights": rep["weights"],
            }
            for cls, value in rep["test"]["class_ap50"].items():
                row[f"class_map50::{cls}"] = value
            for cls, value in rep["test"]["class_ap50_95"].items():
                row[f"class_map50_95::{cls}"] = value
            rows.append(row)
    aggregate["completed"] = len(rows)
    aggregate["expected"] = len(DATASETS) * len(MODELS)
    aggregate["status"] = "done" if len(rows) == aggregate["expected"] else "partial"
    atomic_json(REPORT, aggregate)

    class50_cols = sorted({k for r in rows for k in r if k.startswith("class_map50::")})
    class5095_cols = sorted({k for r in rows for k in r if k.startswith("class_map50_95::")})
    fields = ["dataset", "model", "test_map50", "test_map50_95", "fps", "infer_ms_per_image", "fused_params", *class50_cols, *class5095_cols, "weights"]
    with COMPARISON_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# YOLO26n 与 MSDGS Wheat 迁移性/鲁棒性对比",
        "",
        "> 本报告是两个结构在各目标数据集上按同一协议重新训练后的迁移性与鲁棒性验证，不是 zero-shot 泛化。",
        "",
        "## 统一口径",
        "",
        "- 训练：250 epochs，imgsz=640，batch=32，seed=0，deterministic=True，cache=False，resume=False，patience=0（禁用 EarlyStopping），均从 yolo26n.pt 初始化。",
        "- 真值：每组训练后由独立新 Python 进程从磁盘重载 best.pt，在 test split 评测。",
        "- FPS：同一 RTX 4090，模型 fuse 后，batch=32，固定 warmup/iterations，仅计 forward，排除预处理与后处理。",
        "",
    ]
    for dataset in DATASETS:
        exps = aggregate["experiments"].get(dataset, {})
        lines += [f"## {dataset}", "", "### 总体性能", "", "| 模型 | test mAP50 | test mAP50-95 | FPS | fused params |", "|---|---:|---:|---:|---:|"]
        for model in MODELS:
            rep = exps.get(model)
            if rep:
                lines.append(f"| {model} | {rep['test']['map50']:.6f} | {rep['test']['map50_95']:.6f} | {rep['benchmark']['fps']:.1f} | {rep['structure']['fused_params']:,} |")
            else:
                lines.append(f"| {model} | 待完成 | 待完成 | 待完成 | 待完成 |")
        lines += ["", "### 各缺陷 test mAP50", "", "| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |", "|---|---:|---:|---:|"]
        class_names = []
        for model in MODELS:
            rep = exps.get(model)
            if rep:
                class_names = list(rep["test"]["class_ap50"].keys())
                break
        if not class_names:
            lines.append("| 待训练完成 | - | - | - |")
        else:
            for cls in class_names:
                a = exps.get("yolo26n", {}).get("test", {}).get("class_ap50", {}).get(cls)
                b = exps.get("msdgs", {}).get("test", {}).get("class_ap50", {}).get(cls)
                av = f"{a:.6f}" if isinstance(a, (int, float)) else "待完成"
                bv = f"{b:.6f}" if isinstance(b, (int, float)) else "待完成"
                dv = f"{b-a:+.6f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "待完成"
                lines.append(f"| {cls} | {av} | {bv} | {dv} |")
        lines += ["", "### 各缺陷 test mAP50-95", "", "| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |", "|---|---:|---:|---:|"]
        if not class_names:
            lines.append("| 待训练完成 | - | - | - |")
        else:
            for cls in class_names:
                a = exps.get("yolo26n", {}).get("test", {}).get("class_ap50_95", {}).get(cls)
                b = exps.get("msdgs", {}).get("test", {}).get("class_ap50_95", {}).get(cls)
                av = f"{a:.6f}" if isinstance(a, (int, float)) else "待完成"
                bv = f"{b:.6f}" if isinstance(b, (int, float)) else "待完成"
                dv = f"{b-a:+.6f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "待完成"
                lines.append(f"| {cls} | {av} | {bv} | {dv} |")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SUMMARY_WRITTEN status={aggregate['status']} completed={aggregate['completed']}/{aggregate['expected']} report={REPORT_MD}", flush=True)
    return aggregate


def experiment_done(dataset: str, model: str) -> bool:
    rep = read_json(independent_report(dataset, model), {})
    return rep.get("status") == "done" and Path(rep.get("weights", "")).is_file()


def run_orchestrator(epochs: int, batch: int, imgsz: int, seed: int, only: str) -> int:
    matrix = [(d, m) for d in DATASETS for m in MODELS]
    if only:
        try:
            od, om = only.split("/", 1)
        except ValueError as exc:
            raise ValueError("--only format must be dataset/model") from exc
        matrix = [(d, m) for d, m in matrix if d == od and m == om]
        if not matrix:
            raise ValueError(f"unknown --only {only}")

    completed = [f"{d}/{m}" for d, m in matrix if experiment_done(d, m)]
    failures = []
    atomic_json(STATUS, {"status": "running", "phase": "orchestrating", "started_at": now(), "pid": os.getpid(), "current": None,
                         "completed": completed, "failures": failures,
                         "pending": [f"{d}/{m}" for d, m in matrix if not experiment_done(d, m)]})
    for dataset, model in matrix:
        tag = f"{dataset}/{model}"
        if experiment_done(dataset, model):
            print(f"EXPERIMENT_SKIP {tag} independent report already done", flush=True)
            continue
        tr = read_json(training_report(dataset, model), {})
        if tr.get("status") == "done" and Path(tr.get("best", "")).is_file() and int(tr.get("completed_epochs", -1)) == epochs:
            best = Path(tr["best"])
            print(f"TRAIN_REUSE {tag} best={best}", flush=True)
        else:
            pending = [f"{d}/{m}" for d, m in matrix if not experiment_done(d, m) and f"{d}/{m}" != tag]
            atomic_json(STATUS, {"status": "running", "phase": "training", "updated_at": now(), "pid": os.getpid(), "current": tag,
                                 "completed": completed, "failures": failures, "pending": pending})
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--dataset", dataset, "--model", model,
                   "--epochs", str(epochs), "--batch", str(batch), "--imgsz", str(imgsz), "--seed", str(seed)]
            rc = subprocess.run(cmd).returncode
            tr = read_json(training_report(dataset, model), {})
            if rc != 0 or tr.get("status") != "done":
                failures.append({"experiment": tag, "stage": "training", "rc": rc, "error": tr.get("error")})
                print(f"EXPERIMENT_TRAIN_FAILED {tag}; continue", flush=True)
                summarize()
                continue
            best = Path(tr["best"])

        atomic_json(STATUS, {"status": "running", "phase": "independent_eval", "updated_at": now(), "pid": os.getpid(), "current": tag,
                             "completed": completed, "failures": failures,
                             "pending": [f"{d}/{m}" for d, m in matrix if not experiment_done(d, m) and f"{d}/{m}" != tag]})
        out = independent_report(dataset, model)
        cmd = [sys.executable, str(ROOT / "eval_generalization_wheat.py"), "--dataset", dataset, "--model", model,
               "--weights", str(best), "--data", str(DATA_DIR / f"{dataset}.yaml"), "--output", str(out),
               "--imgsz", str(imgsz), "--batch", str(batch)]
        rc = subprocess.run(cmd).returncode
        ev = read_json(out, {})
        if rc != 0 or ev.get("status") != "done":
            failures.append({"experiment": tag, "stage": "independent_eval", "rc": rc})
            print(f"EXPERIMENT_EVAL_FAILED {tag}; continue", flush=True)
            summarize()
            continue
        completed.append(tag)
        summarize()

    final = summarize()
    status = "done" if not failures and final["completed"] == len(matrix) else "done_with_failures"
    atomic_json(STATUS, {"status": status, "phase": "done", "finished_at": now(), "pid": os.getpid(), "current": None,
                         "completed": completed, "failures": failures, "pending": []})
    print(f"GENERALIZATION_ORCHESTRATOR_DONE status={status} completed={len(completed)}/{len(matrix)} failures={len(failures)}", flush=True)
    return 0 if status == "done" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-check", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--only", default="")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    PROJECT.mkdir(parents=True, exist_ok=True)
    if args.build_check:
        return build_check()
    if args.worker:
        if not args.dataset or not args.model:
            ap.error("--worker requires --dataset and --model")
        return worker(args.dataset, args.model, args.epochs, args.batch, args.imgsz, args.seed)
    return run_orchestrator(args.epochs, args.batch, args.imgsz, args.seed, args.only)


if __name__ == "__main__":
    raise SystemExit(main())


