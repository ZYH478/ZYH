#!/usr/bin/env python
"""Run Gold-YOLO-n and YOLOv11n seeds 0/1/2/3 on NEU-DET, Aluminum, and PCB.

This is a queue-safe top-level orchestrator. It delegates each model/seed to the
model-specific restartable trainer, and writes a combined status/report without
mixing Python environments.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
PROJECT = ROOT / "runs_sota_gold_yolo_n_yolo11n_multiseed_e250"
STATUS = PROJECT / "status.json"
REPORT = PROJECT / "report.json"
COMPARISON_CSV = PROJECT / "comparison.csv"
REPORT_MD = PROJECT / "report.md"
SEEDS = (0, 1, 2, 3)
DATASETS = ("neudet", "aluminum", "pcb")

TASKS = [
    {
        "model": "Gold-YOLO-n",
        "model_key": "gold_yolo_n",
        "script": ROOT / "train_sota_gold_yolo_n_seed0.py",
        "python": Path(os.environ.get("LIGHTYOLO_PYTHON", "/root/miniconda3/envs/lightyolo/bin/python")),
        "project_template": "runs_sota_gold_yolo_n_seed{seed}_e250",
        "weight_name": "best_ckpt.pt",
    },
    {
        "model": "yolo11n",
        "model_key": "yolo11n",
        "script": ROOT / "train_sota_yolo11n_seed0.py",
        "python": Path(os.environ.get("YOLO26_PYTHON", "/root/miniconda3/envs/yolo26/bin/python")),
        "project_template": "runs_sota_yolo11n_seed{seed}_e250",
        "weight_name": "best.pt",
    },
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def seed_project(task: dict, seed: int) -> Path:
    return ROOT / task["project_template"].format(seed=seed)


def seed_status(task: dict, seed: int) -> Path:
    return seed_project(task, seed) / "status.json"


def seed_report(task: dict, seed: int) -> Path:
    return seed_project(task, seed) / "report.json"


def seed_done(task: dict, seed: int) -> bool:
    status = read_json(seed_status(task, seed), {})
    report = read_json(seed_report(task, seed), {})
    if status.get("status") != "done" or report.get("status") != "done":
        return False
    experiments = report.get("experiments", {})
    return all(dataset in experiments for dataset in DATASETS)


def collect_rows() -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    pending: list[dict] = []
    summary: dict = {}
    for task in TASKS:
        model = task["model"]
        summary[model] = {}
        for seed in SEEDS:
            report = read_json(seed_report(task, seed), {})
            per_seed = {"status": report.get("status", "missing"), "project": str(seed_project(task, seed)), "datasets": {}}
            experiments = report.get("experiments", {}) if isinstance(report, dict) else {}
            for dataset in DATASETS:
                exp = experiments.get(dataset) if isinstance(experiments, dict) else None
                if exp and exp.get("status") == "done":
                    row = {
                        "model": model,
                        "seed": seed,
                        "dataset": dataset,
                        "test_map50": exp["test"]["map50"],
                        "test_map50_95": exp["test"]["map50_95"],
                        "precision": exp["test"]["precision"],
                        "recall": exp["test"]["recall"],
                        "fps": exp["benchmark"]["fps"],
                        "infer_ms_per_image": exp["benchmark"].get("infer_ms_per_image"),
                        "fused_params": exp["structure"].get("fused_params"),
                        "weights": exp.get("weights"),
                    }
                    for name, value in exp["test"].get("class_ap50", {}).items():
                        row[f"class_ap50::{name}"] = value
                    for name, value in exp["test"].get("class_ap50_95", {}).items():
                        row[f"class_ap50_95::{name}"] = value
                    rows.append(row)
                    per_seed["datasets"][dataset] = {"status": "done", "map50": row["test_map50"], "map50_95": row["test_map50_95"]}
                else:
                    pending.append({"model": model, "seed": seed, "dataset": dataset, "project": str(seed_project(task, seed))})
                    per_seed["datasets"][dataset] = {"status": "pending"}
            summary[model][f"seed{seed}"] = per_seed
    return rows, pending, summary


def write_summary(status: str = "partial") -> dict:
    rows, pending, summary = collect_rows()
    expected = len(TASKS) * len(SEEDS) * len(DATASETS)
    completed = len(rows)
    aggregate = {
        "status": "done" if completed == expected else status,
        "created_at": now(),
        "models": [task["model"] for task in TASKS],
        "seeds": list(SEEDS),
        "datasets": list(DATASETS),
        "completed": completed,
        "expected": expected,
        "pending": pending,
        "summary": summary,
        "protocol": {
            "training": "250 epochs / imgsz640 / batch32 / deterministic per seed / cache=False where applicable",
            "truth": "fresh-process evaluation reloads best checkpoint from disk for each model/seed/dataset",
            "order": "Gold-YOLO-n seeds 0-3, then YOLOv11n seeds 0-3; each seed runs NEU-DET, Aluminum, PCB serially",
        },
    }
    atomic_json(REPORT, aggregate)
    class_columns = sorted({key for row in rows for key in row if key.startswith("class_")})
    fields = ["model", "seed", "dataset", "test_map50", "test_map50_95", "precision", "recall", "fps", "infer_ms_per_image", "fused_params", *class_columns, "weights"]
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Gold-YOLO-n 与 YOLOv11n 四 Seed SOTA 训练汇总",
        "",
        f"- 状态：{aggregate['status']}",
        f"- 完成：{completed}/{expected}",
        "- 顺序：Gold-YOLO-n seed0/1/2/3 → YOLOv11n seed0/1/2/3；每个 seed 内按 NEU-DET、Aluminum、PCB 串行。",
        "- 真值：每组训练结束后独立新进程从磁盘重载 best checkpoint，在 test split 评测。",
        "",
        "| Model | Seed | Dataset | mAP50 | mAP50-95 | FPS | Weights |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["model"], int(r["seed"]), r["dataset"])):
        lines.append(f"| {row['model']} | {row['seed']} | {row['dataset']} | {row['test_map50']:.6f} | {row['test_map50_95']:.6f} | {row['fps']:.2f} | `{row['weights']}` |")
    if pending:
        lines += ["", "## Pending", "", "| Model | Seed | Dataset | Project |", "|---|---:|---|---|"]
        for item in pending[:80]:
            lines.append(f"| {item['model']} | {item['seed']} | {item['dataset']} | `{item['project']}` |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def run_task(task: dict, seed: int) -> int:
    project = seed_project(task, seed)
    env = os.environ.copy()
    env["YOLO26_EXP_ROOT"] = str(ROOT)
    env["SOTA_SEED"] = str(seed)
    env["SOTA_PROJECT"] = str(project)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not task["python"].is_file():
        raise FileNotFoundError(task["python"])
    if not task["script"].is_file():
        raise FileNotFoundError(task["script"])
    if seed_done(task, seed):
        log(f"SKIP_MODEL_SEED_DONE model={task['model']} seed={seed} project={project}")
        return 0
    atomic_json(STATUS, {
        "status": "running", "phase": "model_seed", "updated_at": now(),
        "current": {"model": task["model"], "seed": seed, "project": str(project)},
        "models": [t["model"] for t in TASKS], "seeds": list(SEEDS), "datasets": list(DATASETS),
        "report": str(REPORT),
    })
    log(f"MODEL_SEED_START model={task['model']} seed={seed} project={project} python={task['python']}")
    rc = subprocess.call([str(task["python"]), str(task["script"])], env=env, cwd=str(ROOT))
    if rc != 0:
        raise RuntimeError(f"model seed failed model={task['model']} seed={seed} rc={rc}")
    write_summary("partial")
    log(f"MODEL_SEED_DONE model={task['model']} seed={seed} project={project}")
    return 0


def main() -> int:
    PROJECT.mkdir(parents=True, exist_ok=True)
    atomic_json(STATUS, {
        "status": "running", "phase": "start", "started_at": now(),
        "models": [task["model"] for task in TASKS], "seeds": list(SEEDS), "datasets": list(DATASETS),
        "order": "Gold-YOLO-n seeds 0-3 then YOLOv11n seeds 0-3",
    })
    try:
        write_summary("partial")
        for task in TASKS:
            for seed in SEEDS:
                run_task(task, seed)
        aggregate = write_summary("done")
        atomic_json(STATUS, {
            "status": "done", "phase": "done", "finished_at": now(),
            "completed": aggregate["completed"], "expected": aggregate["expected"],
            "aggregate_report": str(REPORT), "comparison_csv": str(COMPARISON_CSV),
        })
        log(f"ALL_DONE completed={aggregate['completed']}/{aggregate['expected']} report={REPORT}")
        return 0
    except Exception as exc:
        atomic_json(STATUS, {
            "status": "failed", "phase": "failed", "updated_at": now(),
            "error": repr(exc), "traceback": traceback.format_exc(),
            "aggregate_report": str(REPORT),
        })
        log(f"FAILED error={exc!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
