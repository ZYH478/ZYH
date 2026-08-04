#!/usr/bin/env python
"""Train YOLOv11n for one selectable seed on NEU-DET, Aluminum, and PCB, then evaluate best.pt.

The orchestrator is restartable. Each dataset trains in its own Python process,
and each final report is produced by another fresh process that reloads best.pt.
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
SEED = int(os.environ.get("SOTA_SEED", "0"))
PROJECT = Path(os.environ.get("SOTA_PROJECT", str(ROOT / f"runs_sota_yolo11n_seed{SEED}_e250")))
OFFICIAL = ROOT / "yolo11n.pt"
STATUS = PROJECT / "status.json"
REPORT = PROJECT / "report.json"
REPORT_MD = PROJECT / "report.md"
COMPARISON_CSV = PROJECT / "comparison.csv"
BUILD_REPORT = PROJECT / "build_check_report.json"
PYTHON = sys.executable

DATASETS = {
    "neudet": ROOT / "dataset" / "neu-det.yaml",
    "aluminum": ROOT / "runs_generalization_aluminum_pcb_yolo26_msdgs_e250" / "datasets" / "aluminum.yaml",
    "pcb": ROOT / "runs_generalization_aluminum_pcb_yolo26_msdgs_e250" / "datasets" / "pcb.yaml",
}
EXPECTED = {
    "neudet": {
        "counts": {"train": 1200, "val": 300, "test": 299},
        "names": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
    },
    "aluminum": {
        "counts": {"train": 1000, "val": 200, "test": 200},
        "names": ["zhen_kong", "ca_shang", "zang_wu", "zhe_zhou"],
    },
    "pcb": {
        "counts": {"train": 891, "val": 120, "test": 60},
        "names": ["missing_hole", "mouse_bite", "open_circuit", "short", "spurious_copper", "spur"],
    },
}

EPOCHS = 250
BATCH = 32
IMGSZ = 640
PATIENCE = 0
WORKERS = 8


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


def run_dir(dataset: str) -> Path:
    return PROJECT / dataset / f"yolo11n_seed{SEED}"


def training_report(dataset: str) -> Path:
    return PROJECT / f"training_{dataset}_yolo11n_seed{SEED}_report.json"


def independent_report(dataset: str) -> Path:
    return PROJECT / f"independent_{dataset}_yolo11n_seed{SEED}_report.json"


def result_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    return max(0, len(rows) - 1)


def split_count(value) -> int:
    paths = value if isinstance(value, list) else [value]
    total = 0
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for raw in paths:
        path = Path(str(raw))
        if path.is_file():
            total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif path.is_dir():
            total += sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in image_exts)
        else:
            raise FileNotFoundError(path)
    return total


def ensure_official_weights() -> None:
    if OFFICIAL.is_file():
        return
    log("yolo11n.pt not found; downloading official Ultralytics checkpoint")
    cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        YOLO("yolo11n.pt")
    finally:
        os.chdir(cwd)
    if not OFFICIAL.is_file():
        raise FileNotFoundError(f"official checkpoint download did not create {OFFICIAL}")


def build_check() -> int:
    from ultralytics.data.utils import check_det_dataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ensure_official_weights()

    checked_datasets = {}
    for dataset, data in DATASETS.items():
        if not data.is_file():
            raise FileNotFoundError(data)
        checked = check_det_dataset(str(data), autodownload=False)
        raw_names = checked["names"]
        names = [str(raw_names[i]) for i in sorted(raw_names)] if isinstance(raw_names, dict) else list(map(str, raw_names))
        counts = {split: split_count(checked[split]) for split in ("train", "val", "test")}
        expected = EXPECTED[dataset]
        if names != expected["names"]:
            raise RuntimeError(f"{dataset} class mismatch: expected={expected['names']} actual={names}")
        if counts != expected["counts"]:
            raise RuntimeError(f"{dataset} split mismatch: expected={expected['counts']} actual={counts}")
        checked_datasets[dataset] = {
            "yaml": str(data), "nc": int(checked["nc"]), "names": names,
            "split_counts": counts, "train": str(checked["train"]),
            "val": str(checked["val"]), "test": str(checked["test"]),
        }
        log(f"DATA_GATE_PASS {dataset} counts={counts} classes={names}")

    wrapper = YOLO(str(OFFICIAL))
    net = wrapper.model.to("cuda:0").eval()
    with torch.inference_mode():
        output = net(torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda:0"))
    det = net.model[-1]
    fused = net.fuse() if hasattr(net, "fuse") else net
    structure = {
        "checkpoint": str(OFFICIAL),
        "model_class": type(net).__name__,
        "detect_class": type(det).__name__,
        "end2end": bool(getattr(fused, "end2end", False)),
        "reg_max": int(getattr(det, "reg_max", -1)),
        "nl": int(getattr(det, "nl", -1)),
        "fused_params_coco_head": int(sum(parameter.numel() for parameter in fused.parameters())),
        "forward_type": type(output).__name__,
    }
    if structure["detect_class"] != "Detect" or structure["nl"] != 3:
        raise RuntimeError(f"unexpected YOLOv11n topology: {structure}")

    report = {
        "status": "pass", "created_at": now(), "cuda": torch.cuda.get_device_name(0),
        "ultralytics": __import__("ultralytics").__version__, "datasets": checked_datasets,
        "model": structure,
        "recipe": {
            "epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ, "seed": SEED,
            "deterministic": True, "cache": False, "resume": False,
            "patience": PATIENCE, "workers": WORKERS, "pretrained": str(OFFICIAL),
        },
    }
    atomic_json(BUILD_REPORT, report)
    log(f"BUILD_GATE_PASS model=YOLOv11n report={BUILD_REPORT}")
    return 0


def train_worker(dataset: str) -> int:
    data = DATASETS[dataset]
    target = run_dir(dataset)
    last = target / "weights" / "last.pt"
    report_path = training_report(dataset)
    report = {
        "status": "running", "started_at": now(), "pid": os.getpid(),
        "dataset": dataset, "model": "yolo11n", "seed": SEED, "data": str(data),
        "recipe": {
            "epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ, "seed": SEED,
            "device": 0, "workers": WORKERS, "cache": False, "resume": False,
            "patience": PATIENCE, "deterministic": True, "pretrained": str(OFFICIAL),
        },
    }
    atomic_json(report_path, report)
    try:
        rows = result_rows(target / "results.csv")
        if last.is_file() and 0 < rows < EPOCHS:
            log(f"TRAIN_RESUME dataset={dataset} completed_epochs={rows} last={last}")
            model = YOLO(str(last))
            model.train(resume=True)
        else:
            log(f"TRAIN_START dataset={dataset} model=yolo11n seed={SEED}")
            model = YOLO(str(OFFICIAL))
            model.train(
                data=str(data), epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ, device=0,
                workers=WORKERS, seed=SEED, deterministic=True, pretrained=True,
                cache=False, resume=False, patience=PATIENCE, project=str(PROJECT / dataset),
                name=f"yolo11n_seed{SEED}", exist_ok=True, plots=False, save=True,
                verbose=False, rect=False, val=True,
            )

        best = target / "weights" / "best.pt"
        last = target / "weights" / "last.pt"
        results = target / "results.csv"
        rows = result_rows(results)
        if not best.is_file() or not last.is_file() or rows != EPOCHS:
            raise RuntimeError(f"incomplete artifacts: best={best.is_file()} last={last.is_file()} epochs={rows}/{EPOCHS}")
        report.update({
            "status": "done", "finished_at": now(), "run_dir": str(target),
            "best": str(best), "last": str(last), "results_csv": str(results),
            "completed_epochs": rows,
        })
        atomic_json(report_path, report)
        log(f"TRAIN_DONE dataset={dataset} epochs={rows} best={best}")
        return 0
    except Exception as exc:
        report.update({"status": "failed", "finished_at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        atomic_json(report_path, report)
        log(f"TRAIN_FAILED dataset={dataset} error={exc!r}")
        return 1


def extract_metrics(result, names: list[str]) -> dict:
    box = result.box
    ap50 = {name: None for name in names}
    ap50_95 = {name: None for name in names}
    for index, class_index in enumerate(box.ap_class_index):
        name = names[int(class_index)]
        ap50[name] = float(box.ap50[index])
        ap50_95[name] = float(box.ap[index])
    missing = [name for name, value in ap50.items() if value is None]
    if missing:
        raise RuntimeError(f"test evaluation missing classes: {missing}")
    return {
        "precision": float(box.mp), "recall": float(box.mr),
        "map50": float(box.map50), "map50_95": float(box.map),
        "class_ap50": ap50, "class_ap50_95": ap50_95,
    }


def benchmark_forward(net: torch.nn.Module, warmup: int = 50, iterations: int = 200) -> dict:
    net = net.to("cuda:0").eval()
    dtype = next(net.parameters()).dtype
    inputs = torch.randn(BATCH, 3, IMGSZ, IMGSZ, device="cuda:0", dtype=dtype)
    torch.cuda.empty_cache()
    with torch.inference_mode():
        for _ in range(warmup):
            net(inputs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            net(inputs)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    fps = BATCH * iterations / elapsed
    return {
        "fps": float(fps), "infer_ms_per_image": float(1000.0 / fps),
        "elapsed_seconds": float(elapsed), "images": BATCH * iterations,
        "batch": BATCH, "imgsz": IMGSZ, "warmup_iterations": warmup,
        "measured_iterations": iterations, "dtype": str(dtype).replace("torch.", ""),
        "scope": "fused model forward only; excludes preprocess and postprocess",
        "device": torch.cuda.get_device_name(0),
    }


def eval_worker(dataset: str) -> int:
    weights = run_dir(dataset) / "weights" / "best.pt"
    output = independent_report(dataset)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    data_doc = yaml.safe_load(DATASETS[dataset].read_text(encoding="utf-8"))
    raw_names = data_doc["names"]
    names = [str(raw_names[i]) for i in sorted(raw_names)] if isinstance(raw_names, dict) else list(map(str, raw_names))

    log(f"INDEPENDENT_EVAL_START dataset={dataset} weights={weights}")
    wrapper = YOLO(str(weights))
    fused = wrapper.model.fuse() if hasattr(wrapper.model, "fuse") else wrapper.model
    wrapper.model = fused
    result = wrapper.val(
        data=str(DATASETS[dataset]), split="test", imgsz=IMGSZ, batch=BATCH,
        device=0, plots=False, verbose=False, rect=False,
        project=str(PROJECT / "independent_val_runs"), name=f"{dataset}_yolo11n_seed{SEED}",
        exist_ok=True,
    )
    metrics = extract_metrics(result, names)
    benchmark = benchmark_forward(wrapper.model)
    det = wrapper.model.model[-1]
    report = {
        "status": "done", "created_at": now(), "dataset": dataset,
        "model": "yolo11n", "seed": SEED, "weights": str(weights),
        "data": str(DATASETS[dataset]),
        "protocol": {
            "truth": "fresh Python process reloads best.pt from disk and evaluates test split",
            "imgsz": IMGSZ, "batch": BATCH, "device": 0, "rect": False,
        },
        "structure": {
            "fused_params": int(sum(parameter.numel() for parameter in fused.parameters())),
            "end2end": bool(getattr(fused, "end2end", False)),
            "reg_max": int(getattr(det, "reg_max", -1)), "nl": int(getattr(det, "nl", -1)),
        },
        "test": metrics, "benchmark": benchmark,
    }
    atomic_json(output, report)
    log(f"INDEPENDENT_EVAL_DONE dataset={dataset} map50={metrics['map50']:.6f} map50_95={metrics['map50_95']:.6f} fps={benchmark['fps']:.2f}")
    return 0


def experiment_done(dataset: str) -> bool:
    report = read_json(independent_report(dataset), {})
    return report.get("status") == "done" and Path(report.get("weights", "")).is_file()


def summarize() -> dict:
    experiments = {}
    rows = []
    for dataset in DATASETS:
        report = read_json(independent_report(dataset), {})
        if report.get("status") != "done":
            continue
        experiments[dataset] = report
        row = {
            "dataset": dataset, "model": "yolo11n", "seed": SEED,
            "test_map50": report["test"]["map50"],
            "test_map50_95": report["test"]["map50_95"],
            "precision": report["test"]["precision"], "recall": report["test"]["recall"],
            "fps": report["benchmark"]["fps"],
            "infer_ms_per_image": report["benchmark"]["infer_ms_per_image"],
            "fused_params": report["structure"]["fused_params"], "weights": report["weights"],
        }
        for name, value in report["test"]["class_ap50"].items():
            row[f"class_ap50::{name}"] = value
        for name, value in report["test"]["class_ap50_95"].items():
            row[f"class_ap50_95::{name}"] = value
        rows.append(row)

    aggregate = {
        "status": "done" if len(experiments) == len(DATASETS) else "partial",
        "created_at": now(), "model": "yolo11n", "seed": SEED,
        "protocol": {
            "training": f"250 epochs / imgsz640 / batch32 / seed{SEED} / deterministic=True / cache=False / resume=False / patience=0 / official yolo11n.pt",
            "truth": "independent fresh Python process reloads best.pt and evaluates test split",
            "fps": "RTX 4090 / fused FP32 / imgsz640 / batch32 / warmup50 / iterations200 / forward only",
        },
        "experiments": experiments,
    }
    atomic_json(REPORT, aggregate)

    class_columns = sorted({key for row in rows for key in row if key.startswith("class_")})
    fields = [
        "dataset", "model", "seed", "test_map50", "test_map50_95", "precision", "recall",
        "fps", "infer_ms_per_image", "fused_params", *class_columns, "weights",
    ]
    with COMPARISON_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# YOLOv11n Seed{SEED} 三数据集基线", "",
        f"- 训练：250 epochs，imgsz=640，batch=32，seed={SEED}，deterministic=True，cache=False，patience=0。",
        "- 初始化：官方 `yolo11n.pt`。", "- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。",
        "- FPS：RTX 4090，fused FP32，batch=32，warmup=50，iterations=200，仅计 forward。", "",
        "## 总体性能", "", "| 数据集 | test mAP50 | test mAP50-95 | FPS | fused params |", "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        report = experiments.get(dataset)
        if report:
            lines.append(f"| {dataset} | {report['test']['map50']:.6f} | {report['test']['map50_95']:.6f} | {report['benchmark']['fps']:.2f} | {report['structure']['fused_params']:,} |")
        else:
            lines.append(f"| {dataset} | 待完成 | 待完成 | 待完成 | 待完成 |")
    for dataset, report in experiments.items():
        lines += ["", f"## {dataset} 逐类 test AP", "", "| 类别 | AP50 | AP50-95 |", "|---|---:|---:|"]
        for name, value in report["test"]["class_ap50"].items():
            lines.append(f"| {name} | {value:.6f} | {report['test']['class_ap50_95'][name]:.6f} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"SUMMARY_WRITTEN status={aggregate['status']} completed={len(experiments)}/{len(DATASETS)} report={REPORT_MD}")
    return aggregate


def orchestrate(only: str | None = None) -> int:
    build_check()
    selected = [only] if only else list(DATASETS)
    if any(dataset not in DATASETS for dataset in selected):
        raise ValueError(f"unknown dataset: {selected}")

    atomic_json(STATUS, {"status": "running", "phase": "train", "started_at": now(), "datasets": selected, "current": None})
    try:
        for dataset in selected:
            if experiment_done(dataset):
                log(f"SKIP_COMPLETE dataset={dataset}")
                continue
            if result_rows(run_dir(dataset) / "results.csv") != EPOCHS:
                atomic_json(STATUS, {"status": "running", "phase": "train", "updated_at": now(), "datasets": selected, "current": dataset})
                rc = subprocess.call([PYTHON, str(Path(__file__).resolve()), "--train-worker", dataset])
                if rc != 0:
                    raise RuntimeError(f"training worker failed: dataset={dataset} rc={rc}")
            atomic_json(STATUS, {"status": "running", "phase": "eval", "updated_at": now(), "datasets": selected, "current": dataset})
            rc = subprocess.call([PYTHON, str(Path(__file__).resolve()), "--eval-worker", dataset])
            if rc != 0:
                raise RuntimeError(f"evaluation worker failed: dataset={dataset} rc={rc}")
            summarize()
        result = summarize()
        completed = [dataset for dataset in DATASETS if experiment_done(dataset)]
        final_status = "done" if len(completed) == len(DATASETS) else "partial"
        atomic_json(STATUS, {"status": final_status, "phase": "done", "finished_at": now(), "completed": completed, "pending": [d for d in DATASETS if d not in completed]})
        return 0 if result["status"] == "done" or only else 1
    except Exception as exc:
        atomic_json(STATUS, {"status": "failed", "phase": "failed", "updated_at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-check", action="store_true")
    parser.add_argument("--train-worker", choices=tuple(DATASETS))
    parser.add_argument("--eval-worker", choices=tuple(DATASETS))
    parser.add_argument("--only", choices=tuple(DATASETS))
    args = parser.parse_args()
    if args.build_check:
        return build_check()
    if args.train_worker:
        return train_worker(args.train_worker)
    if args.eval_worker:
        return eval_worker(args.eval_worker)
    return orchestrate(args.only)


if __name__ == "__main__":
    raise SystemExit(main())
