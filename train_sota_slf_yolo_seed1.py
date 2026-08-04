#!/usr/bin/env python
"""Train SLF-YOLO seed1 on NEU-DET, Aluminum, and PCB.

Uses the zacianfans/SLF-YOLO checkout under /root/autodl-tmp/neu-det-yolo26/SLF-YOLO/SLF-YOLO
and records restartable status plus independent fresh-process test reports.
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

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
SLF_CODE = ROOT / "SLF-YOLO" / "SLF-YOLO"
if str(SLF_CODE) not in sys.path:
    sys.path.insert(0, str(SLF_CODE))

import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

PROJECT = ROOT / "runs_sota_slf_yolo_seed1_e250"
CFG = SLF_CODE / "new_cfg" / "yolov8-SlimAsfNeck-CGLU.yaml"
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

MODEL_NAME = "slf_yolo_slimasfneck_cglu_n"
EPOCHS = 250
BATCH = 32
IMGSZ = 640
SEED = 1
PATIENCE = 0
WORKERS = 8


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def log(message):
    print("[{}] {}".format(now(), message), flush=True)


def run_dir(dataset):
    return PROJECT / dataset / "slf_yolo_seed1"


def training_report(dataset):
    return PROJECT / "training_{}_slf_yolo_seed1_report.json".format(dataset)


def independent_report(dataset):
    return PROJECT / "independent_{}_slf_yolo_seed1_report.json".format(dataset)


def result_rows(path):
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    return max(0, len(rows) - 1)


def split_count(value):
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


def dataset_names(dataset):
    data_doc = yaml.safe_load(DATASETS[dataset].read_text(encoding="utf-8"))
    raw = data_doc["names"]
    return [str(raw[i]) for i in sorted(raw)] if isinstance(raw, dict) else list(map(str, raw))


def build_check():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not CFG.is_file():
        raise FileNotFoundError(CFG)

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
            raise RuntimeError("{} class mismatch: expected={} actual={}".format(dataset, expected["names"], names))
        if counts != expected["counts"]:
            raise RuntimeError("{} split mismatch: expected={} actual={}".format(dataset, expected["counts"], counts))
        checked_datasets[dataset] = {
            "yaml": str(data), "nc": int(checked["nc"]), "names": names,
            "split_counts": counts, "train": str(checked["train"]),
            "val": str(checked["val"]), "test": str(checked["test"]),
        }
        log("DATA_GATE_PASS dataset={} counts={} classes={}".format(dataset, counts, names))

    wrapper = YOLO(str(CFG))
    net = wrapper.model.to("cuda:0").eval()
    with torch.inference_mode():
        output = net(torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda:0"))
    det = net.model[-1]
    try:
        fused = net.fuse() if hasattr(net, "fuse") else net
    except Exception:
        fused = net
    structure = {
        "repo": "https://github.com/zacianfans/SLF-YOLO",
        "repo_head": None,
        "code_root": str(SLF_CODE),
        "cfg": str(CFG),
        "model_name": MODEL_NAME,
        "model_class": type(net).__name__,
        "detect_class": type(det).__name__,
        "end2end": bool(getattr(fused, "end2end", False)),
        "reg_max": int(getattr(det, "reg_max", -1)),
        "nl": int(getattr(det, "nl", -1)),
        "fused_params_default_head": int(sum(p.numel() for p in fused.parameters())),
        "forward_type": type(output).__name__,
        "scale_note": "cfg has scales; Ultralytics 8.2.91 warns no explicit scale and assumes scale='n'",
    }
    if structure["detect_class"] != "Detect" or structure["nl"] != 3:
        raise RuntimeError("unexpected SLF-YOLO topology: {}".format(structure))

    try:
        rc = subprocess.check_output(["git", "-C", str(SLF_CODE.parent), "rev-parse", "HEAD"], text=True).strip()
        structure["repo_head"] = rc
    except Exception:
        pass

    report = {
        "status": "pass", "created_at": now(), "cuda": torch.cuda.get_device_name(0),
        "python": sys.version, "torch": torch.__version__,
        "ultralytics_file": __import__("ultralytics").__file__,
        "ultralytics_version": __import__("ultralytics").__version__,
        "datasets": checked_datasets, "model": structure,
        "recipe": {
            "epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ, "seed": SEED,
            "deterministic": True, "cache": False, "resume": False,
            "patience": PATIENCE, "workers": WORKERS, "pretrained": False,
            "optimizer": "SGD", "amp": False, "close_mosaic": 10,
        },
    }
    atomic_json(BUILD_REPORT, report)
    log("BUILD_GATE_PASS model={} report={}".format(MODEL_NAME, BUILD_REPORT))
    return 0


def train_worker(dataset):
    data = DATASETS[dataset]
    target = run_dir(dataset)
    last = target / "weights" / "last.pt"
    report_path = training_report(dataset)
    report = {
        "status": "running", "started_at": now(), "pid": os.getpid(),
        "dataset": dataset, "model": MODEL_NAME, "seed": SEED, "cfg": str(CFG), "data": str(data),
        "recipe": {
            "epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ, "seed": SEED,
            "device": 0, "workers": WORKERS, "cache": False, "resume": False,
            "patience": PATIENCE, "deterministic": True, "pretrained": False,
            "optimizer": "SGD", "amp": False, "close_mosaic": 10,
        },
    }
    atomic_json(report_path, report)
    try:
        rows = result_rows(target / "results.csv")
        if last.is_file() and 0 < rows < EPOCHS:
            log("TRAIN_RESUME dataset={} completed_epochs={} last={}".format(dataset, rows, last))
            model = YOLO(str(last))
            model.train(resume=True)
        elif rows == EPOCHS and (target / "weights" / "best.pt").is_file():
            log("TRAIN_ALREADY_DONE dataset={}".format(dataset))
        else:
            log("TRAIN_START dataset={} model={} seed=1".format(dataset, MODEL_NAME))
            model = YOLO(str(CFG))
            model.train(
                data=str(data), epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ, device=0,
                workers=WORKERS, seed=SEED, deterministic=True, pretrained=False,
                cache=False, resume=False, patience=PATIENCE, project=str(PROJECT / dataset),
                name="slf_yolo_seed1", exist_ok=True, plots=False, save=True,
                verbose=False, rect=False, val=True, optimizer="SGD", amp=False,
                close_mosaic=10,
            )

        best = target / "weights" / "best.pt"
        last = target / "weights" / "last.pt"
        results = target / "results.csv"
        rows = result_rows(results)
        if not best.is_file() or not last.is_file() or rows != EPOCHS:
            raise RuntimeError("incomplete artifacts: best={} last={} epochs={}/{}".format(best.is_file(), last.is_file(), rows, EPOCHS))
        report.update({
            "status": "done", "finished_at": now(), "run_dir": str(target),
            "best": str(best), "last": str(last), "results_csv": str(results),
            "completed_epochs": rows,
        })
        atomic_json(report_path, report)
        log("TRAIN_DONE dataset={} epochs={} best={}".format(dataset, rows, best))
        return 0
    except Exception as exc:
        report.update({"status": "failed", "finished_at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        atomic_json(report_path, report)
        log("TRAIN_FAILED dataset={} error={!r}".format(dataset, exc))
        return 1


def extract_metrics(result, names):
    box = result.box
    ap50 = {name: None for name in names}
    ap50_95 = {name: None for name in names}
    for index, class_index in enumerate(box.ap_class_index):
        name = names[int(class_index)]
        ap50[name] = float(box.ap50[index])
        ap50_95[name] = float(box.ap[index])
    missing = [name for name, value in ap50.items() if value is None]
    if missing:
        raise RuntimeError("test evaluation missing classes: {}".format(missing))
    return {
        "precision": float(box.mp), "recall": float(box.mr),
        "map50": float(box.map50), "map50_95": float(box.map),
        "class_ap50": ap50, "class_ap50_95": ap50_95,
    }


def benchmark_forward(net, warmup=50, iterations=200):
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


def eval_worker(dataset):
    weights = run_dir(dataset) / "weights" / "best.pt"
    output = independent_report(dataset)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    names = dataset_names(dataset)

    log("INDEPENDENT_EVAL_START dataset={} weights={}".format(dataset, weights))
    wrapper = YOLO(str(weights))
    try:
        fused = wrapper.model.fuse() if hasattr(wrapper.model, "fuse") else wrapper.model
        wrapper.model = fused
    except Exception:
        fused = wrapper.model
    result = wrapper.val(
        data=str(DATASETS[dataset]), split="test", imgsz=IMGSZ, batch=BATCH,
        device=0, plots=False, verbose=False, rect=False,
        project=str(PROJECT / "independent_val_runs"), name="{}_slf_yolo_seed1".format(dataset),
        exist_ok=True,
    )
    metrics = extract_metrics(result, names)
    benchmark = benchmark_forward(wrapper.model)
    det = wrapper.model.model[-1]
    report = {
        "status": "done", "created_at": now(), "dataset": dataset,
        "model": MODEL_NAME, "seed": SEED, "weights": str(weights),
        "data": str(DATASETS[dataset]), "cfg": str(CFG),
        "protocol": {
            "truth": "fresh Python process reloads best.pt from disk and evaluates test split",
            "imgsz": IMGSZ, "batch": BATCH, "device": 0, "rect": False,
        },
        "structure": {
            "fused_params": int(sum(p.numel() for p in fused.parameters())),
            "end2end": bool(getattr(fused, "end2end", False)),
            "reg_max": int(getattr(det, "reg_max", -1)), "nl": int(getattr(det, "nl", -1)),
        },
        "test": metrics, "benchmark": benchmark,
    }
    atomic_json(output, report)
    log("INDEPENDENT_EVAL_DONE dataset={} map50={:.6f} map50_95={:.6f} fps={:.2f}".format(dataset, metrics["map50"], metrics["map50_95"], benchmark["fps"]))
    return 0


def experiment_done(dataset):
    report = read_json(independent_report(dataset), {})
    return report.get("status") == "done" and Path(report.get("weights", "")).is_file()


def summarize():
    experiments = {}
    rows = []
    for dataset in DATASETS:
        report = read_json(independent_report(dataset), {})
        if report.get("status") != "done":
            continue
        experiments[dataset] = report
        row = {
            "dataset": dataset, "model": MODEL_NAME, "seed": SEED,
            "test_map50": report["test"]["map50"],
            "test_map50_95": report["test"]["map50_95"],
            "precision": report["test"]["precision"], "recall": report["test"]["recall"],
            "fps": report["benchmark"]["fps"],
            "infer_ms_per_image": report["benchmark"]["infer_ms_per_image"],
            "fused_params": report["structure"]["fused_params"], "weights": report["weights"],
        }
        for name, value in report["test"]["class_ap50"].items():
            row["class_ap50::{}".format(name)] = value
        for name, value in report["test"]["class_ap50_95"].items():
            row["class_ap50_95::{}".format(name)] = value
        rows.append(row)

    aggregate = {
        "status": "done" if len(experiments) == len(DATASETS) else "partial",
        "created_at": now(), "model": MODEL_NAME, "seed": SEED,
        "protocol": {
            "training": "250 epochs / imgsz640 / batch32 / seed1 / deterministic=True / cache=False / resume=False / patience=0 / SLF-YOLO yolov8-SlimAsfNeck-CGLU.yaml / scale n / from scratch / SGD / amp=False / close_mosaic=10",
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
        "# SLF-YOLO Seed1 三数据集训练结果", "",
        "- 模型：`yolov8-SlimAsfNeck-CGLU.yaml`（Ultralytics 默认 `scale='n'`，本轮记作 `slf_yolo_slimasfneck_cglu_n`）。",
        "- 训练：250 epochs、imgsz=640、batch=32、seed=1、deterministic=True、cache=False、patience=0、optimizer=SGD、amp=False、close_mosaic=10、from scratch。",
        "- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。",
        "- FPS：RTX 4090、fused FP32、batch=32、warmup=50、iterations=200，仅模型 forward。", "",
        "## 总体指标", "", "| 数据集 | test mAP50 | test mAP50-95 | FPS | fused params |", "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        report = experiments.get(dataset)
        if report:
            lines.append("| {} | {:.6f} | {:.6f} | {:.2f} | {:,} |".format(dataset, report["test"]["map50"], report["test"]["map50_95"], report["benchmark"]["fps"], report["structure"]["fused_params"]))
        else:
            lines.append("| {} | 未完成 | 未完成 | 未完成 | 未完成 |".format(dataset))
    for dataset, report in experiments.items():
        lines += ["", "## {} 逐类 test AP".format(dataset), "", "| 类别 | AP50 | AP50-95 |", "|---|---:|---:|"]
        for name, value in report["test"]["class_ap50"].items():
            lines.append("| {} | {:.6f} | {:.6f} |".format(name, value, report["test"]["class_ap50_95"][name]))
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("SUMMARY_WRITTEN status={} completed={}/{} report={}".format(aggregate["status"], len(experiments), len(DATASETS), REPORT_MD))
    return aggregate


def orchestrate(only=None):
    build_check()
    selected = [only] if only else list(DATASETS)
    if any(dataset not in DATASETS for dataset in selected):
        raise ValueError("unknown dataset: {}".format(selected))

    failures = []
    atomic_json(STATUS, {"status": "running", "phase": "train", "started_at": now(), "datasets": selected, "current": None, "completed": [], "failures": []})
    for dataset in selected:
        try:
            if experiment_done(dataset):
                log("SKIP_COMPLETE dataset={}".format(dataset))
                continue
            if result_rows(run_dir(dataset) / "results.csv") != EPOCHS:
                atomic_json(STATUS, {"status": "running", "phase": "train", "updated_at": now(), "datasets": selected, "current": dataset, "completed": [d for d in DATASETS if experiment_done(d)], "failures": failures})
                rc = subprocess.call([PYTHON, str(Path(__file__).resolve()), "--train-worker", dataset])
                if rc != 0:
                    raise RuntimeError("training worker failed: dataset={} rc={}".format(dataset, rc))
            atomic_json(STATUS, {"status": "running", "phase": "eval", "updated_at": now(), "datasets": selected, "current": dataset, "completed": [d for d in DATASETS if experiment_done(d)], "failures": failures})
            rc = subprocess.call([PYTHON, str(Path(__file__).resolve()), "--eval-worker", dataset])
            if rc != 0:
                raise RuntimeError("evaluation worker failed: dataset={} rc={}".format(dataset, rc))
            summarize()
        except Exception as exc:
            failure = {"dataset": dataset, "error": repr(exc), "traceback": traceback.format_exc(), "time": now()}
            failures.append(failure)
            log("DATASET_FAILED dataset={} error={!r}".format(dataset, exc))
            continue
    result = summarize()
    completed = [dataset for dataset in DATASETS if experiment_done(dataset)]
    pending = [d for d in DATASETS if d not in completed and all(f.get("dataset") != d for f in failures)]
    final_status = "done" if len(completed) == len(DATASETS) and not failures else ("failed" if failures else "partial")
    atomic_json(STATUS, {"status": final_status, "phase": "done" if final_status == "done" else final_status, "finished_at": now(), "completed": completed, "pending": pending, "failures": failures})
    return 0 if final_status == "done" or only else 1


def main():
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

