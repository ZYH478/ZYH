#!/usr/bin/env python
"""Train Ultralytics RT-DETR-HGNetv2-L seeds 1/2/3 on NEU-DET, Aluminum and PCB.

Each dataset/seed runs in its own process and final test metrics come from a
fresh best.pt reload.
"""
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
import yaml
from ultralytics import RTDETR

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
PROJECT = ROOT / "runs_sota_rtdetr_hgnetv2_l_seed123_e250"
SEED0_PROJECT = ROOT / "runs_sota_rtdetr_hgnetv2_l_seed0_e250"
OFFICIAL = ROOT / "rtdetr-l.pt"
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
BATCH = int(os.environ.get("RTDETR_BATCH", "8"))
IMGSZ = 640
SEEDS = (1, 2, 3)
PATIENCE = 0
WORKERS = int(os.environ.get("RTDETR_WORKERS", "8"))
MODEL_NAME = "RT-DETR-HGNetv2-L"


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


def run_dir(dataset: str, seed: int) -> Path:
    return PROJECT / dataset / f"rtdetr_hgnetv2_l_seed{seed}"


def training_report(dataset: str, seed: int) -> Path:
    return PROJECT / f"training_{dataset}_rtdetr_hgnetv2_l_seed{seed}_report.json"


def independent_report(dataset: str, seed: int) -> Path:
    return PROJECT / f"independent_{dataset}_rtdetr_hgnetv2_l_seed{seed}_report.json"


def result_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    return max(0, len(rows) - 1)


def split_count(value) -> int:
    paths = value if isinstance(value, list) else [value]
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    total = 0
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
    if not OFFICIAL.is_file() or OFFICIAL.stat().st_size < 60_000_000:
        raise FileNotFoundError(f"official checkpoint is missing or incomplete: {OFFICIAL}")


def names_from_yaml(path: Path) -> list[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    return [str(raw[i]) for i in sorted(raw)] if isinstance(raw, dict) else list(map(str, raw))


def build_check() -> int:
    from ultralytics.data.utils import check_det_dataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ensure_official_weights()
    checked_datasets = {}
    for dataset, data in DATASETS.items():
        checked = check_det_dataset(str(data), autodownload=False)
        names = names_from_yaml(data)
        counts = {split: split_count(checked[split]) for split in ("train", "val", "test")}
        expected = EXPECTED[dataset]
        if names != expected["names"] or counts != expected["counts"]:
            raise RuntimeError(f"{dataset} data gate mismatch: names={names} counts={counts}")
        checked_datasets[dataset] = {
            "yaml": str(data),
            "nc": int(checked["nc"]),
            "names": names,
            "split_counts": counts,
            "train": str(checked["train"]),
            "val": str(checked["val"]),
            "test": str(checked["test"]),
        }
        log(f"DATA_GATE_PASS {dataset} counts={counts} classes={names}")

    wrapper = RTDETR(str(OFFICIAL))
    net = wrapper.model.to("cuda:0").eval()
    with torch.inference_mode():
        output = net(torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda:0"))
    decoder = net.model[-1]
    params = int(sum(parameter.numel() for parameter in net.parameters()))
    structure = {
        "checkpoint": str(OFFICIAL),
        "model_class": type(net).__name__,
        "decoder_class": type(decoder).__name__,
        "nc": int(getattr(net, "nc", -1)),
        "params": params,
        "forward_type": type(output).__name__,
    }
    if "RTDETR" not in structure["decoder_class"]:
        raise RuntimeError(f"unexpected RT-DETR topology: {structure}")

    report = {
        "status": "pass",
        "created_at": now(),
        "cuda": torch.cuda.get_device_name(0),
        "ultralytics": __import__("ultralytics").__version__,
        "datasets": checked_datasets,
        "model": structure,
        "recipe": {
            "epochs": EPOCHS,
            "batch": BATCH,
            "imgsz": IMGSZ,
            "seeds": list(SEEDS),
            "deterministic": False,
            "cache": False,
            "resume": False,
            "patience": PATIENCE,
            "workers": WORKERS,
            "pretrained": str(OFFICIAL),
        },
    }
    atomic_json(BUILD_REPORT, report)
    log(f"BUILD_GATE_PASS model={MODEL_NAME} params={params:,} report={BUILD_REPORT}")
    return 0


def train_worker(dataset: str, seed: int) -> int:
    data = DATASETS[dataset]
    target = run_dir(dataset, seed)
    last = target / "weights" / "last.pt"
    report_path = training_report(dataset, seed)
    report = {
        "status": "running",
        "started_at": now(),
        "pid": os.getpid(),
        "dataset": dataset,
        "model": MODEL_NAME,
        "seed": seed,
        "data": str(data),
        "recipe": {
            "epochs": EPOCHS,
            "batch": BATCH,
            "imgsz": IMGSZ,
            "seed": seed,
            "device": 0,
            "workers": WORKERS,
            "cache": False,
            "resume": False,
            "deterministic": False,
            "patience": PATIENCE,
            "pretrained": str(OFFICIAL),
        },
    }
    atomic_json(report_path, report)
    try:
        rows = result_rows(target / "results.csv")
        if last.is_file() and 0 < rows < EPOCHS:
            log(f"TRAIN_RESUME dataset={dataset} seed={seed} completed_epochs={rows} last={last}")
            model = RTDETR(str(last))
            model.train(resume=True)
        else:
            log(f"TRAIN_START dataset={dataset} model={MODEL_NAME} seed={seed} batch={BATCH}")
            model = RTDETR(str(OFFICIAL))
            model.train(
                data=str(data),
                epochs=EPOCHS,
                batch=BATCH,
                imgsz=IMGSZ,
                device=0,
                workers=WORKERS,
                seed=seed,
                deterministic=False,
                pretrained=True,
                cache=False,
                resume=False,
                patience=PATIENCE,
                project=str(PROJECT / dataset),
                name=f"rtdetr_hgnetv2_l_seed{seed}",
                exist_ok=True,
                plots=False,
                save=True,
                verbose=False,
                rect=False,
                val=True,
            )

        best = target / "weights" / "best.pt"
        last = target / "weights" / "last.pt"
        results = target / "results.csv"
        rows = result_rows(results)
        if not best.is_file() or not last.is_file() or rows != EPOCHS:
            raise RuntimeError(f"incomplete artifacts: best={best.is_file()} last={last.is_file()} epochs={rows}/{EPOCHS}")
        report.update(
            {
                "status": "done",
                "finished_at": now(),
                "run_dir": str(target),
                "best": str(best),
                "last": str(last),
                "results_csv": str(results),
                "completed_epochs": rows,
            }
        )
        atomic_json(report_path, report)
        log(f"TRAIN_DONE dataset={dataset} seed={seed} epochs={rows} best={best}")
        return 0
    except Exception as exc:
        report.update({"status": "failed", "finished_at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        atomic_json(report_path, report)
        log(f"TRAIN_FAILED dataset={dataset} seed={seed} error={exc!r}")
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
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "class_ap50": ap50,
        "class_ap50_95": ap50_95,
    }


def benchmark_forward(net: torch.nn.Module, warmup: int = 20, iterations: int = 100) -> dict:
    net = net.to("cuda:0").eval()
    dtype = next(net.parameters()).dtype
    inputs = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda:0", dtype=dtype)
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
    fps = iterations / elapsed
    return {
        "fps": float(fps),
        "infer_ms_per_image": float(1000.0 / fps),
        "elapsed_seconds": float(elapsed),
        "images": iterations,
        "batch": 1,
        "imgsz": IMGSZ,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "dtype": str(dtype).replace("torch.", ""),
        "scope": "fused model forward only; excludes preprocess and postprocess",
        "device": torch.cuda.get_device_name(0),
    }


def eval_worker(dataset: str, seed: int) -> int:
    weights = run_dir(dataset, seed) / "weights" / "best.pt"
    output = independent_report(dataset, seed)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    names = names_from_yaml(DATASETS[dataset])
    log(f"INDEPENDENT_EVAL_START dataset={dataset} seed={seed} weights={weights}")
    wrapper = RTDETR(str(weights))
    fused = wrapper.model.fuse() if hasattr(wrapper.model, "fuse") else wrapper.model
    wrapper.model = fused
    result = wrapper.val(
        data=str(DATASETS[dataset]),
        split="test",
        imgsz=IMGSZ,
        batch=1,
        device=0,
        plots=False,
        verbose=False,
        rect=False,
        project=str(PROJECT / "independent_val_runs"),
        name=f"{dataset}_rtdetr_hgnetv2_l_seed{seed}",
        exist_ok=True,
    )
    metrics = extract_metrics(result, names)
    benchmark = benchmark_forward(fused)
    report = {
        "status": "done",
        "created_at": now(),
        "dataset": dataset,
        "model": MODEL_NAME,
        "seed": seed,
        "weights": str(weights),
        "data": str(DATASETS[dataset]),
        "protocol": {
            "truth": "fresh Python process reloads best.pt from disk and evaluates test split",
            "imgsz": IMGSZ,
            "batch": 1,
            "device": 0,
            "rect": False,
        },
        "structure": {
            "fused_params": int(sum(parameter.numel() for parameter in fused.parameters())),
            "decoder_class": type(fused.model[-1]).__name__,
        },
        "test": metrics,
        "benchmark": benchmark,
    }
    atomic_json(output, report)
    log(f"INDEPENDENT_EVAL_DONE dataset={dataset} seed={seed} map50={metrics['map50']:.6f} map50_95={metrics['map50_95']:.6f} fps={benchmark['fps']:.2f}")
    return 0


def experiment_done(dataset: str, seed: int) -> bool:
    report = read_json(independent_report(dataset, seed), {})
    return report.get("status") == "done" and Path(report.get("weights", "")).is_file()


def metric_stats(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def load_experiments() -> dict:
    experiments = {}
    for dataset in DATASETS:
        per_seed = {}
        for seed in SEEDS:
            report = read_json(independent_report(dataset, seed), {})
            if report.get("status") == "done":
                report = dict(report)
                report["source_report"] = str(independent_report(dataset, seed))
                per_seed[f"seed{seed}"] = report
        experiments[dataset] = per_seed
    return experiments


def dataset_summary(per_seed: dict) -> dict:
    done = sorted(per_seed.values(), key=lambda item: int(item["seed"]))
    if not done:
        return {"n": 0, "seeds": []}
    summary = {
        "n": len(done),
        "seeds": [int(item["seed"]) for item in done],
        "precision": metric_stats([item["test"]["precision"] for item in done]),
        "recall": metric_stats([item["test"]["recall"] for item in done]),
        "test_map50": metric_stats([item["test"]["map50"] for item in done]),
        "test_map50_95": metric_stats([item["test"]["map50_95"] for item in done]),
        "fps": metric_stats([item["benchmark"]["fps"] for item in done]),
        "infer_ms_per_image": metric_stats([item["benchmark"]["infer_ms_per_image"] for item in done]),
        "fused_params": sorted({int(item["structure"]["fused_params"]) for item in done}),
        "class_ap50": {},
        "class_ap50_95": {},
    }
    names = list(done[0]["test"]["class_ap50"])
    for name in names:
        summary["class_ap50"][name] = metric_stats([item["test"]["class_ap50"][name] for item in done])
        summary["class_ap50_95"][name] = metric_stats([item["test"]["class_ap50_95"][name] for item in done])
    return summary


def summarize() -> dict:
    experiments = load_experiments()
    summaries = {dataset: dataset_summary(per_seed) for dataset, per_seed in experiments.items()}
    completed_total = sum(len(per_seed) for per_seed in experiments.values())
    expected_total = len(DATASETS) * len(SEEDS)
    aggregate = {
        "status": "done" if completed_total == expected_total else "partial",
        "created_at": now(),
        "model": MODEL_NAME,
        "seeds": list(SEEDS),
        "protocol": {
            "training": f"250 epochs / imgsz640 / batch{BATCH} / seeds1,2,3 / deterministic=False / cache=False / resume=False / patience=0 / official rtdetr-l.pt",
            "seed0_reference": str(SEED0_PROJECT),
            "truth": "independent fresh Python process reloads best.pt and evaluates test split",
            "fps": "RTX 4090 / fused FP32 / imgsz640 / batch1 / warmup20 / iterations100 / forward only",
        },
        "completed": completed_total,
        "expected": expected_total,
        "experiments": experiments,
        "summary": summaries,
    }
    atomic_json(REPORT, aggregate)

    rows = []
    for dataset, per_seed in experiments.items():
        for key, report in sorted(per_seed.items(), key=lambda item: int(item[0].replace("seed", ""))):
            row = {
                "dataset": dataset,
                "model": MODEL_NAME,
                "seed": report["seed"],
                "test_map50": report["test"]["map50"],
                "test_map50_95": report["test"]["map50_95"],
                "precision": report["test"]["precision"],
                "recall": report["test"]["recall"],
                "fps": report["benchmark"]["fps"],
                "infer_ms_per_image": report["benchmark"]["infer_ms_per_image"],
                "fused_params": report["structure"]["fused_params"],
                "source_report": report["source_report"],
                "weights": report["weights"],
            }
            for name, value in report["test"]["class_ap50"].items():
                row[f"class_ap50::{name}"] = value
            for name, value in report["test"]["class_ap50_95"].items():
                row[f"class_ap50_95::{name}"] = value
            rows.append(row)
    class_columns = sorted({key for row in rows for key in row if key.startswith("class_")})
    fields = ["dataset", "model", "seed", "test_map50", "test_map50_95", "precision", "recall", "fps", "infer_ms_per_image", "fused_params", *class_columns, "source_report", "weights"]
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {MODEL_NAME} Seed1/2/3 三数据集报告",
        "",
        f"- 训练：250 epochs，imgsz=640，batch={BATCH}，seeds=1/2/3，deterministic=False，cache=False，patience=0。",
        "- 初始化：官方 Ultralytics `rtdetr-l.pt`（HGNetv2-L）。",
        "- 真值：独立新 Python 进程重载 `best.pt` 后评测 test split。",
        "- FPS：RTX 4090，fused FP32，batch=1，warmup=20，iterations=100，仅计 forward。",
        "",
        "## 总体性能",
        "",
        "| 数据集 | seeds 完成数 | test mAP50 | test mAP50-95 | FPS | fused params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        per_seed = experiments.get(dataset, {})
        if per_seed:
            reports = list(per_seed.values())
            m50 = [item["test"]["map50"] for item in reports]
            m95 = [item["test"]["map50_95"] for item in reports]
            fps = [item["benchmark"]["fps"] for item in reports]
            params = sorted({int(item["structure"]["fused_params"]) for item in reports})
            lines.append(f"| {dataset} | {len(reports)} | {statistics.mean(m50):.6f} | {statistics.mean(m95):.6f} | {statistics.mean(fps):.2f} | {', '.join(f'{value:,}' for value in params)} |")
        else:
            lines.append(f"| {dataset} | 0 | 待完成 | 待完成 | 待完成 | 待完成 |")
    for dataset, per_seed in experiments.items():
        for seed_name, report in sorted(per_seed.items(), key=lambda item: int(item[0].replace("seed", ""))):
            lines += ["", f"## {dataset} {seed_name} 逐类 test AP", "", "| 类别 | AP50 | AP50-95 |", "|---|---:|---:|"]
            for name, value in report["test"]["class_ap50"].items():
                lines.append(f"| {name} | {value:.6f} | {report['test']['class_ap50_95'][name]:.6f} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"SUMMARY_WRITTEN status={aggregate['status']} completed={completed_total}/{expected_total} report={REPORT_MD}")
    return aggregate


def orchestrate(only: str | None = None) -> int:
    build_check()
    selected = [only] if only else list(DATASETS)
    if any(dataset not in DATASETS for dataset in selected):
        raise ValueError(f"unknown dataset: {selected}")
    atomic_json(STATUS, {
        "status": "running",
        "phase": "train",
        "started_at": now(),
        "datasets": selected,
        "seeds": list(SEEDS),
        "current": None,
        "model": MODEL_NAME,
        "seed0_source": str(SEED0_PROJECT),
    })
    try:
        for dataset in selected:
            for seed in SEEDS:
                if experiment_done(dataset, seed):
                    log(f"SKIP_COMPLETE dataset={dataset} seed={seed}")
                    continue
                if result_rows(run_dir(dataset, seed) / "results.csv") != EPOCHS:
                    atomic_json(STATUS, {
                        "status": "running",
                        "phase": "train",
                        "updated_at": now(),
                        "datasets": selected,
                        "seeds": list(SEEDS),
                        "current": {"dataset": dataset, "seed": seed},
                        "model": MODEL_NAME,
                    })
                    rc = subprocess.call([PYTHON, str(Path(__file__).resolve()), "--train-worker", dataset, "--seed", str(seed)])
                    if rc != 0:
                        raise RuntimeError(f"training worker failed: dataset={dataset} seed={seed} rc={rc}")
                atomic_json(STATUS, {
                    "status": "running",
                    "phase": "eval",
                    "updated_at": now(),
                    "datasets": selected,
                    "seeds": list(SEEDS),
                    "current": {"dataset": dataset, "seed": seed},
                    "model": MODEL_NAME,
                })
                rc = subprocess.call([PYTHON, str(Path(__file__).resolve()), "--eval-worker", dataset, "--seed", str(seed)])
                if rc != 0:
                    raise RuntimeError(f"evaluation worker failed: dataset={dataset} seed={seed} rc={rc}")
                summarize()
        result = summarize()
        completed_new = [{"dataset": dataset, "seed": seed} for dataset in DATASETS for seed in SEEDS if experiment_done(dataset, seed)]
        pending_new = [{"dataset": dataset, "seed": seed} for dataset in DATASETS for seed in SEEDS if not experiment_done(dataset, seed)]
        atomic_json(STATUS, {
            "status": "done" if not pending_new else "partial",
            "phase": "done",
            "finished_at": now(),
            "completed": completed_new,
            "pending": pending_new,
            "model": MODEL_NAME,
            "seeds": list(SEEDS),
        })
        return 0 if not pending_new or only else 1
    except Exception as exc:
        atomic_json(STATUS, {
            "status": "failed",
            "phase": "failed",
            "updated_at": now(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "model": MODEL_NAME,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-check", action="store_true")
    parser.add_argument("--train-worker", choices=tuple(DATASETS))
    parser.add_argument("--eval-worker", choices=tuple(DATASETS))
    parser.add_argument("--only", choices=tuple(DATASETS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.build_check:
        return build_check()
    if args.train_worker:
        if args.seed is None:
            parser.error("--train-worker requires --seed")
        return train_worker(args.train_worker, args.seed)
    if args.eval_worker:
        if args.seed is None:
            parser.error("--eval-worker requires --seed")
        return eval_worker(args.eval_worker, args.seed)
    return orchestrate(args.only)


if __name__ == "__main__":
    raise SystemExit(main())
