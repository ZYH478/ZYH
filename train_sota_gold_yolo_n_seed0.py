#!/usr/bin/env python
"""Train Gold-YOLO-n for one selectable seed on NEU-DET, Aluminum, and PCB, then test best checkpoints.

Official Gold-YOLO base recipe is used (`--fuse_ab`) without self-distillation.
The script is restartable at dataset granularity; final metrics are produced by
fresh Python subprocesses that reload `best_ckpt.pt` from disk.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
GOLD_ROOT = ROOT / "Gold-YOLO-src" / "Detection" / "Gold-YOLO"
SEED = int(os.environ.get("SOTA_SEED", "0"))
PROJECT = Path(os.environ.get("SOTA_PROJECT", str(ROOT / f"runs_sota_gold_yolo_n_seed{SEED}_e250")))
DATA_ROOT = ROOT / "gold_yolo_data"
STATUS = PROJECT / "status.json"
REPORT = PROJECT / "report.json"
REPORT_MD = PROJECT / "report.md"
COMPARISON_CSV = PROJECT / "comparison.csv"
BUILD_REPORT = PROJECT / "build_check_report.json"
PYTHON = sys.executable
MODEL_NAME = "Gold-YOLO-n"
CONFIG = GOLD_ROOT / "configs" / "gold_yolo-n.py"
EPOCHS = 250
BATCH = int(os.environ.get("GOLD_YOLO_BATCH", "32"))
IMGSZ = 640
WORKERS = int(os.environ.get("GOLD_YOLO_WORKERS", "8"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_gold_import_path() -> None:
    gold_path = str(GOLD_ROOT)
    if gold_path not in sys.path:
        sys.path.insert(0, gold_path)


def gold_subprocess_env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    gold_path = str(GOLD_ROOT)
    env["PYTHONPATH"] = gold_path if not existing else gold_path + os.pathsep + existing
    env["GOLD_YOLO_SEED"] = str(SEED)
    return env

SOURCE_DATASETS = {
    "neudet": ROOT / "dataset" / "neu-det.yaml",
    "aluminum": ROOT / "runs_generalization_aluminum_pcb_yolo26_msdgs_e250" / "datasets" / "aluminum.yaml",
    "pcb": ROOT / "runs_generalization_aluminum_pcb_yolo26_msdgs_e250" / "datasets" / "pcb.yaml",
}
EXPECTED = {
    "neudet": {"counts": {"train": 1200, "val": 300, "test": 299}, "names": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]},
    "aluminum": {"counts": {"train": 1000, "val": 200, "test": 200}, "names": ["zhen_kong", "ca_shang", "zang_wu", "zhe_zhou"]},
    "pcb": {"counts": {"train": 891, "val": 120, "test": 60}, "names": ["missing_hole", "mouse_bite", "open_circuit", "short", "spurious_copper", "spur"]},
}

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)

def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def read_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))

def names_from_yaml_dict(raw: dict) -> list[str]:
    names = raw["names"]
    return [str(names[i]) for i in sorted(names)] if isinstance(names, dict) else [str(x) for x in names]

def resolve_split(raw: dict, source_yaml: Path, split: str) -> Path:
    value = raw[split]
    if isinstance(value, list):
        raise TypeError(f"{source_yaml}:{split} list split is not supported for Gold-YOLO materialization")
    path = Path(str(value))
    if not path.is_absolute():
        base = Path(str(raw.get("path", source_yaml.parent)))
        if not base.is_absolute():
            base = source_yaml.parent / base
        path = base / path
    return path.resolve()

def image_paths_from_split(path: Path) -> list[Path]:
    if path.is_file():
        return [Path(line.strip()).resolve() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.is_dir():
        return sorted(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(path)

def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")

def safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def materialize_dataset(dataset: str) -> dict:
    source = SOURCE_DATASETS[dataset]
    raw = load_yaml(source)
    names = names_from_yaml_dict(raw)
    expected = EXPECTED[dataset]
    if names != expected["names"]:
        raise RuntimeError(f"{dataset} class mismatch: expected={expected['names']} actual={names}")
    target = DATA_ROOT / dataset
    counts = {}
    links = {}
    missing_labels = {}
    for split in ("train", "val", "test"):
        split_path = resolve_split(raw, source, split)
        images = image_paths_from_split(split_path)
        counts[split] = len(images)
        if counts[split] != expected["counts"][split]:
            raise RuntimeError(f"{dataset} {split} count mismatch: expected={expected['counts'][split]} actual={counts[split]}")
        seen = set()
        missing_labels[split] = 0
        for image in images:
            if not image.is_file():
                raise FileNotFoundError(image)
            label = label_for_image(image)
            name = image.name
            if name in seen:
                digest = hashlib.sha1(str(image).encode()).hexdigest()[:10]
                name = f"{image.stem}_{digest}{image.suffix}"
            seen.add(name)
            safe_link(image, target / "images" / split / name)
            target_label = target / "labels" / split / Path(name).with_suffix(".txt")
            if label.is_file():
                safe_link(label, target_label)
            else:
                target_label.parent.mkdir(parents=True, exist_ok=True)
                if target_label.exists() or target_label.is_symlink():
                    target_label.unlink()
                target_label.write_text("", encoding="utf-8")
                missing_labels[split] += 1
        links[split] = str((target / "images" / split).resolve())
    gold_yaml = DATA_ROOT / f"{dataset}.yaml"
    gold = {"train": links["train"], "val": links["val"], "test": links["test"], "nc": len(names), "is_coco": False, "names": names}
    gold_yaml.parent.mkdir(parents=True, exist_ok=True)
    gold_yaml.write_text(yaml.safe_dump(gold, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log(f"DATA_GATE_PASS {dataset} counts={counts} missing_labels={missing_labels} yaml={gold_yaml}")
    return {"source_yaml": str(source), "gold_yaml": str(gold_yaml), "names": names, "split_counts": counts, "missing_labels": missing_labels, **links}

def prepare_data() -> dict:
    return {dataset: materialize_dataset(dataset) for dataset in SOURCE_DATASETS}

def run_dir(dataset: str) -> Path:
    return PROJECT / dataset / f"gold_yolo_n_seed{SEED}"

def training_report(dataset: str) -> Path:
    return PROJECT / f"training_{dataset}_gold_yolo_n_seed{SEED}_report.json"

def independent_report(dataset: str) -> Path:
    return PROJECT / f"independent_{dataset}_gold_yolo_n_seed{SEED}_report.json"

def checkpoint_epoch(path: Path) -> int:
    if not path.is_file():
        return 0
    ckpt = torch.load(path, map_location="cpu")
    epoch = int(ckpt.get("epoch", -1))
    return epoch + 1

def build_check() -> int:
    if not GOLD_ROOT.is_dir() or not CONFIG.is_file():
        raise FileNotFoundError(f"Gold-YOLO source/config missing: {GOLD_ROOT} {CONFIG}")
    datasets = prepare_data()
    old_cwd = os.getcwd(); os.chdir(GOLD_ROOT)
    try:
        ensure_gold_import_path()
        from yolov6.utils.config import Config
        from yolov6.models.yolo import build_model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg = Config.fromfile(str(CONFIG))
        if not hasattr(cfg, "training_mode"):
            setattr(cfg, "training_mode", "repvgg")
        model = build_model(cfg, EXPECTED["neudet"]["names"].__len__(), device, fuse_ab=True).eval()
        params = int(sum(p.numel() for p in model.parameters()))
        with torch.inference_mode():
            _ = model(torch.randn(1, 3, IMGSZ, IMGSZ, device=device))
    finally:
        os.chdir(old_cwd)
    report = {"status": "pass", "created_at": now(), "model": MODEL_NAME, "official_source": str(GOLD_ROOT), "config": str(CONFIG), "git_commit": git_commit(), "datasets": datasets, "structure": {"params_neudet_head_build": params, "fuse_ab": True}, "recipe": {"epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ, "seed": SEED, "cache": False, "workers": WORKERS, "base_training": True, "self_distillation": False}}
    atomic_json(BUILD_REPORT, report)
    log(f"BUILD_GATE_PASS model={MODEL_NAME} params={params:,} report={BUILD_REPORT}")
    return 0

def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT / "Gold-YOLO-src"), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None

def train_worker(dataset: str) -> int:
    data_yaml = DATA_ROOT / f"{dataset}.yaml"
    target = run_dir(dataset)
    last = target / "weights" / "last_ckpt.pt"
    best = target / "weights" / "best_ckpt.pt"
    report_path = training_report(dataset)
    report = {"status": "running", "started_at": now(), "pid": os.getpid(), "dataset": dataset, "model": MODEL_NAME, "seed": SEED, "data": str(data_yaml), "recipe": {"epochs": EPOCHS, "batch": BATCH, "imgsz": IMGSZ, "workers": WORKERS, "device": 0, "fuse_ab": True, "base_training": True, "self_distillation": False}}
    atomic_json(report_path, report)
    try:
        completed = checkpoint_epoch(last)
        if best.is_file() and last.is_file() and completed >= EPOCHS:
            log(f"TRAIN_ALREADY_DONE dataset={dataset} epochs={completed}")
        else:
            if last.is_file() and 0 < completed < EPOCHS:
                cmd = [PYTHON, "tools/train.py", "--resume", str(last)]
                log(f"TRAIN_RESUME dataset={dataset} completed_epochs={completed} last={last}")
            else:
                cmd = [PYTHON, "tools/train.py", "--batch-size", str(BATCH), "--conf-file", str(CONFIG), "--data-path", str(data_yaml), "--epochs", str(EPOCHS), "--img-size", str(IMGSZ), "--device", "0", "--workers", str(WORKERS), "--output-dir", str(PROJECT / dataset), "--name", f"gold_yolo_n_seed{SEED}", "--eval-final-only", "--fuse_ab", "--bs_per_gpu", str(BATCH)]
                log(f"TRAIN_START dataset={dataset} model={MODEL_NAME} seed={SEED} batch={BATCH}")
            env = gold_subprocess_env()
            rc = subprocess.call(cmd, cwd=str(GOLD_ROOT), env=env)
            if rc != 0:
                raise RuntimeError(f"Gold-YOLO train command failed rc={rc}")
        completed = checkpoint_epoch(last)
        if not best.is_file() or not last.is_file() or completed < EPOCHS:
            raise RuntimeError(f"incomplete artifacts: best={best.is_file()} last={last.is_file()} epochs={completed}/{EPOCHS}")
        report.update({"status": "done", "finished_at": now(), "run_dir": str(target), "best": str(best), "last": str(last), "completed_epochs": completed})
        atomic_json(report_path, report)
        log(f"TRAIN_DONE dataset={dataset} epochs={completed} best={best}")
        return 0
    except Exception as exc:
        report.update({"status": "failed", "finished_at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        atomic_json(report_path, report)
        log(f"TRAIN_FAILED dataset={dataset} error={exc!r}")
        return 1

def parse_eval_table(text: str, names: list[str]) -> dict:
    wanted = set(names) | {"all"}
    rows = {}
    for line in text.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        parts = clean.split()
        if len(parts) >= 8 and parts[0] in wanted:
            try:
                rows[parts[0]] = {"images": int(float(parts[-7])), "labels": int(float(parts[-6])), "precision": float(parts[-5]), "recall": float(parts[-4]), "f1": float(parts[-3]), "map50": float(parts[-2]), "map50_95": float(parts[-1])}
            except Exception:
                continue
    missing = [name for name in ["all", *names] if name not in rows]
    if missing:
        raise RuntimeError(f"could not parse PR metric rows for {missing}; eval output tail:\n" + "\n".join(text.splitlines()[-80:]))
    return {"precision": rows["all"]["precision"], "recall": rows["all"]["recall"], "map50": rows["all"]["map50"], "map50_95": rows["all"]["map50_95"], "class_ap50": {n: rows[n]["map50"] for n in names}, "class_ap50_95": {n: rows[n]["map50_95"] for n in names}, "raw_rows": rows}

def benchmark_checkpoint(weights: Path, warmup: int = 20, iterations: int = 100) -> tuple[dict, int]:
    old_cwd = os.getcwd(); os.chdir(GOLD_ROOT)
    try:
        ensure_gold_import_path()
        from yolov6.utils.checkpoint import load_checkpoint
        model = load_checkpoint(str(weights), map_location="cuda:0", fuse=True).float().to("cuda:0").eval()
        params = int(sum(p.numel() for p in model.parameters()))
        inputs = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda:0")
        torch.cuda.empty_cache()
        with torch.inference_mode():
            for _ in range(warmup): model(inputs)
            torch.cuda.synchronize(); start = time.perf_counter()
            for _ in range(iterations): model(inputs)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        os.chdir(old_cwd)
    fps = iterations / elapsed
    return {"fps": float(fps), "infer_ms_per_image": float(1000.0 / fps), "elapsed_seconds": float(elapsed), "images": iterations, "batch": 1, "imgsz": IMGSZ, "warmup_iterations": warmup, "measured_iterations": iterations, "scope": "fused model forward only; excludes preprocess and postprocess", "device": torch.cuda.get_device_name(0)}, params

def eval_worker(dataset: str) -> int:
    weights = run_dir(dataset) / "weights" / "best_ckpt.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    data_yaml = DATA_ROOT / f"{dataset}.yaml"
    names = EXPECTED[dataset]["names"]
    save_dir = PROJECT / "independent_val_runs"
    cmd = [PYTHON, "tools/eval.py", "--data", str(data_yaml), "--weights", str(weights), "--batch-size", "1", "--img-size", str(IMGSZ), "--task", "test", "--device", "0", "--do_coco_metric", "False", "--do_pr_metric", "True", "--plot_curve", "False", "--verbose", "--save_dir", str(save_dir), "--name", f"{dataset}_gold_yolo_n_seed{SEED}"]
    log(f"INDEPENDENT_EVAL_START dataset={dataset} weights={weights}")
    proc = subprocess.run(cmd, cwd=str(GOLD_ROOT), env=gold_subprocess_env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    raw_log = PROJECT / f"eval_{dataset}_gold_yolo_n_seed{SEED}.log"
    raw_log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"eval command failed rc={proc.returncode}; see {raw_log}")
    metrics = parse_eval_table(proc.stdout, names)
    benchmark, params = benchmark_checkpoint(weights)
    report = {"status": "done", "created_at": now(), "dataset": dataset, "model": MODEL_NAME, "seed": SEED, "weights": str(weights), "data": str(data_yaml), "eval_log": str(raw_log), "protocol": {"truth": "fresh Python process reloads best_ckpt.pt from disk and evaluates test split", "imgsz": IMGSZ, "batch": 1, "device": 0, "do_pr_metric": True, "do_coco_metric": False}, "structure": {"fused_params": params}, "test": metrics, "benchmark": benchmark}
    atomic_json(independent_report(dataset), report)
    log(f"INDEPENDENT_EVAL_DONE dataset={dataset} map50={metrics['map50']:.6f} map50_95={metrics['map50_95']:.6f} fps={benchmark['fps']:.2f}")
    return 0

def experiment_done(dataset: str) -> bool:
    report = read_json(independent_report(dataset), {})
    return report.get("status") == "done" and Path(report.get("weights", "")).is_file()

def summarize() -> dict:
    experiments, rows = {}, []
    pending = "\u5f85\u5b8c\u6210"
    for dataset in SOURCE_DATASETS:
        report = read_json(independent_report(dataset), {})
        if report.get("status") != "done":
            continue
        experiments[dataset] = report
        row = {"dataset": dataset, "model": MODEL_NAME, "seed": SEED, "test_map50": report["test"]["map50"], "test_map50_95": report["test"]["map50_95"], "precision": report["test"]["precision"], "recall": report["test"]["recall"], "fps": report["benchmark"]["fps"], "infer_ms_per_image": report["benchmark"]["infer_ms_per_image"], "fused_params": report["structure"]["fused_params"], "weights": report["weights"]}
        for name, value in report["test"]["class_ap50"].items(): row[f"class_ap50::{name}"] = value
        for name, value in report["test"]["class_ap50_95"].items(): row[f"class_ap50_95::{name}"] = value
        rows.append(row)
    aggregate = {"status": "done" if len(experiments) == len(SOURCE_DATASETS) else "partial", "created_at": now(), "model": MODEL_NAME, "seed": SEED, "protocol": {"training": f"official Gold-YOLO base recipe, --fuse_ab, no self-distillation, 250 epochs / imgsz640 / batch{BATCH} / seed{SEED} / cache=False", "truth": "independent fresh Python process reloads best_ckpt.pt and evaluates test split", "fps": "RTX 4090 / fused FP32 / imgsz640 / batch1 / warmup20 / iterations100 / forward only"}, "experiments": experiments}
    atomic_json(REPORT, aggregate)
    class_columns = sorted({key for row in rows for key in row if key.startswith("class_")})
    fields = ["dataset", "model", "seed", "test_map50", "test_map50_95", "precision", "recall", "fps", "infer_ms_per_image", "fused_params", *class_columns, "weights"]
    with COMPARISON_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    u = lambda s: s.encode("ascii").decode("unicode_escape")
    lines = [
        f"# {MODEL_NAME} Seed{SEED} " + u(r"\u4e09\u6570\u636e\u96c6\u521d\u7b5b"),
        "",
        "- " + u(r"\u8bad\u7ec3\uff1a\u5b98\u65b9 Gold-YOLO base recipe\uff0c") + "`--fuse_ab`" + u(r"\uff0c\u4e0d\u505a self-distillation\uff1b") + f"250 epochs, imgsz=640, batch={BATCH}, seed={SEED}" + u(r"\u3002"),
        "- " + u(r"\u771f\u503c\uff1a\u72ec\u7acb Python \u8fdb\u7a0b\u91cd\u8f7d ") + "`best_ckpt.pt`" + u(r"\uff0c\u5728 test split \u8bc4\u6d4b\u3002"),
        "- FPS" + u(r"\uff1aRTX 4090\uff0cfused FP32\uff0cbatch=1\uff0cwarmup=20\uff0citerations=100\uff0c\u4ec5 forward\u3002"),
        "- " + f"说明：本报告为 seed{SEED} 单 seed 结果，多 seed 稳定性以汇总器完成后为准。",
        "",
        "## " + u(r"\u6574\u4f53\u7ed3\u679c"),
        "",
        "| " + u(r"\u6570\u636e\u96c6") + " | test mAP50 | test mAP50-95 | Precision | Recall | FPS | fused params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in SOURCE_DATASETS:
        r = experiments.get(dataset)
        lines.append(f"| {dataset} | {r['test']['map50']:.6f} | {r['test']['map50_95']:.6f} | {r['test']['precision']:.6f} | {r['test']['recall']:.6f} | {r['benchmark']['fps']:.2f} | {r['structure']['fused_params']:,} |" if r else f"| {dataset} | {pending} | {pending} | {pending} | {pending} | {pending} | {pending} |")
    for dataset, r in experiments.items():
        lines += ["", f"## {dataset} " + u(r"\u9010\u7c7b test AP"), "", "| " + u(r"\u7c7b\u522b") + " | AP50 | AP50-95 |", "|---|---:|---:|"]
        for name, value in r["test"]["class_ap50"].items(): lines.append(f"| {name} | {value:.6f} | {r['test']['class_ap50_95'][name]:.6f} |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"SUMMARY_WRITTEN status={aggregate['status']} completed={len(experiments)}/{len(SOURCE_DATASETS)} report={REPORT_MD}")
    return aggregate

def orchestrate(only: str | None = None) -> int:
    build_check()
    selected = [only] if only else list(SOURCE_DATASETS)
    atomic_json(STATUS, {"status": "running", "phase": "train", "started_at": now(), "datasets": selected, "current": None, "model": MODEL_NAME})
    try:
        for dataset in selected:
            if experiment_done(dataset):
                log(f"SKIP_COMPLETE dataset={dataset}"); continue
            if checkpoint_epoch(run_dir(dataset) / "weights" / "last_ckpt.pt") < EPOCHS:
                atomic_json(STATUS, {"status": "running", "phase": "train", "updated_at": now(), "datasets": selected, "current": dataset, "model": MODEL_NAME})
                if subprocess.call([PYTHON, str(Path(__file__).resolve()), "--train-worker", dataset]) != 0:
                    raise RuntimeError(f"training worker failed: dataset={dataset}")
            atomic_json(STATUS, {"status": "running", "phase": "eval", "updated_at": now(), "datasets": selected, "current": dataset, "model": MODEL_NAME})
            if subprocess.call([PYTHON, str(Path(__file__).resolve()), "--eval-worker", dataset]) != 0:
                raise RuntimeError(f"evaluation worker failed: dataset={dataset}")
            summarize()
        result = summarize(); completed = [d for d in SOURCE_DATASETS if experiment_done(d)]
        atomic_json(STATUS, {"status": "done" if len(completed) == len(SOURCE_DATASETS) else "partial", "phase": "done", "finished_at": now(), "completed": completed, "pending": [d for d in SOURCE_DATASETS if d not in completed], "model": MODEL_NAME})
        return 0 if result["status"] == "done" or only else 1
    except Exception as exc:
        atomic_json(STATUS, {"status": "failed", "phase": "failed", "updated_at": now(), "error": repr(exc), "traceback": traceback.format_exc(), "model": MODEL_NAME})
        raise

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--build-check", action="store_true")
    parser.add_argument("--train-worker", choices=tuple(SOURCE_DATASETS))
    parser.add_argument("--eval-worker", choices=tuple(SOURCE_DATASETS))
    parser.add_argument("--only", choices=tuple(SOURCE_DATASETS))
    args = parser.parse_args()
    if args.prepare_data:
        atomic_json(BUILD_REPORT, {"status": "data_prepared", "created_at": now(), "datasets": prepare_data()}); return 0
    if args.build_check: return build_check()
    if args.train_worker: return train_worker(args.train_worker)
    if args.eval_worker: return eval_worker(args.eval_worker)
    return orchestrate(args.only)

if __name__ == "__main__":
    raise SystemExit(main())


