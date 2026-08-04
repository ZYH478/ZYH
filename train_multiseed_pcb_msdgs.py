#!/usr/bin/env python
"""PCB 上 MSDGS neck-only seed1/2/3 训练与 seed0/1/2/3 稳健性汇总。"""
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
PROJECT = ROOT / "runs_pcb_msdgs_multiseed_e250"
SOURCE_PROJECT = ROOT / "runs_generalization_aluminum_pcb_yolo26_msdgs_e250"
DATA = SOURCE_PROJECT / "datasets" / "pcb.yaml"
OFFICIAL_WEIGHTS = ROOT / "yolo26n.pt"
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
EVAL_SCRIPT = ROOT / "eval_generalization_aluminum_pcb.py"
MODEL_NAME = "msdgs"
SEEDS = (1, 2, 3)
ALL_SEEDS = (0, 1, 2, 3)
DEFAULT_EPOCHS = 250
DEFAULT_BATCH = 32
DEFAULT_IMGSZ = 640
EXPECTED_SPLIT_COUNTS = {"train": 891, "val": 120, "test": 60}
EXPECTED_CLASSES = (
    "missing_hole", "mouse_bite", "open_circuit", "short", "spurious_copper", "spur"
)


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


def training_report(seed: int) -> Path:
    return PROJECT / f"training_pcb_msdgs_seed{seed}_report.json"


def independent_report(seed: int) -> Path:
    if seed == 0:
        return SOURCE_PROJECT / "independent_pcb_msdgs_report.json"
    return PROJECT / f"independent_pcb_msdgs_seed{seed}_report.json"


def split_counts(checked: dict) -> dict[str, int]:
    counts = {}
    for split in ("train", "val", "test"):
        path = Path(str(checked[split]))
        counts[split] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return counts


def build_check() -> int:
    from ultralytics.data.utils import check_det_dataset

    required = [DATA, OFFICIAL_WEIGHTS, MSDGS_YAML, EVAL_SCRIPT]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checked = check_det_dataset(str(DATA), autodownload=False)
    if int(checked["nc"]) != len(EXPECTED_CLASSES):
        raise RuntimeError(f"expected {len(EXPECTED_CLASSES)} classes, got {checked['nc']}")
    raw_names = checked["names"]
    names = tuple(str(raw_names[i]) for i in sorted(raw_names)) if isinstance(raw_names, dict) else tuple(map(str, raw_names))
    if names != EXPECTED_CLASSES:
        raise RuntimeError(f"class names changed: {names}")
    counts = split_counts(checked)
    if counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"split counts changed: expected={EXPECTED_SPLIT_COUNTS}, actual={counts}")

    seed0 = read_json(independent_report(0), {})
    if seed0.get("status") != "done":
        raise RuntimeError("missing completed PCB MSDGS seed0 independent report")

    yolo = YOLO(str(MSDGS_YAML))
    yolo.load(str(OFFICIAL_WEIGHTS))
    net = yolo.model.to("cuda:0").eval()
    with torch.inference_mode():
        out = net(torch.randn(1, 3, DEFAULT_IMGSZ, DEFAULT_IMGSZ, device="cuda:0"))
    det = net.model[-1]
    custom = sum(1 for m in net.modules() if m.__class__.__name__ == "MSDGS")
    fused = net.fuse() if hasattr(net, "fuse") else net
    params = int(sum(p.numel() for p in fused.parameters()))
    structure = {
        "cfg": str(MSDGS_YAML),
        "pretrained": str(OFFICIAL_WEIGHTS),
        "end2end": bool(getattr(fused, "end2end", False)),
        "reg_max": int(getattr(det, "reg_max", -1)),
        "nl": int(getattr(det, "nl", -1)),
        "msdgs_modules": custom,
        "fused_params": params,
        "forward_type": type(out).__name__,
    }
    if not structure["end2end"] or structure["reg_max"] != 1 or structure["nl"] != 3:
        raise RuntimeError(f"bad YOLO26 structure: {structure}")
    if custom != 4:
        raise RuntimeError(f"expected 4 MSDGS modules, got {custom}")
    if params != int(seed0["structure"]["fused_params"]):
        raise RuntimeError(f"fused params drifted: build={params}, seed0={seed0['structure']['fused_params']}")

    report = {
        "status": "pass",
        "created_at": now(),
        "dataset": "pcb",
        "data": str(DATA),
        "nc": int(checked["nc"]),
        "names": list(names),
        "split_counts": counts,
        "seeds_to_train": list(SEEDS),
        "seed0_anchor": str(independent_report(0)),
        "structure": structure,
        "recipe": {
            "epochs": DEFAULT_EPOCHS,
            "imgsz": DEFAULT_IMGSZ,
            "batch": DEFAULT_BATCH,
            "deterministic": True,
            "cache": False,
            "resume": False,
            "patience": 0,
        },
        "cuda": torch.cuda.get_device_name(0),
    }
    atomic_json(PROJECT / "build_check_report.json", report)
    print(
        f"PCB_MSDGS_MULTISEED_BUILD_GATE_PASS params={params} msdgs={custom} "
        f"split_counts={counts}", flush=True
    )
    return 0


def worker(seed: int, epochs: int, batch: int, imgsz: int) -> int:
    out = training_report(seed)
    report = {
        "status": "running",
        "started_at": now(),
        "dataset": "pcb",
        "model": MODEL_NAME,
        "seed": seed,
        "pid": os.getpid(),
        "cfg": str(MSDGS_YAML),
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
    print(f"TRAIN_START pcb/msdgs/seed{seed}", flush=True)
    try:
        yolo = YOLO(str(MSDGS_YAML))
        yolo.train(
            data=str(DATA), epochs=epochs, batch=batch, imgsz=imgsz, device=0,
            seed=seed, deterministic=True, pretrained=str(OFFICIAL_WEIGHTS),
            cache=False, resume=False, patience=0,
            project=str(PROJECT / f"seed{seed}"), name=MODEL_NAME, exist_ok=False,
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
        print(f"TRAIN_DONE pcb/msdgs/seed{seed} best={best}", flush=True)
        return 0
    except Exception as exc:
        report.update({
            "status": "failed", "finished_at": now(), "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        atomic_json(out, report)
        print(f"TRAIN_FAILED pcb/msdgs/seed{seed} error={exc!r}", flush=True)
        return 1


def experiment_done(seed: int) -> bool:
    rep = read_json(independent_report(seed), {})
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
    experiments: dict[str, dict] = {}
    rows = []
    for seed in ALL_SEEDS:
        rep = read_json(independent_report(seed), {})
        if rep.get("status") != "done":
            continue
        if rep.get("seed") != seed:
            rep["seed"] = seed
            if seed != 0:
                atomic_json(independent_report(seed), rep)
        experiments[str(seed)] = rep
        row = {
            "model": MODEL_NAME,
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
    if len(experiments) == len(ALL_SEEDS):
        stats = {
            "test_map50": metric_stats([experiments[str(s)]["test"]["map50"] for s in ALL_SEEDS]),
            "test_map50_95": metric_stats([experiments[str(s)]["test"]["map50_95"] for s in ALL_SEEDS]),
            "precision": metric_stats([experiments[str(s)]["test"]["precision"] for s in ALL_SEEDS]),
            "recall": metric_stats([experiments[str(s)]["test"]["recall"] for s in ALL_SEEDS]),
            "fps": metric_stats([experiments[str(s)]["benchmark"]["fps"] for s in ALL_SEEDS]),
            "fused_params": experiments["0"]["structure"]["fused_params"],
            "class_ap50": {},
            "class_ap50_95": {},
        }
        for cls in EXPECTED_CLASSES:
            stats["class_ap50"][cls] = metric_stats(
                [experiments[str(s)]["test"]["class_ap50"][cls] for s in ALL_SEEDS]
            )
            stats["class_ap50_95"][cls] = metric_stats(
                [experiments[str(s)]["test"]["class_ap50_95"][cls] for s in ALL_SEEDS]
            )

    seed0 = experiments.get("0")
    relative_to_seed0 = {}
    if seed0:
        for seed in SEEDS:
            rep = experiments.get(str(seed))
            if not rep:
                continue
            relative_to_seed0[str(seed)] = {
                "map50_delta": rep["test"]["map50"] - seed0["test"]["map50"],
                "map50_95_delta": rep["test"]["map50_95"] - seed0["test"]["map50_95"],
                "class_ap50_delta": {
                    cls: rep["test"]["class_ap50"][cls] - seed0["test"]["class_ap50"][cls]
                    for cls in EXPECTED_CLASSES
                },
                "class_ap50_95_delta": {
                    cls: rep["test"]["class_ap50_95"][cls] - seed0["test"]["class_ap50_95"][cls]
                    for cls in EXPECTED_CLASSES
                },
            }

    report = {
        "status": "done" if len(rows) == len(ALL_SEEDS) else "partial",
        "created_at": now(),
        "dataset": "pcb",
        "model": MODEL_NAME,
        "protocol": "250e/imgsz640/batch32/deterministic/cacheFalse/resumeFalse/patience0/yolo26n.pt; independent fresh-process test evaluation",
        "seed0_source": str(independent_report(0)),
        "new_seeds": list(SEEDS),
        "experiments": experiments,
        "statistics": stats,
        "relative_to_seed0": relative_to_seed0,
        "completed_total": len(rows),
        "expected_total": len(ALL_SEEDS),
    }
    atomic_json(PROJECT / "report.json", report)

    class_cols = sorted({k for row in rows for k in row if k.startswith("class_")})
    fields = [
        "model", "seed", "test_map50", "test_map50_95", "precision", "recall", "fps",
        "fused_params", *class_cols, "weights",
    ]
    PROJECT.mkdir(parents=True, exist_ok=True)
    with (PROJECT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["seed"]))

    lines = [
        "# PCB MSDGS neck-only 四 Seed 稳健性验证",
        "",
        "> seed0 复用既有独立 test 真值；seed1/2/3 使用完全相同训练协议补跑。所有最终指标均由独立新 Python 进程从磁盘重载 best.pt 后在 test split 评测。",
        "",
        "## 各 Seed 总体结果",
        "",
        "| Seed | test mAP50 | test mAP50-95 | Precision | Recall | FPS | Fused Params |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in ALL_SEEDS:
        rep = experiments.get(str(seed))
        if rep:
            lines.append(
                f"| {seed} | {rep['test']['map50']:.6f} | {rep['test']['map50_95']:.6f} | "
                f"{rep['test']['precision']:.6f} | {rep['test']['recall']:.6f} | "
                f"{rep['benchmark']['fps']:.1f} | {rep['structure']['fused_params']:,} |"
            )
        else:
            lines.append(f"| {seed} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")

    lines += [
        "", "## 四 Seed 总体稳定性", "",
        "| 指标 | mean ± std | min | max | range |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("test_map50", "test mAP50"), ("test_map50_95", "test mAP50-95")):
        st = stats.get(key)
        if st:
            lines.append(
                f"| {label} | {st['mean']:.6f} ± {st['std_population']:.6f} | "
                f"{st['min']:.6f} | {st['max']:.6f} | {st['range']:.6f} |"
            )
        else:
            lines.append(f"| {label} | 待完成 | 待完成 | 待完成 | 待完成 |")

    lines += [
        "", "## 各缺陷四 Seed 稳定性", "",
        "| 类别 | AP50 mean ± std | min | range | AP50-95 mean ± std | min | range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if stats:
        for cls in EXPECTED_CLASSES:
            a = stats["class_ap50"][cls]
            b = stats["class_ap50_95"][cls]
            lines.append(
                f"| {cls} | {a['mean']:.6f} ± {a['std_population']:.6f} | {a['min']:.6f} | "
                f"{a['range']:.6f} | {b['mean']:.6f} ± {b['std_population']:.6f} | "
                f"{b['min']:.6f} | {b['range']:.6f} |"
            )
    else:
        lines.append("| 待训练完成 | - | - | - | - | - | - |")

    lines += [
        "", "## 稳健性重点", "",
        "- 重点观察 seed0 中唯一明显退化的 `short` 是否跨 Seed 持续退化。",
        "- 重点观察较弱类别 `spur` 的均值、最差值与极差。",
        "- 检查 `missing_hole` 与 `spurious_copper` 的 seed0 大幅提升是否可复现。",
    ]
    (PROJECT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SUMMARY_WRITTEN status={report['status']} completed={len(rows)}/{len(ALL_SEEDS)}", flush=True)
    return report


def status_payload(phase: str, current: str | None, completed: list[str], failures: list[dict]) -> dict:
    pending = [f"pcb/msdgs/seed{s}" for s in SEEDS if f"pcb/msdgs/seed{s}" not in completed]
    return {
        "status": "running",
        "phase": phase,
        "updated_at": now(),
        "pid": os.getpid(),
        "current": current,
        "completed": completed,
        "failures": failures,
        "pending": pending,
    }


def run_orchestrator(epochs: int, batch: int, imgsz: int) -> int:
    completed = [f"pcb/msdgs/seed{s}" for s in SEEDS if experiment_done(s)]
    failures: list[dict] = []
    status_path = PROJECT / "status.json"
    initial = status_payload("orchestrating", None, completed, failures)
    initial["started_at"] = now()
    atomic_json(status_path, initial)

    for seed in SEEDS:
        tag = f"pcb/msdgs/seed{seed}"
        if experiment_done(seed):
            print(f"EXPERIMENT_SKIP {tag}", flush=True)
            continue
        tr = read_json(training_report(seed), {})
        if tr.get("status") == "done" and tr.get("completed_epochs") == epochs and Path(tr.get("best", "")).is_file():
            best = Path(tr["best"])
            print(f"TRAIN_REUSE {tag} best={best}", flush=True)
        else:
            atomic_json(status_path, status_payload("training", tag, completed, failures))
            cmd = [
                sys.executable, str(Path(__file__).resolve()), "--worker", "--seed", str(seed),
                "--epochs", str(epochs), "--batch", str(batch), "--imgsz", str(imgsz),
            ]
            rc = subprocess.run(cmd).returncode
            tr = read_json(training_report(seed), {})
            if rc != 0 or tr.get("status") != "done":
                failures.append({
                    "experiment": tag, "stage": "training", "rc": rc, "error": tr.get("error")
                })
                summarize()
                continue
            best = Path(tr["best"])

        atomic_json(status_path, status_payload("independent_eval", tag, completed, failures))
        out = independent_report(seed)
        cmd = [
            sys.executable, str(EVAL_SCRIPT), "--dataset", "pcb", "--model", MODEL_NAME,
            "--weights", str(best), "--data", str(DATA), "--output", str(out),
            "--imgsz", str(imgsz), "--batch", str(batch),
        ]
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
    pending = [f"pcb/msdgs/seed{s}" for s in SEEDS if not experiment_done(s)]
    status = "done" if not failures and final["status"] == "done" else "done_with_failures"
    atomic_json(status_path, {
        "status": status,
        "phase": "done",
        "finished_at": now(),
        "pid": os.getpid(),
        "current": None,
        "completed": completed,
        "failures": failures,
        "pending": pending,
    })
    print(
        f"PCB_MSDGS_MULTISEED_DONE status={status} completed={len(completed)}/{len(SEEDS)} "
        f"failures={len(failures)} pending={len(pending)}", flush=True
    )
    return 0 if status == "done" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-check", action="store_true")
    ap.add_argument("--worker", action="store_true")
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
        if args.seed not in SEEDS:
            raise ValueError("worker requires --seed in 1,2,3")
        return worker(args.seed, args.epochs, args.batch, args.imgsz)
    return run_orchestrator(args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    raise SystemExit(main())
