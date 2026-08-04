#!/usr/bin/env python
"""Iteration 045: MSDGS135eq with a single C3k2EMSC at backbone layer 8 (P5 stage)."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_emsc_msdgs_e250"
GEN_DIR = ROOT / "generated_models_emsc_msdgs_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"
EVAL_SCRIPT = ROOT / "eval_msdgs_candidate.py"
NAME = "msdgs_p5stage_emsc"

BASELINES = {
    "gsdown": {"test_map50": 0.7338, "test_map50_95": 0.401794, "fused_params": 1935814},
    "msdgs135eq": {"test_map50": 0.732411, "test_map50_95": 0.398837, "fused_params": 1777318},
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_status(status: str, **extra) -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(
        {"status": status, "current": NAME, "updated_at": now(), **extra},
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")


def archive_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    archived = path.with_name(f"{path.name}.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.move(str(path), str(archived))
    print(f"ARCHIVED {path} -> {archived}")
    return archived


def build_emsc() -> dict:
    doc = yaml.safe_load(MSDGS_YAML.read_text(encoding="utf-8"))
    backbone = doc["backbone"]
    if len(backbone) <= 8 or backbone[8][2] != "C3k2":
        raise AssertionError(f"backbone[8] expected C3k2, got {backbone[8] if len(backbone)>8 else None}")
    original = list(backbone[8])
    backbone[8][2] = "C3k2EMSC"
    changed = [i for i, block in enumerate(backbone) if block[2] == "C3k2EMSC"]
    if changed != [8]:
        raise AssertionError(f"expected only backbone[8] C3k2EMSC, got {changed}")
    print(f"SWAP backbone[8] {original} -> {backbone[8]}")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for required in (DATA, MSDGS_YAML, OFFICIAL_WEIGHTS, EVAL_SCRIPT):
        if not required.exists():
            raise FileNotFoundError(required)

    run_dir = PROJECT / NAME
    if run_dir.exists():
        if not args.force:
            raise RuntimeError(f"fresh-train protection: {run_dir} exists; pass --force to archive it")
        archive_existing(run_dir)

    PROJECT.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GEN_DIR / f"{NAME}.yaml"
    cfg.write_text(
        "# Iteration 045: MSDGS135eq + backbone[8] P5-stage C3k2EMSC\n"
        + yaml.safe_dump(build_emsc(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    save_status("training", pid=os.getpid(), cfg=str(cfg), data=str(DATA), seed=args.seed)

    print(f"CFG {cfg}")
    print(f"DATA {DATA}")
    print(f"INIT_WEIGHTS {OFFICIAL_WEIGHTS}")
    print(f"TRAIN_START {NAME} epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} seed={args.seed}")
    try:
        model = YOLO(str(cfg), task="detect").load(str(OFFICIAL_WEIGHTS))
        params_unfused = int(sum(p.numel() for p in model.model.parameters()))
        model.train(
            data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=8,
            seed=args.seed, device=0, project=str(PROJECT), name=NAME, exist_ok=False,
            patience=max(args.epochs, 250), cache=False, verbose=True,
        )
        weights = run_dir / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        print(f"TRAIN_DONE {NAME} weights={weights}")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        eval_json = run_dir / "independent_eval.json"
        save_status("evaluating", pid=os.getpid(), weights=str(weights), evaluator="new subprocess")
        cmd = [
            sys.executable, str(EVAL_SCRIPT), "--weights", str(weights), "--data", str(DATA),
            "--project", str(PROJECT), "--name", NAME, "--output", str(eval_json),
            "--imgsz", str(args.imgsz), "--batch", str(args.batch),
        ]
        print("EVAL_SUBPROCESS", " ".join(cmd))
        subprocess.run(cmd, check=True)
        evaluation = json.loads(eval_json.read_text(encoding="utf-8"))
        result = {
            "status": "done", "candidate": NAME, "iteration": 45,
            "cfg": str(cfg), "data": str(DATA), "weights": str(weights),
            "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz, "seed": args.seed,
            "cache": False, "init_weights": str(OFFICIAL_WEIGHTS),
            "params_unfused_before_train": params_unfused,
            **evaluation, "finished_at": now(),
        }
        REPORT_JSON.write_text(json.dumps(
            {"schema_version": 2, "baseline": BASELINES, "experiments": {NAME: result}},
            indent=2, ensure_ascii=False,
        ), encoding="utf-8")
        save_status("done", pid=os.getpid(), report_json=str(REPORT_JSON), weights=str(weights))
        print("EMSC_RESULT", f"test_mAP50={result['test']['map50']:.6f}",
              f"test_mAP50-95={result['test']['map50_95']:.6f}")
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        save_status("failed", pid=os.getpid(), error=repr(exc))
        REPORT_JSON.write_text(json.dumps({
            "schema_version": 2, "baseline": BASELINES,
            "experiments": {NAME: {"status": "failed", "error": repr(exc), "failed_at": now()}},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
