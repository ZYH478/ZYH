#!/usr/bin/env python
"""Fresh-train MSDGS Backbone P3 C3k seed0 and independently evaluate best.pt."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from generate_msdgs_backbone_p3_c3k import CANDIDATE, generate_candidate_yaml

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_msdgs_backbone_p3_c3k_e250"
RUN_NAME = CANDIDATE
STATUS = PROJECT / "status.json"
REPORT_JSON = PROJECT / "report.json"
REPORT_MD = PROJECT / "report.md"
INDEPENDENT_REPORT = PROJECT / "independent_seed0_report.json"
BUILD_REPORT = PROJECT / "build_check" / "report.json"
LOCK = PROJECT / ".train.lock"
SCRIPT = Path(__file__).resolve()
EVALUATOR = ROOT / "eval_msdgs_candidate.py"

EPOCHS = 250
IMGSZ = 640
BATCH = 32
WORKERS = 8
SEED = 0

MSDGS = {
    "test_map50": 0.7324113100352823,
    "test_map50_95": 0.3988372172746915,
    "test_crazing_map50_95": 0.1770695825201802,
}
IDEAL = {
    "test_map50": 0.738,
    "test_map50_95": 0.402,
    "test_crazing_map50_95": 0.187,
}
CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_status(status: str, note: str = "", **extra) -> None:
    atomic_json(
        STATUS,
        {
            "status": status,
            "candidate": RUN_NAME,
            "note": note,
            "updated_at": now(),
            "pid": os.getpid(),
            "project": str(PROJECT),
            "report_json": str(REPORT_JSON),
            "resume": False,
            **extra,
        },
    )


def archive_previous_artifacts() -> Path | None:
    artifacts = [PROJECT / RUN_NAME, STATUS, REPORT_JSON, REPORT_MD, INDEPENDENT_REPORT]
    existing = [path for path in artifacts if path.exists()]
    if not existing:
        return None
    archive = PROJECT / "archive" / time.strftime("%Y%m%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        destination = archive / path.name
        shutil.move(str(path), str(destination))
        print("ARCHIVED_PREVIOUS", path, "->", destination, flush=True)
    return archive


def gpu_idle() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    if result.stdout.strip():
        raise RuntimeError(f"GPU already busy; refusing duplicate training:\n{result.stdout.strip()}")


def run_checked(command: list[str]) -> None:
    print("SUBPROCESS", " ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"subprocess failed rc={result.returncode}: {' '.join(command)}")


def assert_model_contract(model) -> dict:
    layer2 = model.model[2]
    layer4 = model.model[4]
    layer6 = model.model[6]
    layer8 = model.model[8]
    head = model.model[-1]
    assert type(layer2).__name__ == "C3k2" and type(layer2.m[0]).__name__ == "Bottleneck"
    assert type(layer4).__name__ == "C3k2" and type(layer4.m[0]).__name__ == "C3k"
    assert type(layer6).__name__ == "C3k2" and type(layer6.m[0]).__name__ == "C3k"
    assert type(layer8).__name__ == "C3k2" and type(layer8.m[0]).__name__ == "C3k"
    assert head.end2end is True and head.reg_max == 1 and head.nl == 3
    channels = tuple(int(tower[0].conv.in_channels) for tower in head.cv2)
    assert channels == (64, 128, 256), channels
    return {
        "layer2_inner": type(layer2.m[0]).__name__,
        "layer4_inner": type(layer4.m[0]).__name__,
        "layer6_inner": type(layer6.m[0]).__name__,
        "layer8_inner": type(layer8.m[0]).__name__,
        "end2end": bool(head.end2end),
        "reg_max": int(head.reg_max),
        "nl": int(head.nl),
        "detect_input_channels": list(channels),
    }


def train_worker(cfg: str) -> int:
    from ultralytics import YOLO

    target = PROJECT / RUN_NAME
    if target.exists():
        raise FileExistsError(f"fresh worker refuses existing run directory: {target}")
    print("TRAIN_WORKER_START", os.getpid(), cfg, flush=True)
    wrapper = YOLO(cfg, task="detect").load(str(OFFICIAL_WEIGHTS))
    structure = assert_model_contract(wrapper.model)
    params_unfused = int(sum(parameter.numel() for parameter in wrapper.model.parameters()))
    wrapper.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        seed=SEED,
        device=0,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        patience=EPOCHS,
        cache=False,
        resume=False,
        verbose=True,
    )
    weights = target / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)
    atomic_json(
        target / "train_worker.json",
        {
            "status": "done",
            "candidate": RUN_NAME,
            "seed": SEED,
            "cfg": cfg,
            "weights": str(weights),
            "params_unfused": params_unfused,
            "structure": structure,
            "finished_at": now(),
        },
    )
    print("TRAIN_WORKER_DONE", weights, flush=True)
    return 0


def gate(report: dict) -> dict:
    test = report["test"]
    crazing = test["per_class"]["crazing"]["map50_95"]
    deltas = {
        "map50": test["map50"] - MSDGS["test_map50"],
        "map50_95": test["map50_95"] - MSDGS["test_map50_95"],
        "crazing_map50_95": crazing - MSDGS["test_crazing_map50_95"],
    }
    owner_checks = {
        "test_map50_gt_msdgs": test["map50"] > MSDGS["test_map50"],
        "test_map50_95_ge_msdgs": test["map50_95"] >= MSDGS["test_map50_95"],
        "test_crazing_map50_95_gt_msdgs": crazing > MSDGS["test_crazing_map50_95"],
    }
    ideal_checks = {
        "test_map50_ge_0.738": test["map50"] >= IDEAL["test_map50"],
        "test_map50_95_ge_0.402": test["map50_95"] >= IDEAL["test_map50_95"],
        "test_crazing_map50_95_ge_0.187": crazing >= IDEAL["test_crazing_map50_95"],
    }
    return {
        "owner_gate_pass": all(owner_checks.values()),
        "owner_checks": owner_checks,
        "ideal_gate_pass": all(ideal_checks.values()),
        "ideal_checks": ideal_checks,
        "deltas_vs_msdgs": deltas,
    }


def metric_table(split: dict) -> list[str]:
    return [
        f"Precision = {split['precision']:.6f}",
        f"Recall = {split['recall']:.6f}",
        f"mAP50 = {split['map50']:.6f}",
        f"mAP50-95 = {split['map50_95']:.6f}",
    ]


def write_report(payload: dict) -> None:
    atomic_json(REPORT_JSON, payload)
    experiment = payload.get("experiment", {})
    lines = [
        "# MSDGS Backbone P3 官方 C3k 主路径 seed0 报告",
        "",
        "固定协议：250 epochs / imgsz=640 / batch=32 / seed0 / cache=False / resume=False / yolo26n.pt。",
        "最终真值由独立新进程重新加载磁盘 `best.pt`，分别评测 val/test。",
        "",
    ]
    if experiment.get("status") == "done":
        val = experiment["val"]
        test = experiment["test"]
        result_gate = experiment["gate"]
        crazing = test["per_class"]["crazing"]["map50_95"]
        lines += [
            "## 总体结果",
            "",
            "| split | Precision | Recall | mAP50 | mAP50-95 |",
            "|---|---:|---:|---:|---:|",
            f"| val | {val['precision']:.6f} | {val['recall']:.6f} | {val['map50']:.6f} | {val['map50_95']:.6f} |",
            f"| test | {test['precision']:.6f} | {test['recall']:.6f} | {test['map50']:.6f} | {test['map50_95']:.6f} |",
            "",
            "## Test 六类 AP",
            "",
            "| 类别 | AP50 | AP50-95 |",
            "|---|---:|---:|",
        ]
        for name in CLASS_NAMES:
            item = test["per_class"][name]
            lines.append(f"| {name} | {item['map50']:.6f} | {item['map50_95']:.6f} |")
        lines += [
            "",
            "## Gate",
            "",
            "| test mAP50 | test mAP50-95 | crazing mAP50-95 | Owner gate | 理想 gate |",
            "|---:|---:|---:|---|---|",
            f"| {test['map50']:.6f} | {test['map50_95']:.6f} | {crazing:.6f} | {'PASS' if result_gate['owner_gate_pass'] else 'FAIL'} | {'PASS' if result_gate['ideal_gate_pass'] else 'FAIL'} |",
            "",
            "```json",
            json.dumps(result_gate, indent=2, ensure_ascii=False),
            "```",
            "",
            "## 效率",
            "",
            f"- fused 参数：`{experiment.get('params_fused')}`",
            f"- GFLOPs：`{experiment.get('gflops')}`",
            f"- test infer ms/image：`{test.get('infer_ms')}`",
            f"- test FPS：`{test.get('fps_infer_only')}`",
        ]
    else:
        lines += [f"当前状态：`{experiment.get('status', 'pending')}`。"]
        if experiment.get("error"):
            lines += ["", "```text", experiment["error"], "```"]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def orchestrate() -> int:
    import fcntl

    PROJECT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another MSDGS Backbone P3 C3k run is already active") from exc

    for required in (DATA, OFFICIAL_WEIGHTS, EVALUATOR, BUILD_REPORT):
        if not required.exists():
            raise FileNotFoundError(required)
    gpu_idle()
    archive = archive_previous_artifacts()
    cfg, generation = generate_candidate_yaml()
    payload = {
        "schema_version": 1,
        "started_at": now(),
        "candidate": RUN_NAME,
        "protocol": {
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "workers": WORKERS,
            "seed": SEED,
            "cache": False,
            "resume": False,
            "init_weights": str(OFFICIAL_WEIGHTS),
            "data": str(DATA),
            "final_eval": "independent subprocess reloads disk best.pt for val and test",
        },
        "baseline": MSDGS,
        "ideal": IDEAL,
        "generation": generation,
        "build_report": str(BUILD_REPORT),
        "archive": str(archive) if archive else None,
        "experiment": {"status": "training", "cfg": str(cfg)},
    }
    write_report(payload)
    try:
        save_status("training", "fresh seed0 train; resume disabled", cfg=str(cfg))
        run_checked([sys.executable, str(SCRIPT), "--worker-train", "--cfg", str(cfg)])
        weights = PROJECT / RUN_NAME / "weights" / "best.pt"
        save_status("evaluating", "independent process reloads disk best.pt", weights=str(weights))
        run_checked(
            [
                sys.executable,
                str(EVALUATOR),
                "--weights",
                str(weights),
                "--data",
                str(DATA),
                "--project",
                str(PROJECT),
                "--name",
                RUN_NAME,
                "--output",
                str(INDEPENDENT_REPORT),
                "--imgsz",
                str(IMGSZ),
                "--batch",
                str(BATCH),
            ]
        )
        result = json.loads(INDEPENDENT_REPORT.read_text(encoding="utf-8"))
        result.update(
            {
                "status": "done",
                "candidate": RUN_NAME,
                "cfg": str(cfg),
                "weights": str(weights),
                "gate": gate(result),
            }
        )
        payload["experiment"] = result
        payload["finished_at"] = now()
        write_report(payload)
        save_status(
            "done",
            "seed0 fresh train and independent val/test complete",
            weights=str(weights),
            independent_report=str(INDEPENDENT_REPORT),
            owner_gate_pass=result["gate"]["owner_gate_pass"],
            ideal_gate_pass=result["gate"]["ideal_gate_pass"],
        )
        print("MSDGS_BACKBONE_P3_C3K_DONE", REPORT_JSON, flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        payload["experiment"] = {
            "status": "failed",
            "cfg": str(cfg),
            "error": repr(exc),
            "failed_at": now(),
        }
        write_report(payload)
        save_status("failed", repr(exc), cfg=str(cfg))
        raise


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("--worker-train", action="store_true")
    command.add_argument("--cfg")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.worker_train:
        if not args.cfg:
            raise SystemExit("--worker-train requires --cfg")
        return train_worker(args.cfg)
    return orchestrate()


if __name__ == "__main__":
    raise SystemExit(main())