#!/usr/bin/env python
"""Serial setup/build/train queue for MSDGS GIIS then P5-stage EMSC."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
QUEUE_DIR = ROOT / "runs_msdgs_literature_queue"
STATUS = QUEUE_DIR / "status.json"
LOCK = QUEUE_DIR / "queue.lock"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_status(state: dict) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    state["queue_pid"] = os.getpid()
    STATUS.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def run_step(state: dict, name: str, argv: list[str], fatal: bool) -> bool:
    state["status"] = "running"
    state["current"] = name
    step = {"name": name, "argv": argv, "started_at": now(), "status": "running"}
    state.setdefault("steps", []).append(step)
    write_status(state)
    print(f"\n===== STEP {name} =====", flush=True)
    print("COMMAND", " ".join(argv), flush=True)
    rc = subprocess.run(argv, cwd=ROOT).returncode
    step["returncode"] = rc
    step["finished_at"] = now()
    step["status"] = "done" if rc == 0 else "failed"
    write_status(state)
    if rc != 0:
        print(f"STEP_FAILED {name} rc={rc} fatal={fatal}", flush=True)
        if fatal:
            raise RuntimeError(f"fatal step failed: {name}, rc={rc}")
        return False
    return True


def main() -> int:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("QUEUE_ALREADY_RUNNING")
        return 3

    py = sys.executable
    state = {
        "schema_version": 1,
        "status": "starting",
        "current": "preflight",
        "order": ["msdgs_giis_k3", "msdgs_p5stage_emsc"],
        "protocol": {"epochs": 250, "imgsz": 640, "batch": 32, "seed": 0, "cache": False},
        "started_at": now(),
        "steps": [],
    }
    write_status(state)

    try:
        setup = [
            ("install_gsconv", [py, str(ROOT / "install_gsconv_modules.py")]),
            ("install_msdgs", [py, str(ROOT / "install_msdgs_module.py")]),
            ("install_emsc", [py, str(ROOT / "install_emsc_module.py")]),
            ("generate_giis", [py, str(ROOT / "make_giis_dataset.py")]),
            ("build_check", [py, str(ROOT / "build_check_giis_emsc.py")]),
        ]
        for name, argv in setup:
            run_step(state, name, argv, fatal=True)

        giis_ok = run_step(state, "train_msdgs_giis_k3", [
            py, "-u", str(ROOT / "train_giis_msdgs.py"),
            "--epochs", "250", "--batch", "32", "--imgsz", "640", "--seed", "0", "--force",
        ], fatal=False)
        emsc_ok = run_step(state, "train_msdgs_p5stage_emsc", [
            py, "-u", str(ROOT / "train_emsc_msdgs.py"),
            "--epochs", "250", "--batch", "32", "--imgsz", "640", "--seed", "0", "--force",
        ], fatal=False)

        state["status"] = "done" if giis_ok and emsc_ok else "partial_failed"
        state["current"] = None
        state["candidate_results"] = {"msdgs_giis_k3": giis_ok, "msdgs_p5stage_emsc": emsc_ok}
        state["finished_at"] = now()
        write_status(state)
        print("QUEUE_FINISHED", state["status"], flush=True)
        return 0 if giis_ok and emsc_ok else 1
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        state["status"] = "failed"
        state["error"] = repr(exc)
        state["finished_at"] = now()
        write_status(state)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
