#!/usr/bin/env python
"""Build/fuse/forward/save/reload checks for GIIS and P5-stage EMSC candidates."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml
from ultralytics import YOLO

import train_emsc_msdgs as E

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
BASE_YAML = E.MSDGS_YAML
CHECK_DIR = ROOT / "generated_models_literature_checks"


def output_shape(output):
    if torch.is_tensor(output):
        return tuple(output.shape)
    if isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
        return tuple(output[0].shape)
    return None


def reload_worker(weights: Path) -> int:
    model = YOLO(str(weights))
    model.model.eval()
    with torch.no_grad():
        shape = output_shape(model.model(torch.zeros(1, 3, 640, 640)))
    print(json.dumps({"reload": str(weights), "shape": shape,
                      "layer8": type(model.model.model[8]).__name__}))
    return 0 if shape == (1, 300, 6) else 2


def build_cfg(name: str, doc: dict) -> Path:
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CHECK_DIR / f"{name}.yaml"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg


def check_candidate(name: str, doc: dict, expected_layer8: str) -> dict:
    cfg = build_cfg(name, doc)
    wrapper = YOLO(str(cfg), task="detect").load(str(E.OFFICIAL_WEIGHTS))
    model = wrapper.model.eval()
    head = model.model[-1]
    assert getattr(model, "end2end", None) is True
    assert int(head.reg_max) == 1 and int(head.nl) == 3
    assert type(model.model[8]).__name__ == expected_layer8
    detect_inputs = [int(tower[0].conv.in_channels) for tower in head.cv2]
    assert detect_inputs == [64, 128, 256], detect_inputs
    with torch.no_grad():
        shape = output_shape(model(torch.zeros(1, 3, 640, 640)))
    assert shape == (1, 300, 6), shape

    checkpoint = CHECK_DIR / f"{name}_reload.pt"
    wrapper.save(str(checkpoint))
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--reload-check", str(checkpoint)],
        text=True, capture_output=True, check=False,
    )
    print(child.stdout, end="")
    if child.stderr:
        print(child.stderr, file=sys.stderr, end="")
    assert child.returncode == 0, f"reload failed rc={child.returncode}"

    fused = copy.deepcopy(model).eval()
    fused.fuse()
    with torch.no_grad():
        fused_shape = output_shape(fused(torch.zeros(1, 3, 640, 640)))
    assert fused_shape == (1, 300, 6), fused_shape
    fused_params = int(sum(p.numel() for p in fused.parameters()))
    return {
        "name": name, "cfg": str(cfg), "layer8": expected_layer8,
        "end2end": True, "reg_max": 1, "nl": 3,
        "detect_input_channels": detect_inputs, "shape": shape,
        "fused_shape": fused_shape, "fused_params": fused_params,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload-check", type=Path)
    args = parser.parse_args()
    if args.reload_check:
        return reload_worker(args.reload_check)

    base_doc = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    emsc_doc = E.build_emsc()
    results = [
        check_candidate("msdgs_giis_structure", base_doc, "C3k2"),
        check_candidate("msdgs_p5stage_emsc", emsc_doc, "C3k2EMSC"),
    ]
    baseline_params = results[0]["fused_params"]
    emsc_params = results[1]["fused_params"]
    assert baseline_params == 1_777_318, baseline_params
    assert emsc_params > baseline_params
    print("BUILD_CHECK_OK")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
