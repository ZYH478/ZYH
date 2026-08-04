#!/usr/bin/env python
"""Strict build gate for the official-path MSDGS Backbone P3 C3k candidate."""
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

from generate_msdgs_backbone_p3_c3k import (
    BASE_YAML,
    CANDIDATE,
    generate_candidate_yaml,
    validate_documents,
)

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_msdgs_backbone_p3_c3k_e250"
CHECK_DIR = PROJECT / "build_check"
EXPECTED = {
    "source_state_tensors": 708,
    "baseline_matched_tensors": 402,
    "baseline_matched_elements": 1_531_900,
    "candidate_matched_tensors": 396,
    "candidate_matched_elements": 1_522_556,
    "params_unfused": 1_907_644,
    "params_fused": 1_779_446,
    "gflops_fused": 4.1086464,
}


def output_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"cannot extract decoded tensor from {type(output)!r}")


def output_shape(output) -> tuple[int, ...]:
    return tuple(output_tensor(output).shape)


def model_gflops(model) -> float:
    from ultralytics.utils.torch_utils import get_flops

    return float(get_flops(model, imgsz=640))


def matched_tensor_stats(source, target) -> dict:
    source_state = source.state_dict()
    target_state = target.state_dict()
    matched = [
        (key, tensor)
        for key, tensor in source_state.items()
        if key in target_state and target_state[key].shape == tensor.shape
    ]
    return {
        "source_state_tensors": len(source_state),
        "target_state_tensors": len(target_state),
        "matched_tensors": len(matched),
        "matched_elements": int(sum(tensor.numel() for _, tensor in matched)),
    }


def assert_structure(model) -> dict:
    layer2 = model.model[2]
    layer4 = model.model[4]
    layer6 = model.model[6]
    layer8 = model.model[8]
    head = model.model[-1]

    assert type(layer2).__name__ == "C3k2"
    assert len(layer2.m) >= 1 and type(layer2.m[0]).__name__ == "Bottleneck"
    assert type(layer4).__name__ == "C3k2"
    assert len(layer4.m) >= 1 and type(layer4.m[0]).__name__ == "C3k"
    assert len(layer4.m[0].m) == 2
    assert all(type(unit).__name__ == "Bottleneck" for unit in layer4.m[0].m)
    for index, layer in ((6, layer6), (8, layer8)):
        assert type(layer).__name__ == "C3k2", (index, type(layer).__name__)
        assert len(layer.m) >= 1 and type(layer.m[0]).__name__ == "C3k", index
    assert head.end2end is True and head.reg_max == 1 and head.nl == 3
    detect_channels = tuple(int(tower[0].conv.in_channels) for tower in head.cv2)
    assert detect_channels == (64, 128, 256), detect_channels
    return {
        "layer2": type(layer2).__name__,
        "layer2_inner": type(layer2.m[0]).__name__,
        "layer4": type(layer4).__name__,
        "layer4_inner": type(layer4.m[0]).__name__,
        "layer4_c3k_bottlenecks": len(layer4.m[0].m),
        "layer6": type(layer6).__name__,
        "layer6_inner": type(layer6.m[0]).__name__,
        "layer8": type(layer8).__name__,
        "layer8_inner": type(layer8.m[0]).__name__,
        "end2end": bool(head.end2end),
        "reg_max": int(head.reg_max),
        "nl": int(head.nl),
        "detect_input_channels": list(detect_channels),
    }


def reload_worker(weights: Path) -> int:
    from ultralytics import YOLO

    wrapper = YOLO(str(weights), task="detect")
    model = wrapper.model.eval()
    structure = assert_structure(model)
    with torch.no_grad():
        shape = output_shape(model(torch.zeros(1, 3, 640, 640)))
    payload = {"weights": str(weights), "structure": structure, "eval_shape": list(shape)}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if shape == (1, 300, 6) else 2


def build_check() -> dict:
    from ultralytics import YOLO

    cfg, generation = generate_candidate_yaml()
    base_doc = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    candidate_doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    changed = validate_documents(base_doc, candidate_doc)
    assert changed == [4]

    source = YOLO(str(OFFICIAL_WEIGHTS), task="detect").model
    baseline = YOLO(str(BASE_YAML), task="detect").model
    wrapper = YOLO(str(cfg), task="detect")
    model = wrapper.model
    structure = assert_structure(model)

    baseline_match = matched_tensor_stats(source, baseline)
    candidate_match = matched_tensor_stats(source, model)
    assert baseline_match["source_state_tensors"] == EXPECTED["source_state_tensors"], baseline_match
    assert baseline_match["matched_tensors"] == EXPECTED["baseline_matched_tensors"], baseline_match
    assert baseline_match["matched_elements"] == EXPECTED["baseline_matched_elements"], baseline_match
    assert candidate_match["matched_tensors"] == EXPECTED["candidate_matched_tensors"], candidate_match
    assert candidate_match["matched_elements"] == EXPECTED["candidate_matched_elements"], candidate_match
    assert baseline_match["matched_tensors"] - candidate_match["matched_tensors"] == 6

    wrapper.load(str(OFFICIAL_WEIGHTS))
    model = wrapper.model.eval()
    structure_after_load = assert_structure(model)
    assert structure_after_load == structure

    probe = torch.zeros(1, 3, 640, 640)
    capture: dict[str, torch.Tensor] = {}
    hook = model.model[4].register_forward_hook(
        lambda _module, _inputs, output: capture.setdefault("unfused_layer4", output.detach())
    )
    with torch.no_grad():
        eval_shape = output_shape(model(probe))
    hook.remove()
    assert eval_shape == (1, 300, 6), eval_shape

    params_unfused = int(sum(parameter.numel() for parameter in model.parameters()))
    assert abs(params_unfused - EXPECTED["params_unfused"]) <= 64, params_unfused

    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECK_DIR / f"{CANDIDATE}_roundtrip.pt"
    wrapper.save(str(checkpoint))
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--reload-check", str(checkpoint)],
        text=True,
        capture_output=True,
        check=False,
    )
    print(child.stdout, end="")
    if child.stderr:
        print(child.stderr, file=sys.stderr, end="")
    assert child.returncode == 0, child.returncode

    fused = copy.deepcopy(model).eval()
    fused.fuse()
    fused_hook = fused.model[4].register_forward_hook(
        lambda _module, _inputs, output: capture.setdefault("fused_layer4", output.detach())
    )
    with torch.no_grad():
        fused_shape = output_shape(fused(probe))
    fused_hook.remove()
    assert fused_shape == (1, 300, 6), fused_shape
    fuse_layer4_error = float((capture["unfused_layer4"] - capture["fused_layer4"]).abs().max())
    assert fuse_layer4_error < 1e-3, fuse_layer4_error

    params_fused = int(sum(parameter.numel() for parameter in fused.parameters()))
    gflops = model_gflops(fused)
    assert abs(params_fused - EXPECTED["params_fused"]) <= 64, params_fused
    assert abs(gflops - EXPECTED["gflops_fused"]) <= 0.02, gflops

    return {
        "candidate": CANDIDATE,
        "cfg": str(cfg),
        "generation": generation,
        "changed_backbone_indices": changed,
        "structure": structure,
        "eval_shape": list(eval_shape),
        "fused_shape": list(fused_shape),
        "fuse_layer4_max_abs_error": fuse_layer4_error,
        "official_weight_match": {
            "baseline": baseline_match,
            "candidate": candidate_match,
        },
        "params_unfused": params_unfused,
        "params_fused": params_fused,
        "gflops_fused_inference": gflops,
        "expected": EXPECTED,
        "roundtrip_checkpoint": str(checkpoint),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload-check", type=Path)
    args = parser.parse_args()
    if args.reload_check:
        return reload_worker(args.reload_check)
    for required in (BASE_YAML, OFFICIAL_WEIGHTS):
        if not required.exists():
            raise FileNotFoundError(required)
    payload = build_check()
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    report = CHECK_DIR / "report.json"
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("BUILD_CHECK_MSDGS_BACKBONE_P3_C3K_OK", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())