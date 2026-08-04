#!/usr/bin/env python
"""iter21 build 验证：4 个组合能否构建成模型 + end2end/reg_max=1/nl=3/输出[1,300,6] + fused 参数量。"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
GEN_DIR = ROOT / "generated_models_iter21_combo_e250"

GSDOWN_YAML = ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"
WINNER_YAML = ROOT / "generated_models_module_combo2_e250" / "y26n_s2_spd_p3_dysample_e250.yaml"

SPECS = {
    "y26n_i21_gsdown_pki_e250": (GSDOWN_YAML, {2: "PKIC3k2", 4: "PKIC3k2"}),
    "y26n_i21_gsdown_pki_dwr_e250": (GSDOWN_YAML, {2: "PKIC3k2", 4: "PKIC3k2", 6: "DWRC3k2", 8: "DWRC3k2"}),
    "y26n_i21_winner_dwr_e250": (WINNER_YAML, {6: "DWRC3k2", 8: "DWRC3k2"}),
    "y26n_i21_winner_pki_dwr_e250": (WINNER_YAML, {2: "PKIC3k2", 4: "PKIC3k2", 6: "DWRC3k2", 8: "DWRC3k2"}),
}


def build_doc(base_yaml: Path, swaps: dict) -> dict:
    doc = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    bb = doc["backbone"]
    for idx, mod in swaps.items():
        cur = bb[idx][2]
        assert cur == "C3k2", f"layer {idx} is {cur}, not C3k2"
        bb[idx][2] = mod
    return doc


GEN_DIR.mkdir(parents=True, exist_ok=True)
for name, (base_yaml, swaps) in SPECS.items():
    doc = build_doc(base_yaml, swaps)
    out = GEN_DIR / f"{name}.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    model = YOLO(str(out))
    m = model.model
    head = m.model[-1]
    nl = getattr(head, "nl", None)
    e2e = getattr(head, "end2end", None)
    reg_max = getattr(head, "reg_max", None)
    mf = model.model.fuse() if hasattr(model.model, "fuse") else m
    params = sum(p.numel() for p in mf.parameters())
    x = torch.zeros(1, 3, 640, 640)
    mf.eval()
    with torch.no_grad():
        y = mf(x)
    if isinstance(y, (list, tuple)):
        shape = tuple(y[0].shape) if hasattr(y[0], "shape") else str(type(y[0]))
    else:
        shape = tuple(y.shape)
    print(f"{name}: params={params/1e6:.4f}M nl={nl} end2end={e2e} reg_max={reg_max} out={shape}")
print("BUILD_CHECK_ITER21_DONE")
