#!/usr/bin/env python
"""iter30 build 验证（不训练）：MSDGS neck + backbone 层6 C3k2→DySnakeC3k2。

构造基座 gsdown doc（end2end/reg_max=1）→ neck 4×VoVGSCSP 换 MSDGS(135eq)
→ backbone 层6 C3k2 换 DySnakeC3k2 → 建模 fuse，报参数/GFLOPs/end2end/reg_max/输出形状。

前置：install_msdgs_module.py + install_dysnake_module.py 已注入。
远程用法：
    python build_check_dysnake.py
"""
from __future__ import annotations

import copy
import sys

import torch
import yaml as _yaml

RED_LINE = 1_936_000  # gsdown fused params 红线


def base_gsdown_doc() -> dict:
    """gsdown (vovgscsp_gsdown) 基座，来自 module_sweep 生成的真实 YAML。"""
    return {
        "nc": 6,
        "end2end": True,
        "reg_max": 1,
        "scales": {
            "n": [0.5, 0.25, 1024],
            "s": [0.5, 0.5, 1024],
            "m": [0.5, 1.0, 512],
            "l": [1.0, 1.0, 512],
            "x": [1.0, 1.5, 512],
        },
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 2, "C3k2", [256, False, 0.25]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 2, "C3k2", [512, False, 0.25]],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 2, "C3k2", [512, True]],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 2, "C3k2", [1024, True]],
            [-1, 1, "SPPF", [1024, 5, 3, True]],
            [-1, 2, "C2PSA", [1024]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 2, "VoVGSCSP", [512]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 2, "VoVGSCSP", [256]],
            [-1, 1, "GSConv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 2, "VoVGSCSP", [512]],
            [-1, 1, "GSConv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 1, "VoVGSCSP", [1024]],
            [[16, 19, 22], 1, "Detect", ["nc"]],
        ],
    }


def swap_neck_to_msdgs(doc: dict, dilations=(1, 3, 5), fracs=(1, 1, 1)) -> dict:
    """把 head 里 4 个 VoVGSCSP 换成 MSDGS(135eq)。MSDGS args=[c2, True, 1, 0.5, dilations, fracs]。"""
    d = copy.deepcopy(doc)
    n_swapped = 0
    for layer in d["head"]:
        if layer[2] == "VoVGSCSP":
            c2 = layer[3][0]
            layer[2] = "MSDGS"
            layer[3] = [c2, True, 1, 0.5, list(dilations), list(fracs)]
            n_swapped += 1
    assert n_swapped == 4, f"expected 4 VoVGSCSP, swapped {n_swapped}"
    return d


def swap_backbone_layer_to_dysnake(doc: dict, idx: int = 6) -> dict:
    """把 backbone 第 idx 层 C3k2 换成 DySnakeC3k2（保留 args）。"""
    d = copy.deepcopy(doc)
    b = d["backbone"]
    assert b[idx][2] == "C3k2", f"backbone[{idx}] expected C3k2, got {b[idx][2]}"
    b[idx][2] = "DySnakeC3k2"
    return d


def build_and_report(doc: dict, tag: str) -> dict:
    from ultralytics import YOLO

    path = f"/root/autodl-tmp/neu-det-yolo26/generated_models_dysnake_e250/{tag}.yaml"
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

    model = YOLO(path)
    m = model.model
    end2end = getattr(m.model[-1], "end2end", None)
    reg_max = getattr(m.model[-1], "reg_max", None)
    nl = getattr(m.model[-1], "nl", None)

    model.model.eval()
    model.model.fuse()
    fused_params = sum(p.numel() for p in model.model.parameters())

    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        y = model.model(x)
    if isinstance(y, (list, tuple)):
        out_shape = tuple(y[0].shape) if hasattr(y[0], "shape") else str(type(y[0]))
    else:
        out_shape = tuple(y.shape)

    return {
        "tag": tag,
        "cfg": path,
        "fused_params": fused_params,
        "under_red_line": fused_params < RED_LINE,
        "end2end": bool(end2end) if end2end is not None else None,
        "reg_max": reg_max,
        "nl": nl,
        "out_shape": out_shape,
    }


def main() -> int:
    base = base_gsdown_doc()
    doc = swap_neck_to_msdgs(base)
    doc = swap_backbone_layer_to_dysnake(doc, idx=6)
    try:
        info = build_and_report(doc, "msdgs_dysnake_p4")
    except Exception as exc:  # noqa: BLE001
        print(f"BUILD_CHECK_DYSNAKE_FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    print("BUILD_CHECK_DYSNAKE_RESULT")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"RED_LINE(gsdown fused)= {RED_LINE}")
    print("BUILD_CHECK_DYSNAKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
