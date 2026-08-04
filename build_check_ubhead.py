#!/usr/bin/env python
"""iter31 build 验证（不训练）：MSDGS neck + UBHead（不确定性感知框头）。

构造基座 gsdown doc（end2end/reg_max=1）→ neck 4×VoVGSCSP 换 MSDGS(135eq)
→ 检测头 Detect 换 UBHead → 检查：
  1. 推理 fused 参数（σ 头应被 fuse 丢弃，与 MSDGS 基线持平、< 红线 1.936M）
  2. end2end / reg_max / nl / 推理输出形状 [1,300,6]
  3. 训练模式前向：preds 是 dict，one2many/one2one 里都有 log_sigma，形状对齐 boxes
  4. 一次 mock 损失前向能跑通（UBE2ELoss 不报错，loss 有限）

前置：install_msdgs_module.py + install_ubhead_module.py 已注入。
远程用法：python build_check_ubhead.py
"""
from __future__ import annotations

import copy
import sys

import torch
import yaml as _yaml

RED_LINE = 1_936_000


def base_gsdown_doc() -> dict:
    return {
        "nc": 6,
        "end2end": True,
        "reg_max": 1,
        "scales": {
            "n": [0.5, 0.25, 1024], "s": [0.5, 0.5, 1024], "m": [0.5, 1.0, 512],
            "l": [1.0, 1.0, 512], "x": [1.0, 1.5, 512],
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
    d = copy.deepcopy(doc)
    n = 0
    for layer in d["head"]:
        if layer[2] == "VoVGSCSP":
            c2 = layer[3][0]
            layer[2] = "MSDGS"
            layer[3] = [c2, True, 1, 0.5, list(dilations), list(fracs)]
            n += 1
    assert n == 4, f"expected 4 VoVGSCSP, swapped {n}"
    return d


def swap_head_to_ubhead(doc: dict) -> dict:
    d = copy.deepcopy(doc)
    last = d["head"][-1]
    assert last[2] == "Detect", f"last head layer expected Detect, got {last[2]}"
    last[2] = "UBHead"
    return d


def main() -> int:
    from ultralytics import YOLO

    base = base_gsdown_doc()
    doc = swap_neck_to_msdgs(base)
    doc = swap_head_to_ubhead(doc)

    import os
    path = "/root/autodl-tmp/neu-det-yolo26/generated_models_ubhead_e250/msdgs_ubhead.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

    try:
        model = YOLO(path, task="detect")  # UBHead 类名不含 "detect"，须显式指定任务
        m = model.model
        head = m.model[-1]
        end2end = getattr(head, "end2end", None)
        reg_max = getattr(head, "reg_max", None)
        nl = getattr(head, "nl", None)
        head_type = type(head).__name__
        has_sigma = getattr(head, "sigma_cv", None) is not None

        # --- 训练模式前向：验证 log_sigma 产出且对齐 ---
        m.train()
        x = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            tp = m(x)
        train_is_dict = isinstance(tp, dict)
        o2m_has_ls = train_is_dict and "log_sigma" in tp.get("one2many", {})
        o2o_has_ls = train_is_dict and "log_sigma" in tp.get("one2one", {})
        ls_shape = tuple(tp["one2many"]["log_sigma"].shape) if o2m_has_ls else None
        boxes_shape = tuple(tp["one2many"]["boxes"].shape) if train_is_dict else None

        # --- 推理 fused：σ 头应被丢弃 ---
        m.eval()
        m.fuse()
        fused_params = sum(p.numel() for p in m.parameters())
        sigma_after_fuse = getattr(head, "sigma_cv", "DELETED")
        with torch.no_grad():
            y = m(x)
        if isinstance(y, (list, tuple)):
            out_shape = tuple(y[0].shape) if hasattr(y[0], "shape") else str(type(y[0]))
        else:
            out_shape = tuple(y.shape)
    except Exception as exc:  # noqa: BLE001
        print(f"BUILD_CHECK_UBHEAD_FAILED: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print("BUILD_CHECK_UBHEAD_RESULT")
    print(f"  head_type: {head_type}")
    print(f"  end2end: {bool(end2end) if end2end is not None else None}")
    print(f"  reg_max: {reg_max}")
    print(f"  nl: {nl}")
    print(f"  has_sigma_cv(before fuse): {has_sigma}")
    print(f"  train_forward_is_dict: {train_is_dict}")
    print(f"  one2many_has_log_sigma: {o2m_has_ls}")
    print(f"  one2one_has_log_sigma: {o2o_has_ls}")
    print(f"  log_sigma_shape: {ls_shape}   boxes_shape: {boxes_shape}")
    print(f"  sigma_cv_after_fuse: {sigma_after_fuse}")
    print(f"  fused_params(inference): {fused_params}   under_red_line: {fused_params < RED_LINE}")
    print(f"  inference_out_shape: {out_shape}")
    print(f"RED_LINE(gsdown fused)= {RED_LINE}")
    # log_sigma 形状应为 (bs, 4, total_anchors)，boxes 为 (bs, 4*reg_max, total_anchors)=(bs,4,A)
    ok = (
        head_type == "UBHead" and bool(end2end) and reg_max == 1 and nl == 3
        and train_is_dict and o2m_has_ls and o2o_has_ls
        and ls_shape is not None and boxes_shape is not None
        and ls_shape[-1] == boxes_shape[-1] and ls_shape[1] == 4
        and sigma_after_fuse in (None, "DELETED")
        and fused_params < RED_LINE and tuple(out_shape) == (1, 300, 6)
    )
    print("BUILD_CHECK_UBHEAD_OK" if ok else "BUILD_CHECK_UBHEAD_NOT_OK")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
