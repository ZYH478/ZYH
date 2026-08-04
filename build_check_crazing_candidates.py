#!/usr/bin/env python
"""iter32 build 验证（不训练）：crazing 前景/背景可分性双候选。

诊断坐实（gsdown test 混淆矩阵）：crazing 病根是前景/背景不可分——
52.4% GT 判成背景(漏检)、42.7% crazing 预测来自背景(误报)，类间混淆≈0。
前四轮（backbone×3/DySnake/UBHead）都治错了病（治的是回归/特征，不是可分性）。

双候选对照（都在 MSDGS neck 基座上，从 yolo26n.pt 迁移，保 end2end/reg_max=1）：
- FocalCW（损失侧，零结构参数）：分类损失 BCE→Focal(聚焦难样本) + crazing 定向类加权。
  模型结构与 MSDGS 基线完全一致，改动只在 loss——build 主要确认模型正常构建 +
  init_criterion 分发到 FocalCW 损失。
- FBCon（结构侧，轻量）：分类分支前置前景-背景对比模块（depthwise 高通残差
  x-avgpool(x)，gamma 初始化 0=起步恒等）。只增强分类路径，box/feats 不碰。
  build 需完整验证推理 fused 参数/end2end/reg_max/输出形状。

用法（远程，先注入依赖链）：
    python install_gsconv_modules.py
    python install_msdgs_module.py
    python install_focalcw_module.py
    python install_fbcon_module.py
    python build_check_crazing_candidates.py
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


def swap_head_to_fbcon(doc: dict) -> dict:
    """把 Detect 换成 FBHead（结构侧候选 A）。"""
    d = copy.deepcopy(doc)
    for layer in d["head"]:
        if layer[2] == "Detect":
            layer[2] = "FBHead"
    return d


def build_and_report(doc: dict, tag: str, task: str = "detect") -> dict:
    from ultralytics import YOLO
    import os

    path = f"/root/autodl-tmp/neu-det-yolo26/generated_models_crazing_e250/{tag}.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

    model = YOLO(path, task=task)
    m = model.model
    head = m.model[-1]
    head_type = type(head).__name__
    end2end = getattr(head, "end2end", None)
    reg_max = getattr(head, "reg_max", None)
    nl = getattr(head, "nl", None)

    # 训练模式前向：确认返回 dict 且不报错（尤其 FBCon 的 forward_head override）
    m.train()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        tr = m(x)
    train_is_dict = isinstance(tr, dict)

    # 推理模式 fuse
    m.eval()
    model.model.fuse()
    fused_params = sum(p.numel() for p in model.model.parameters())
    with torch.no_grad():
        y = model.model(x)
    if isinstance(y, (list, tuple)):
        out_shape = tuple(y[0].shape) if hasattr(y[0], "shape") else str(type(y[0]))
    else:
        out_shape = tuple(y.shape)

    return {
        "tag": tag, "cfg": path, "head_type": head_type,
        "fused_params": fused_params, "under_red_line": fused_params < RED_LINE,
        "end2end": bool(end2end) if end2end is not None else None,
        "reg_max": reg_max, "nl": nl,
        "train_forward_is_dict": train_is_dict,
        "out_shape": out_shape,
    }


def check_loss_wiring() -> dict:
    """确认 FocalCW 损失分发已挂载（模型用标准 Detect，损失在 init_criterion 层分发）。"""
    from ultralytics import YOLO
    base = swap_neck_to_msdgs(base_gsdown_doc())
    import os
    path = "/root/autodl-tmp/neu-det-yolo26/generated_models_crazing_e250/msdgs_focalcw.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)
    model = YOLO(path, task="detect")
    m = model.model
    # 挂 crazing 定向类权重（训练脚本也会挂），确认 init_criterion 能拿到 FocalCW 损失
    m.class_weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    crit = m.init_criterion()
    crit_type = type(crit).__name__
    # 探测内部 one2one 损失是否用了 Focal
    inner = getattr(crit, "one2one", None)
    bce_type = type(getattr(inner, "bce", None)).__name__ if inner is not None else "NA"
    head = m.model[-1]
    m.eval(); model.model.fuse()
    fused = sum(p.numel() for p in model.model.parameters())
    return {
        "tag": "msdgs_focalcw", "cfg": path,
        "head_type": type(head).__name__,
        "criterion_type": crit_type,
        "inner_bce_type": bce_type,
        "fused_params": fused, "under_red_line": fused < RED_LINE,
        "note": "结构=MSDGS基线(标准Detect)，改动只在损失",
    }


def main() -> int:
    print("=== 候选 B: FocalCW（损失侧，结构=MSDGS 基线）===")
    try:
        b = check_loss_wiring()
        for k, v in b.items():
            print(f"  {k}: {v}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FOCALCW_CHECK_FAILED: {exc}", file=sys.stderr)
        return 1

    print("=== 候选 A: FBCon（结构侧，分类分支前置对比模块）===")
    try:
        doc = swap_head_to_fbcon(swap_neck_to_msdgs(base_gsdown_doc()))
        a = build_and_report(doc, "msdgs_fbcon")
        for k, v in a.items():
            print(f"  {k}: {v}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"FBCON_CHECK_FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"RED_LINE(gsdown fused)= {RED_LINE}")
    print("BUILD_CHECK_CRAZING_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
