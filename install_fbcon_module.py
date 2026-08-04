#!/usr/bin/env python
"""iter32 候选A（结构侧）：前景-背景对比增强分类头 FBHead（FBCon 模块）。

动机（承接 iter31 混淆矩阵诊断，病根首次坐实）：
- crazing 病根 = 前景/背景不可分：52.4% GT crazing 判成背景（漏检 recall 0.379），
  42.7% crazing 预测来自背景（误报 precision 0.539）。既不是回归、也不是类间混淆。
- crazing 是低对比度弥散纹理，前景相对其局部邻域背景的差异微弱但存在。用局部高通残差
  x - avgpool(x) 放大这种微弱前景对比信号，**只作用于分类分支**（判别前景/背景），
  直击「少检出 + 多误报」。

设计（吸取 iter28 教训 + 最小侵入）：
- FBHead 继承官方 Detect：每个检测层加一个极轻量 FBCon（depthwise 高通残差 + 标量 gamma）。
- **只增强分类路径**：override forward_head，scores 走 cls_head[i](fbcon[i](x[i]))；
  boxes 与 feats（anchor 用）仍用原始 x[i]。box 回归完全不受影响。
- 关键与 iter28 HFDGS 的区别：同样是 x - avgpool(x) 高通算子，iter28 放在 **neck 全局**
  被证伪（放大训练噪声、有害）；这里放在**分类分支前**放大前景-背景对比——诊断指向的
  正确位置。这是有依据的重新定位，不是重走死路。
- 极省参：depthwise conv（每层 c×9）+ 标量 gamma 初始化 0（起步=恒等，训练稳定），
  避免 1×1 pointwise 的 c² 参数撑爆红线（MSDGS 基座 1.777M，红线 1.936M，仅 159k 余量）。
- FBCon 在分类路径、fuse 后随 one2one 保留（推理生效），故计入推理 fused 参数——
  用 depthwise 保证增量可忽略。无自定义损失，不 patch init_criterion。

接线（与 UBHead / AuxDetect 同款范式）：
- FBHead ∈ head；须进 tasks.py parse_model 的 end2end 检测头 frozenset + legacy 集合。
- FBHead 类名不含 "detect"，task 推断会失败 → 训练/构建脚本显式传 task="detect"。

安全：改 tasks.py 前备份 tasks.py.fbcon_bak（存在则不覆盖）。幂等。
用法（远程）：FBCON_K=5 python install_fbcon_module.py  打印 INSTALL_FBCON_OK。
"""
from __future__ import annotations

import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated FBHead (foreground-background contrast cls head) for YOLO26 (NEU-DET iter32 cand A).
# Injected by install_fbcon_module.py. Do not edit by hand.
import os

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect

__all__ = ["FBHead"]

_FBCON_K = int(os.environ.get("FBCON_K", "5"))


class FBCon(nn.Module):
    """Foreground-background contrast enhancement (classification path only).

    Local high-pass residual amplifies low-contrast foreground (crazing) relative to
    its local-mean background: hp = x - avgpool_k(x); out = x + gamma * dw(hp).
    - gamma is a per-module scalar init 0 -> starts as identity (stable training start).
    - dw is a cheap depthwise 3x3 giving the high-pass a learnable spatial filter.
    Channel-preserving; negligible params.
    """

    def __init__(self, c, k=5):
        super().__init__()
        self.pool = nn.AvgPool2d(k, stride=1, padding=k // 2)
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        hp = x - self.pool(x)
        return x + self.gamma * self.dw(hp)


class FBHead(Detect):
    """Detect + foreground-background contrast enhancement on the classification branch.

    Each detection layer prepends an FBCon to the feature before the cls head only;
    box regression and anchor features use the original feature untouched. FBCon stays
    in the one2one path after fuse (inference-active, negligible params).
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.fbcon = nn.ModuleList(FBCon(x, k=_FBCON_K) for x in ch)

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1
        )
        scores = torch.cat(
            [cls_head[i](self.fbcon[i](x[i])).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        return dict(boxes=boxes, scores=scores, feats=x)
'''


PATCH_SRC_IMPORT = "from ultralytics.nn.modules.yolo26_fbcon import FBHead  # noqa: E402,F401\n"


def inject_init_export(text: str) -> tuple[str, bool]:
    if "yolo26_fbcon" in text:
        return text, False
    return text + (
        "\n# FBHead foreground-background contrast cls head\n"
        "from .yolo26_fbcon import FBHead  # noqa: E402,F401\n"
    ), True


def inject_tasks_import(text: str) -> tuple[str, bool]:
    if "yolo26_fbcon" in text:
        return text, False
    loss_anchor = "from ultralytics.utils.loss import ("
    if loss_anchor in text:
        return text.replace(loss_anchor, PATCH_SRC_IMPORT + loss_anchor, 1), True
    return text + "\n" + PATCH_SRC_IMPORT, True


def inject_frozenset(text: str) -> tuple[str, bool]:
    if "                FBHead,\n" in text:
        return text, False
    marker = "                Detect,\n"  # 16-space indent, only inside the multi-line frozenset
    if marker not in text:
        return text, False
    return text.replace(marker, marker + "                FBHead,\n", 1), True


def inject_legacy_set(text: str) -> tuple[str, bool]:
    if "if m in {FBHead, " in text or "FBHead, Detect," in text:
        return text, False
    # tolerate whatever prefix earlier installers left ({AuxDetect, Detect, / {UBHead, ...).
    for marker in ("if m in {UBHead, ", "if m in {AuxDetect, Detect,", "if m in {Detect,"):
        if marker in text:
            repl = marker.replace("{", "{FBHead, ", 1)
            return text.replace(marker, repl, 1), True
    return text, False


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_fbcon.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_init_export(it)
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_export_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".fbcon_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_tasks_import(tk)
    tk, frozen_changed = inject_frozenset(tk)
    tk, legacy_changed = inject_legacy_set(tk)
    if imp_changed or frozen_changed or legacy_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_frozenset_changed", frozen_changed)
    print("tasks_legacy_set_changed", legacy_changed)

    print("INSTALL_FBCON_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
