#!/usr/bin/env python
"""iter32 候选B（损失侧，零结构参数）：把分类损失从 BCE 换成 Focal + crazing 定向类加权。

动机（承接 iter31 混淆矩阵诊断，病根首次坐实）：
- crazing 的病根不是回归、不是类间混淆，而是**前景/背景不可分**：
  test 上 52.4% 的 GT crazing 被判成背景（漏检，recall 0.379 全场最低），
  且 42.7% 的 crazing 预测来自背景（误报，precision 0.539 全场最低）。
- 这是典型的低对比度前景 + 类难度失衡问题。Focal 聚焦难样本（压低易分背景负样本、
  逼模型不漏低置信真 crazing）+ crazing 定向类加权（放大 crazing 通道的损失贡献），
  直击「少检出 + 多误报」。

设计（最小侵入，规避 VFL 崩盘教训）：
- _FocalBCE：reduction="none" 的 focal 调制 BCE，返回**逐元素** loss（与官方
  BCEWithLogitsLoss(reduction="none") 完全同形状 (bs, anchors, nc)），
  使 loss.py 第 436-438 行既有的 `bce_loss *= self.class_weights` 类加权钩子
  与归一化逻辑原样生效，不改调用约定。
- 只替换 self.bce（分类损失）；**不碰 assigner、不碰 target_scores 软标签生成**——
  这是与 VFL 崩盘的本质区别（VFL 死在改软标签逻辑与 one2one 分配冲突；
  Focal 只在既有软标签上乘调制因子，end2end/one2one 完全兼容）。
- crazing 定向加权走 loss.py 已内置的 self.class_weights 钩子
  （v8DetectionLoss.__init__ 读 getattr(model, "class_weights", None) 并 reshape 成 (1,1,nc)）。
  训练脚本设 model.model.class_weights 即可，零额外接线。
- 幂等 monkey-patch v8DetectionLoss.__init__：调原 init 后把 self.bce 换成 _FocalBCE。
  one2many / one2one 两条损失都经此 init，自动一致。

参数：FOCALCW_GAMMA 环境变量控制 focal gamma（默认 1.5，官方默认值）。
推理零影响（损失只在训练期用），fused 参数 / 结构 / 输出与基线完全一致。

用法（远程）：
    FOCALCW_GAMMA=1.5 python install_focalcw_module.py   # 幂等，打印 INSTALL_FOCALCW_OK
"""
from __future__ import annotations

import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated FocalCW classification loss for YOLO26 (NEU-DET iter32 candidate B).
# Injected by install_focalcw_module.py. Do not edit by hand.
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalBCE"]


class FocalBCE(nn.Module):
    """Per-element focal-modulated BCE (reduction="none"), drop-in for nn.BCEWithLogitsLoss.

    Returns the SAME shape as BCEWithLogitsLoss(reduction="none") so the existing
    class_weights multiplication + normalization in v8DetectionLoss stays untouched.
    Only the modulating factor (1-p_t)^gamma is added, down-weighting easy examples
    (easy background negatives) and focusing on hard-to-classify ones (low-contrast
    crazing foreground). No alpha here — crazing-specific weighting goes through the
    built-in class_weights hook, keeping the two concerns cleanly separated.
    """

    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        if self.gamma > 0:
            pred_prob = pred.sigmoid()
            p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
            loss = loss * (1.0 - p_t).clamp_(min=0.0) ** self.gamma
        return loss
'''


PATCH_SRC = '''

# ===== FocalCW classification loss wiring (injected by install_focalcw_module.py) =====
import os as _ub_os  # noqa: E402
from ultralytics.nn.modules.yolo26_focalcw import FocalBCE as _FocalBCE  # noqa: E402

_FOCALCW_GAMMA = float(_ub_os.environ.get("FOCALCW_GAMMA", "1.5"))

if not hasattr(v8DetectionLoss, "_focalcw_orig_init"):
    v8DetectionLoss._focalcw_orig_init = v8DetectionLoss.__init__

    def _focalcw_init(self, model, tal_topk=10, tal_topk2=None):
        v8DetectionLoss._focalcw_orig_init(self, model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        # swap BCE -> focal-modulated BCE (per-element; class_weights hook still applies).
        self.bce = _FocalBCE(gamma=_FOCALCW_GAMMA)

    v8DetectionLoss.__init__ = _focalcw_init
# ===== end FocalCW classification loss wiring =====
'''


def inject_once(text: str, marker: str, addition: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + addition, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_focalcw.py"
    loss_file = pkg / "utils" / "loss.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    backup = loss_file.with_name(loss_file.name + ".focalcw_bak")
    if not backup.exists():
        shutil.copy(loss_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    lk = loss_file.read_text(encoding="utf-8")
    lk, patched = inject_once(lk, "_focalcw_init", PATCH_SRC)
    if patched:
        loss_file.write_text(lk, encoding="utf-8")
    print("loss_patched", patched)

    print("INSTALL_FOCALCW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
