#!/usr/bin/env python
"""把 O2M-QFL（one2many 辅助头 Quality Focal 难例强调）patch 进 loss.py。

设计动机（iter34，诊断驱动，跳出 crazing 攻全局 mean AP）：
gsdown 干净基线混淆矩阵（test）揭示：**类间混淆几乎为零**（仅 1 个 inclusion→crazing），
100% 误差预算 = 前景/背景：180 个缺陷漏检成背景 + 146 个背景误报成缺陷。可回收 mean AP
空间在 rolled-in_scale(0.602)/inclusion(0.740)，非封顶的 crazing。此前 6 次全攻错位置
（crazing 标注天花板 / 特征空间前景增强已耗尽）。

O2M-QFL 与 FocalCW 崩盘的关键分野：
- FocalCW 全局 monkey-patch v8DetectionLoss.bce → 同时污染 one2one 推理头置信度校准
  （VFL/NWD 同源教训），crazing recall 崩到 0.086、整机 map50 崩 13.5pp。
- YOLOv26 是**双分配**架构：one2many(o2m, topk=10)=训练引擎，one2one(o2o, topk=1)=推理头。
  O2M-QFL **只替换 self.one2many.bce**，self.one2one.bce 保持纯 BCEWithLogitsLoss →
  推理头校准分毫不动，绕开崩盘根因。这是针对双分配 NMS-free 架构的原创设计。

机理选型 = Quality Focal（非标准 focal）：
    调制因子 = |target - sigmoid(pred)|^beta
    o2m 的 target_scores 是 TAL 软标签(quality×onehot, 连续)。标准 focal 假设二值标签，
    对软标签会崩；QFL 专为软标签设计。该因子同时强调：
    - 难正例(target 高、pred 低 = 180 漏检) → 因子大 → 强调
    - 难负例(target 0、pred 高 = 146 背景误报) → 因子大 → 强调
    - 易例(pred≈target) → 因子≈0 → 压制
    精准打在诊断出的 fg/bg 误差预算，攻全局 mean AP。

隔离（双重）：
1. env-gated：O2M_QFL_ENABLE=1 时才生效，未设则 patch 惰性无害（不污染其它模型训练）。
2. 只替换 one2many.bce，one2one.bce 保持纯 BCE。
beta 由 O2M_QFL_BETA 控制（默认 2.0，QFL 标准值）。

安全：首次运行前备份 loss.py 为 loss.py.o2mqfl_bak；幂等，重复运行不重复注入。
用法（远程）：python install_o2mqfl_module.py  → 成功打印 INSTALL_O2MQFL_OK。
"""
from __future__ import annotations

import shutil
from pathlib import Path


PATCH_MARKER = "# === iter34 O2M-QFL patch ==="

PATCH_SRC = '''

# === iter34 O2M-QFL patch ===
# one2many 辅助头 Quality Focal 难例强调；one2one 推理头保持纯 BCE。
# env-gated：仅 O2M_QFL_ENABLE=1 生效，否则惰性无害。
import os as _os_o2mqfl
import torch.nn.functional as _F_o2mqfl


class _O2MQualityFocal(nn.Module):
    """Quality Focal 调制的 BCE（reduction="none"），仅用于 one2many 辅助头。

    返回逐元素 loss，形状 (bs, num_anchors, nc)，与 BCEWithLogitsLoss(reduction="none")
    完全兼容，故 loss.py 中 `self.bce(pred, target)` 调用、class_weights 乘法、
    `.sum()/target_scores_sum` 归约全部不变。
    """

    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, pred, target):
        bce = _F_o2mqfl.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if self.beta > 0:
            p = pred.sigmoid()
            scale = (target - p).abs().pow(self.beta)
            bce = bce * scale
        return bce


if _os_o2mqfl.environ.get("O2M_QFL_ENABLE", "0") == "1":
    _O2M_QFL_BETA = float(_os_o2mqfl.environ.get("O2M_QFL_BETA", "2.0"))

    def _make_o2mqfl_init(_orig_init):
        def _patched_init(self, model, *a, **kw):
            _orig_init(self, model, *a, **kw)
            # 只替换 one2many(训练引擎)的 bce；one2one(推理头)保持纯 BCE。
            self.one2many.bce = _O2MQualityFocal(_O2M_QFL_BETA)
        return _patched_init

    # YOLOv26 实际用 E2ELoss（one2one topk=7/tal_topk2=1）；E2EDetectLoss 兜底一并挂。
    _o2mqfl_patched = []
    for _cls_name in ("E2ELoss", "E2EDetectLoss"):
        _cls = globals().get(_cls_name)
        if _cls is not None:
            _cls.__init__ = _make_o2mqfl_init(_cls.__init__)
            _o2mqfl_patched.append(_cls_name)
    print(f"[O2M-QFL] enabled beta={_O2M_QFL_BETA}; patched {_o2mqfl_patched}; "
          f"one2many.bce -> QualityFocal; one2one.bce pure BCE")
# === end iter34 O2M-QFL patch ===
'''


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    loss_file = pkg / "utils" / "loss.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    text = loss_file.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        print("patch_already_present True")
        print("INSTALL_O2MQFL_OK")
        return 0

    backup = loss_file.with_name(loss_file.name + ".o2mqfl_bak")
    if not backup.exists():
        shutil.copy(loss_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    # 确认 E2EDetectLoss 存在（patch 依赖它）
    assert "class E2EDetectLoss" in text, "E2EDetectLoss not found in loss.py"

    text = text + PATCH_SRC
    loss_file.write_text(text, encoding="utf-8")
    print("patch_appended True")
    print("INSTALL_O2MQFL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
