#!/usr/bin/env python
"""把轻量 AuxDetect（辅助训练头）+ AuxE2ELoss 装进当前 ultralytics 8.4.93。

设计（依据 owner 定义）：
- AuxDetect 继承官方 Detect：训练时额外挂 aux_cv2/aux_cv3 两个辅助分支
  （辅助 bbox 回归 + 辅助分类），forward 在 training 时于返回 dict 里加 "aux" key。
- 推理 / fuse 时丢弃 aux 分支（连同 one2many 主分支），推理零额外成本，
  参数量 / FLOPs / FPS 与原模型完全一致。
- AuxE2ELoss 继承官方 E2ELoss：total = o2m*L_o2m + o2o*L_o2o + 0.25*L_aux，
  aux 用 tal_topk=10 的密集分配（和 one2many 同口径）。

接线点（远程 8.4.93 源码探明）：
- Detect ∈ head；E2ELoss / v8DetectionLoss ∈ utils.loss。
- tasks.py parse_model 第 1992-2006 frozenset：end2end 检测头在此拿到
  args.extend([reg_max, end2end, ch])——AuxDetect 必须进这个集合。
- tasks.py 第 2010 行 `m.legacy = legacy` 集合：AuxDetect 也要进。
- DetectionModel.init_criterion（tasks.py 第 596）硬编码 E2ELoss；用文件末尾
  monkey-patch 覆盖：检测头是 AuxDetect 且 end2end 时改用 AuxE2ELoss。
- 其余 isinstance(m, Detect) 判定（stride/bias_init/end2end 处理）AuxDetect
  作为子类自动命中，无需改。

安全：改 tasks.py 前备份为 tasks.py.auxdetect_bak（存在则不覆盖）。幂等。
用法（远程）：python install_auxdetect.py  成功打印 INSTALL_AUXDETECT_OK。
"""
from __future__ import annotations

import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated AuxDetect (auxiliary training head) + AuxE2ELoss for YOLO26.
# Injected by install_auxdetect.py. Do not edit by hand.
import copy

import torch  # noqa: F401
import torch.nn as nn  # noqa: F401

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

__all__ = ["AuxDetect", "AuxE2ELoss"]


class AuxDetect(Detect):
    """Detect + 轻量辅助训练头。

    训练时额外 aux_cv2（辅助 bbox 回归）/ aux_cv3（辅助分类）分支提供辅助监督，
    forward 在 training 时于返回 dict 里加 "aux" key；推理 / fuse 时丢弃，零额外成本。
    aux 分支使用未 detach 的特征，梯度回传 backbone/neck，这是辅助头的目的。
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        # 复用主分支结构建辅助分支（同通道/层数），参数在此注册以便 stride 计算。
        self.aux_cv2 = copy.deepcopy(self.cv2)
        self.aux_cv3 = copy.deepcopy(self.cv3)

    def bias_init(self):
        # 主分支 / one2one 先按官方逻辑初始化 bias，再让 aux 继承初始化好的权重作为起点。
        super().bias_init()
        self.aux_cv2 = copy.deepcopy(self.cv2)
        self.aux_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
            if self.training and getattr(self, "aux_cv2", None) is not None:
                preds["aux"] = self.forward_head(x, box_head=self.aux_cv2, cls_head=self.aux_cv3)
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def fuse(self):
        # 推理优化：删掉 one2many 主分支与 aux 辅助分支，只留 one2one。
        self.cv2 = self.cv3 = None
        self.aux_cv2 = self.aux_cv3 = None


class AuxE2ELoss(E2ELoss):
    """E2ELoss + aux 分支监督。total = o2m*L_o2m + o2o*L_o2o + aux_gain*L_aux。"""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        super().__init__(model, loss_fn)
        # aux 用密集分配（tal_topk=10），与 one2many 同口径。
        self.aux = loss_fn(model, tal_topk=10)
        self.aux_gain = 0.25

    def __call__(self, preds, batch):
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        total = loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o
        if isinstance(preds, dict) and "aux" in preds:
            loss_aux = self.aux.loss(preds["aux"], batch)
            total = total + self.aux_gain * loss_aux[0]
        return total, loss_one2one[1]
'''


PATCH_SRC = '''

# ===== AuxDetect loss wiring (injected by install_auxdetect.py) =====
def _aux_init_criterion(self):
    """检测头是 AuxDetect 且 end2end 时用 AuxE2ELoss，否则回落官方逻辑。"""
    from ultralytics.nn.modules.yolo26_auxdetect import AuxDetect, AuxE2ELoss

    m = self.model[-1]
    if isinstance(m, AuxDetect) and getattr(self, "end2end", False):
        return AuxE2ELoss(self)
    return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


DetectionModel.init_criterion = _aux_init_criterion
# ===== end AuxDetect loss wiring =====
'''


def inject_once(text: str, marker: str, addition: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + addition, True


def inject_init_export(text: str) -> tuple[str, bool]:
    if "yolo26_auxdetect" in text:
        return text, False
    return text + (
        "\n# AuxDetect auxiliary training head\n"
        "from .yolo26_auxdetect import AuxDetect  # noqa: E402,F401\n"
    ), True


def inject_tasks_import(text: str) -> tuple[str, bool]:
    if "yolo26_auxdetect" in text:
        return text, False
    # 追加到文件顶部 import 之后不易定位，直接在文件里靠 head import 行后插入。
    anchor = "from ultralytics.nn.modules import ("
    if anchor in text:
        # 在模块 import 块之外单独加一行，放在第一处 import 之前的安全位置：文件开头 import 段之后。
        pass
    # 简化：在文件末尾 patch 前，import 已由函数内 import 解决；这里只保证 parse_model 能引用。
    # parse_model 通过 globals 里的名字解析模块类，故需在 tasks.py 顶层 import AuxDetect。
    line = "\nfrom ultralytics.nn.modules.yolo26_auxdetect import AuxDetect  # noqa: E402,F401\n"
    # 放在 "from ultralytics.utils.loss import (" 之前，确保在 parse_model 定义前已导入。
    loss_anchor = "from ultralytics.utils.loss import ("
    if loss_anchor in text:
        return text.replace(loss_anchor, line.strip() + "\n" + loss_anchor, 1), True
    return text + line, True


def inject_frozenset(text: str) -> tuple[str, bool]:
    """把 AuxDetect 加进 parse_model 的 end2end 检测头 frozenset（1992-2006）。"""
    marker = "                Detect,\n"  # 16-space indent, only inside the multi-line frozenset
    if "                AuxDetect,\n" in text:
        return text, False
    if marker not in text:
        return text, False
    return text.replace(marker, marker + "                AuxDetect,\n", 1), True


def inject_legacy_set(text: str) -> tuple[str, bool]:
    """把 AuxDetect 加进第 2010 行 `if m in {Detect, ...}` 集合。"""
    marker = "if m in {Detect,"
    if "if m in {AuxDetect, Detect," in text:
        return text, False
    if marker not in text:
        return text, False
    return text.replace(marker, "if m in {AuxDetect, Detect,", 1), True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_auxdetect.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    # 1. module source
    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    # 2. __init__.py export
    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_init_export(it)
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_export_changed", it_changed)

    # 3. tasks.py: backup then import + frozenset + legacy set + init_criterion patch
    backup = tasks_file.with_name(tasks_file.name + ".auxdetect_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_tasks_import(tk)
    tk, frozen_changed = inject_frozenset(tk)
    tk, legacy_changed = inject_legacy_set(tk)
    tk, patch_changed = inject_once(tk, "_aux_init_criterion", PATCH_SRC)
    if imp_changed or frozen_changed or legacy_changed or patch_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_frozenset_changed", frozen_changed)
    print("tasks_legacy_set_changed", legacy_changed)
    print("tasks_init_criterion_patched", patch_changed)

    print("INSTALL_AUXDETECT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
