#!/usr/bin/env python
"""iter36 方案B：把 BoxHead32Detect 头装进 ultralytics 包，供 YAML 引用。

设计动机（方案 B：MSDGS + BoxHead32 + DFL4）：
- MSDGS 输出给 Detect 的通道是 P3=64/P4=128/P5=256，但 box head 宽度
  c2 = max(16, ch[0]//4, reg_max*4)。ch[0]=64 => ch[0]//4=16；reg_max=4 => reg_max*4=16。
  所以 box head 被压到 16 通道，瓶颈很硬。cls head(cv3) 已是 DWConv 轻量结构，不动。
- BoxHead32Detect：仅把 box 分支(cv2 / one2one_cv2)宽度强制 =32，最后一层输出仍 4*reg_max，
  其余（cls 分支、DFL、end2end 双分配、推理路径）完全继承官方 Detect，零行为改动。
- 与方案 A(纯 reg_max=1→4)构成干净嵌套消融：A=只 DFL4；B=DFL4+加宽 box head。

工程要点（从 tasks.py 真实实现反推）：
- parse_model 对 Detect 子类：args.extend([reg_max, end2end, [ch[x] for x in f]])，
  并设 m.legacy = legacy。所以 BoxHead32Detect 必须：
  (a) import 进 tasks.py；
  (b) 加进那个大 frozenset（触发 reg_max/end2end/ch 注入 + legacy 赋值）。
- __init__ 签名必须与 Detect 完全一致 (nc, reg_max, end2end, ch)。
- guess_model_task：类名含 "Detect" => 可被 task 推断为 detect（训练脚本仍显式 task="detect" 兜底）。

安全：首次运行前备份 tasks.py 为 tasks.py.boxhead32_bak；幂等，重复运行不重复注入。
成功打印 INSTALL_BOXHEAD32_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated wide-box-head Detect (BoxHead32Detect) for YOLO26.
# Injected by install_boxhead32_module.py. Do not edit by hand.
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

__all__ = ["BoxHead32Detect"]

BOX_HEAD_WIDTH = 32


class BoxHead32Detect(Detect):
    """Detect 变体：把 box 回归分支(cv2 / one2one_cv2)宽度强制 =BOX_HEAD_WIDTH(32)。

    只加宽 box head，cls head(cv3)、DFL、end2end 双分配、推理/loss 路径全部继承官方 Detect。
    目标：解开 MSDGS(ch[0]=64 => box head 被压到 16)的 box 回归瓶颈，提升 mAP50-95。
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        # 先用官方逻辑建好一切（cv2/cv3/dfl/one2one_*），再重建 box 分支为宽版。
        super().__init__(nc, reg_max, end2end, ch)
        c2 = max(BOX_HEAD_WIDTH, 4 * self.reg_max)  # 宽 box head，且不低于输出通道
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        if end2end:
            import copy
            self.one2one_cv2 = copy.deepcopy(self.cv2)
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_detect_frozenset(text: str) -> tuple[str, bool]:
    """把 BoxHead32Detect 加进那个包含 Detect/FBHead/UBHead 的大 frozenset。

    该 frozenset 触发 args.extend([reg_max, end2end, ch]) + m.legacy 赋值。
    定位锚点：'    {\\n                Detect,\\n' 之后插入。
    幂等检查用「插入后的精确串」，不能用全文含 "BoxHead32Detect"（import 行也含它，会误判）。
    """
    # 在 frozenset({ 里的第一个 "Detect,\n" 后插入（那是 head frozenset，不是 base_modules）
    m = re.search(r"(elif m in frozenset\(\s*\{\s*\n\s*Detect,\n)", text)
    if not m:
        return text, False
    anchor = m.group(1)
    inserted = anchor + "                BoxHead32Detect,\n"
    if inserted in text:  # 已插入过（精确串），幂等返回
        return text, False
    new_text = text.replace(anchor, inserted, 1)
    return new_text, new_text != text


def inject_legacy_set(text: str) -> tuple[str, bool]:
    """把 BoxHead32Detect 加进那行 `if m in {FBHead, UBHead, ... }: m.legacy = legacy`。"""
    if "BoxHead32Detect" in text.split("m.legacy = legacy")[0]:
        # 已在 legacy set 前出现
        pass
    m = re.search(r"(if m in \{FBHead, UBHead, AuxDetect, Detect,)", text)
    if not m:
        return text, False
    anchor = m.group(1)
    if "BoxHead32Detect" in anchor:
        return text, False
    new = "if m in {BoxHead32Detect, FBHead, UBHead, AuxDetect, Detect,"
    new_text = text.replace(anchor, new, 1)
    return new_text, new_text != text


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_boxhead32.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_boxhead32",
        "\n# Wide-box-head Detect variant\nfrom .yolo26_boxhead32 import BoxHead32Detect  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".boxhead32_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_boxhead32",
        "\n# Wide-box-head Detect variant for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_boxhead32 import BoxHead32Detect  # noqa: E402,F401\n",
    )
    tk, fz_changed = inject_detect_frozenset(tk)
    tk, lg_changed = inject_legacy_set(tk)
    if imp_changed or fz_changed or lg_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_frozenset_changed", fz_changed)
    print("tasks_legacy_set_changed", lg_changed)

    print("INSTALL_BOXHEAD32_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
