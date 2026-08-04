#!/usr/bin/env python
"""把多尺度解耦 neck 块 MSDGS 装进当前 ultralytics 包，供 YAML 引用。

设计动机（iter025 暴露的两处粗暴，本轮精修）：
- iter025 双分支解耦 DualBranchGS 首次打破"弱升强必降"铁律（强类真涨：d3 inclusion +2.9pp、
  d5 patches +2.3pp），证明"劫富济贫源于通道共享、物理隔离方向正确"，但没同时守住所有类：
  不同 dilation 各守不同类（d3 守 inclusion，d5 守 patches/rolled-in），说明"感受野-类别"有
  结构性对应，单分支单尺度吃不下全部。
- 两处粗暴：(a) 两分支通道二等分太武断；(b) 单一 dilation 覆盖不了多尺度缺陷。

MSDGS = DualBranchGS 的推广：**K 个 depthwise 分支并联，dilation 各异（多尺度），
通道按 fracs 权重非对称分配**。DualBranchGS 是其特例 (dilations=(1,d), fracs=(1,1))。

    x  -> cv1 (1x1, c1->c_=e*c2) = x1
    x1 -> split 成 K 份 (通道按 fracs 权重分)，物理隔离：
          branch_k: depthwise 3x3 (dilation=dilations[k]) 作用于第 k 份，分支内残差
    cat(所有分支) = feat (c_)
    cv2( cat(feat, x1) ) (1x1, 2*c_->c2)   # 与 DualBranchGS 同款 CSP 残差整合

与 VoVGSCSP 签名兼容（drop-in 替换 neck 里的 VoVGSCSP）：
    MSDGS(c1, c2, n=1, shortcut=True, g=1, e=0.5, dilations=(1,3,5), fracs=None)

前置依赖：先跑 install_gsconv_modules.py（gsdown head 需 GSConv/VoVGSCSP 才能 build）。

安全：首次运行前备份 tasks.py 为 tasks.py.msdgs_bak；幂等，重复运行不重复注入。

用法（远程）：
    python install_gsconv_modules.py
    python install_msdgs_module.py
成功打印 INSTALL_MSDGS_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated multi-scale decoupled neck module (MSDGS) for YOLO26.
# Injected by install_msdgs_module.py. Do not edit by hand.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["MSDGS"]


class _DWBranch(nn.Module):
    """一条瘦分支：depthwise(可空洞) 3x3 -> BN+SiLU，通道全程保持 c_ch。

    dilation=1 => 局部结构；dilation=d>1 => 多尺度弥漫纹理。
    """

    def __init__(self, c_ch, k=3, dilation=1):
        super().__init__()
        p = dilation * (k - 1) // 2  # same padding for dilated conv
        self.dw = nn.Conv2d(c_ch, c_ch, k, 1, p, dilation=dilation, groups=c_ch, bias=False)
        self.bn = nn.BatchNorm2d(c_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class MSDGS(nn.Module):
    """多尺度解耦 neck 块，drop-in 替换 neck 里的 VoVGSCSP。

    K 个 depthwise 分支并联（dilation 各异 = 多尺度），通道按 fracs 权重非对称分配，
    物理隔离各尺度/各类别特征的参数与梯度。总预算对齐原 VoVGSCSP。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, dilations=(1, 3, 5), fracs=None):
        super().__init__()
        c_ = int(c2 * e)
        dils = [int(d) for d in dilations]
        K = len(dils)
        if fracs is None:
            fracs = [1] * K
        fracs = [float(f) for f in fracs]
        assert len(fracs) == K, f"fracs {fracs} len != dilations {dils} len"
        total = sum(fracs)
        # 按 fracs 权重把 c_ 通道分成 K 份，每份 >=2，余数补到第一份（局部分支）
        splits = [max(2, int(round(c_ * f / total))) for f in fracs]
        diff = c_ - sum(splits)
        splits[0] += diff
        assert all(s > 0 for s in splits) and sum(splits) == c_, f"bad splits {splits} sum!=c_={c_}"
        self.splits = splits
        self.dils = dils
        self.n = int(n)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        # 每个 stage 有 K 个独立分支（参数不共享）
        self.stages = nn.ModuleList()
        for _ in range(self.n):
            stage = nn.ModuleList(_DWBranch(splits[k], k=3, dilation=dils[k]) for k in range(K))
            self.stages.append(stage)

    def forward(self, x):
        x1 = self.cv1(x)
        feat = x1
        for stage in self.stages:
            parts = torch.split(feat, self.splits, dim=1)  # 物理隔离：第 k 份 -> 第 k 分支
            outs = [parts[k] + stage[k](parts[k]) for k in range(len(parts))]  # 分支内残差
            feat = torch.cat(outs, dim=1)
        return self.cv2(torch.cat((feat, x1), dim=1))
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    """在 base_modules frozenset 内的 VoVGSCSP, 后插入 MSDGS。"""
    m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}", text, re.S)
    if not m:
        return text, False
    if "MSDGS" in m.group(0):
        return text, False
    new_text = text.replace(
        "            VoVGSCSP,\n",
        "            VoVGSCSP,\n            MSDGS,\n",
        1,
    )
    return new_text, new_text != text


def inject_repeat_modules(text: str) -> tuple[str, bool]:
    """在 repeat_modules frozenset 块内追加 MSDGS（需要 insert n）。"""
    m = re.search(r"(repeat_modules\s*=\s*frozenset\([^{]*\{)(.*?)(\n\s*\})", text, re.S)
    if not m:
        return text, False
    body = m.group(2)
    if "MSDGS" in body:
        return text, False
    new_block = m.group(1) + body.rstrip().rstrip(",") + ",\n            MSDGS," + m.group(3)
    new_text = text[: m.start()] + new_block + text[m.end():]
    return new_text, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_msdgs.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_msdgs",
        "\n# Multi-scale decoupled neck module\nfrom .yolo26_msdgs import MSDGS  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_msdgs",
        "\n# Multi-scale decoupled neck module for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_msdgs import MSDGS  # noqa: E402,F401\n",
    )
    tk, base_changed = inject_base_modules(tk)
    tk, repeat_changed = inject_repeat_modules(tk)
    if imp_changed or base_changed or repeat_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)

    print("INSTALL_MSDGS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
