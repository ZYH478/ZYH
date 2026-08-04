#!/usr/bin/env python
"""把双分支解耦 neck 块 DualBranchGS 装进当前 ultralytics 包，供 YAML 引用。

设计动机（本 goal iter24 实测挖出的真病根）：
- gsdown 上任何全局改动都是"弱类升、强类降"的劫富济贫、净零（iter17 NWD / iter24 加宽
  neck 反复坐实）。根因不是容量不足，而是强类(inclusion/patches/scratches，局部结构)与
  弱类(crazing/rolled-in，弥漫纹理)共用同一套 neck 通道，训练时梯度互相干扰。
- 对症设计：把 neck 块拆成两条各自很瘦的并行分支（总预算 ≈ 原 VoVGSCSP，不增肥），
  物理隔离两类特征的参数与梯度：
    * 局部分支 (local)  : 小感受野 depthwise 3x3     -> 局部结构缺陷
    * 全局分支 (global) : 大感受野 空洞 depthwise 3x3 (dilation=d) -> 弥漫纹理缺陷
  末端 concat(local, global, identity) -> 1x1 投影回 c2。

DualBranchGS 与 VoVGSCSP 签名兼容（drop-in 替换 neck 里的 VoVGSCSP）：
    DualBranchGS(c1, c2, n=1, shortcut=True, g=1, e=0.5, d=3)

登记规则（与 install_gsconv_modules.py 同款）：
- DualBranchGS 同时进 base_modules 与 repeat_modules（替换 VoVGSCSP，需 insert n）。

前置依赖：先跑 install_gsconv_modules.py（本模块复用 GSConv 做压缩/投影，保持与
gsdown 同族的参数量级）。

安全：
- 首次运行前把 tasks.py 备份为 tasks.py.dualbranch_bak（存在则不覆盖）。
- 幂等：重复运行不会重复注入。

用法（远程）：
    python install_gsconv_modules.py         # 先装 GSConv/VoVGSCSP
    python install_dualbranch_module.py      # 再装 DualBranchGS
成功打印 INSTALL_DUALBRANCH_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated dual-branch decoupled neck module (DualBranchGS) for YOLO26.
# Injected by install_dualbranch_module.py. Do not edit by hand.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["DualBranchGS"]


class _DWBranch(nn.Module):
    """一条瘦分支：pointwise 压缩 -> depthwise(可空洞) -> BN+SiLU。

    dilation=1 => 小感受野（局部结构）；dilation=d>1 => 大感受野（弥漫纹理）。
    通道全程保持 c_（分支内不扩张，保证总预算 ~ 原 VoVGSCSP）。
    """

    def __init__(self, c_, k=3, dilation=1):
        super().__init__()
        p = dilation * (k - 1) // 2  # same padding for dilated conv
        self.dw = nn.Conv2d(c_, c_, k, 1, p, dilation=dilation, groups=c_, bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class DualBranchGS(nn.Module):
    """双分支解耦 neck 块，drop-in 替换 neck 里的 VoVGSCSP。

    数据流（总预算对齐原 VoVGSCSP：cv1 压到 c_=e*c2，cv2 从 2*c_ 投影回 c2）：
        x  -> cv1 (1x1, c1->c_)  = x1
        x1 -> chunk 成两半 (各 c_/2)，物理隔离：
              local  : _DWBranch(dilation=1)  作用于前半
              global : _DWBranch(dilation=d)  作用于后半
        cat(local, global) = feat (c_)   # 两分支参数不共享，梯度互不干扰
        cv2( cat(feat, x1) )  (1x1, 2*c_->c2)   # CSP 残差整合
    n>1 时堆叠多个双分支单元（作用在 c_ 上，深度维扩展）。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, d=3):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 2 == 0, f"c_={c_} must be even for symmetric split"
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.n = int(n)
        ch = c_ // 2
        # 每个 stage 一对 (local, global) 分支，参数独立
        self.locals = nn.ModuleList(_DWBranch(ch, k=3, dilation=1) for _ in range(self.n))
        self.globals = nn.ModuleList(_DWBranch(ch, k=3, dilation=int(d)) for _ in range(self.n))

    def forward(self, x):
        x1 = self.cv1(x)
        feat = x1
        for i in range(self.n):
            a, b = feat.chunk(2, dim=1)          # 物理隔离：前半->local，后半->global
            a = a + self.locals[i](a)            # 分支内残差，稳定训练
            b = b + self.globals[i](b)
            feat = torch.cat((a, b), dim=1)
        return self.cv2(torch.cat((feat, x1), dim=1))
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    """在 base_modules frozenset 内的 VoVGSCSP, 后插入 DualBranchGS。"""
    m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}", text, re.S)
    if not m:
        return text, False
    if "DualBranchGS" in m.group(0):
        return text, False
    new_text = text.replace(
        "            VoVGSCSP,\n",
        "            VoVGSCSP,\n            DualBranchGS,\n",
        1,
    )
    return new_text, new_text != text


def inject_repeat_modules(text: str) -> tuple[str, bool]:
    """在 repeat_modules frozenset 块内追加 DualBranchGS（需要 insert n）。"""
    m = re.search(r"(repeat_modules\s*=\s*frozenset\([^{]*\{)(.*?)(\n\s*\})", text, re.S)
    if not m:
        return text, False
    body = m.group(2)
    if "DualBranchGS" in body:
        return text, False
    new_block = m.group(1) + body.rstrip().rstrip(",") + ",\n            DualBranchGS," + m.group(3)
    new_text = text[: m.start()] + new_block + text[m.end():]
    return new_text, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_dualbranch.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    # 1. module source
    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    # 2. __init__.py export
    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_dualbranch",
        "\n# Dual-branch decoupled neck module\nfrom .yolo26_dualbranch import DualBranchGS  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    # 3. tasks.py: backup then import + frozenset
    backup = tasks_file.with_name(tasks_file.name + ".dualbranch_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_dualbranch",
        "\n# Dual-branch decoupled neck module for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_dualbranch import DualBranchGS  # noqa: E402,F401\n",
    )
    tk, base_changed = inject_base_modules(tk)
    tk, repeat_changed = inject_repeat_modules(tk)
    if imp_changed or base_changed or repeat_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)

    print("INSTALL_DUALBRANCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
