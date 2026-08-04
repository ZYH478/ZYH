#!/usr/bin/env python
"""iter35：把各向异性条带解耦 neck 块 SADGS 装进 ultralytics 包，供 YAML 引用。

设计动机（从 iter025-034 的完整路况反推）：
- 唯一站住的解耦轴 = 空间感受野/dilation（MSDGS135eq，-8.2% 参数、n=4 精度持平），
  但它只是「持平不掉点」，尚无任何 neck 形态在 n=4 口径上真正超过 gsdown 0.4018。
- 已证伪、不能重走：global 全局支路（各向同性池化，抹平 crazing 高频→净负，两次证伪）、
  HF 高频支路（放大训练噪声、n=4 翻案）、d5 大空洞（HF 存在时网格伪影有害）、
  偏 local 非对称通道、加宽压缩比。
- 混淆矩阵病根：整机误差几乎全是前景/背景可分性（漏检+误报），类间混淆≈0；
  可恢复空间在 rolled-in_scale（氧化皮，带状/大面积方向纹理）和 scratches（细长划痕），
  这两类都有强烈的方向性/各向异性；crazing 是标注天花板，撬不动。

正交新轴 = 方向性/各向异性（neck 侧从未试过）：
- MSDGS 沿 dilation 各向同性解耦。SADGS 保留 d1/d3 两条各向同性分支（守 crazing 及中小尺度），
  把最大空洞 d5 那条各向同性分支换成「条带分支」：depthwise 1xk（横条）+ kx1（竖条）并联相加。
- 关键机理：global 当初有害是因为各向同性（全局平均把 crazing 高频也抹平）；条带卷积只沿单轴
  聚合、正交轴细节完整保留 —— 给 rolled-in/scratches 提供长程方向上下文，却比 global pooling
  少伤 crazing（裂纹网无主方向，H/V 条带只部分模糊而非全抹平）。
- 干净单变量消融：MSDGS135eq=(d1,d3,d5) → SADGS=(d1,d3,strip)，各向同性大尺度→各向异性长程。

参数预算：条带 1xk + kx1 的 depthwise 参数（2*k*c）与 3x3 空洞 depthwise（9*c）在 k=7 时接近，
通道仍均等三分，整体对齐 MSDGS 的 ~1.777M。

与 VoVGSCSP / MSDGS 签名兼容（drop-in 替换 neck 里的 VoVGSCSP）：
    SADGS(c1, c2, n=1, shortcut=True, g=1, e=0.5, dilations=(1,3), strip_k=7, fracs=None)
  其中 dilations 给「各向同性分支」，末尾再自动追加 1 条条带分支；fracs 长度 = len(dilations)+1。

前置依赖：先跑 install_gsconv_modules.py（gsdown head 需 GSConv/VoVGSCSP 才能 build）。

安全：首次运行前备份 tasks.py 为 tasks.py.sadgs_bak；幂等，重复运行不重复注入。
成功打印 INSTALL_SADGS_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated anisotropic strip decoupled neck module (SADGS) for YOLO26.
# Injected by install_sadgs_module.py. Do not edit by hand.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["SADGS"]


class _DWBranch(nn.Module):
    """各向同性瘦分支：depthwise(可空洞) 3x3 -> BN+SiLU，通道全程保持 c_ch。"""

    def __init__(self, c_ch, k=3, dilation=1):
        super().__init__()
        p = dilation * (k - 1) // 2  # same padding for dilated conv
        self.dw = nn.Conv2d(c_ch, c_ch, k, 1, p, dilation=dilation, groups=c_ch, bias=False)
        self.bn = nn.BatchNorm2d(c_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class _StripBranch(nn.Module):
    """各向异性条带分支：depthwise 1xk（横条）+ kx1（竖条）并联相加 -> BN+SiLU。

    只沿单轴长程聚合，正交轴细节完整保留：给带状/细长缺陷（rolled-in/scratches）方向上下文，
    比各向同性 global pooling 少伤无主方向的 crazing 纹理。
    """

    def __init__(self, c_ch, k=7):
        super().__init__()
        ph = k // 2
        # 横条 1xk：沿宽度方向长程；竖条 kx1：沿高度方向长程。均为 depthwise。
        self.dw_h = nn.Conv2d(c_ch, c_ch, (1, k), 1, (0, ph), groups=c_ch, bias=False)
        self.dw_v = nn.Conv2d(c_ch, c_ch, (k, 1), 1, (ph, 0), groups=c_ch, bias=False)
        self.bn = nn.BatchNorm2d(c_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw_h(x) + self.dw_v(x)))


class SADGS(nn.Module):
    """各向异性条带解耦 neck 块，drop-in 替换 neck 里的 VoVGSCSP / MSDGS。

    前 len(dilations) 条为各向同性 depthwise 分支（多尺度），末尾自动追加 1 条条带分支（各向异性）。
    通道按 fracs 权重分配，物理隔离各分支的参数与梯度。总预算对齐原 VoVGSCSP/MSDGS。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 dilations=(1, 3), strip_k=7, fracs=None):
        super().__init__()
        c_ = int(c2 * e)
        dils = [int(d) for d in dilations]
        n_iso = len(dils)
        K = n_iso + 1  # 各向同性分支 + 1 条条带分支
        self.strip_k = int(strip_k)
        if fracs is None:
            fracs = [1] * K
        fracs = [float(f) for f in fracs]
        assert len(fracs) == K, f"fracs {fracs} len != {K} (iso {n_iso} + 1 strip)"
        total = sum(fracs)
        # 按 fracs 权重把 c_ 通道分成 K 份，每份 >=2，余数补到第一份（局部分支）
        splits = [max(2, int(round(c_ * f / total))) for f in fracs]
        diff = c_ - sum(splits)
        splits[0] += diff
        assert all(s > 0 for s in splits) and sum(splits) == c_, f"bad splits {splits} sum!=c_={c_}"
        self.splits = splits
        self.dils = dils
        self.n_iso = n_iso
        self.n = int(n)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        # 每个 stage：n_iso 条各向同性分支 + 1 条条带分支（参数不共享）
        self.stages = nn.ModuleList()
        for _ in range(self.n):
            branches = [_DWBranch(splits[k], k=3, dilation=dils[k]) for k in range(n_iso)]
            branches.append(_StripBranch(splits[n_iso], k=self.strip_k))
            self.stages.append(nn.ModuleList(branches))

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
    """在 base_modules frozenset 内的 VoVGSCSP, 后插入 SADGS。"""
    m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}", text, re.S)
    if not m:
        return text, False
    if "SADGS" in m.group(0):
        return text, False
    new_text = text.replace(
        "            VoVGSCSP,\n",
        "            VoVGSCSP,\n            SADGS,\n",
        1,
    )
    return new_text, new_text != text


def inject_repeat_modules(text: str) -> tuple[str, bool]:
    """在 repeat_modules frozenset 块内追加 SADGS（需要 insert n）。"""
    m = re.search(r"(repeat_modules\s*=\s*frozenset\([^{]*\{)(.*?)(\n\s*\})", text, re.S)
    if not m:
        return text, False
    body = m.group(2)
    if "SADGS" in body:
        return text, False
    new_block = m.group(1) + body.rstrip().rstrip(",") + ",\n            SADGS," + m.group(3)
    new_text = text[: m.start()] + new_block + text[m.end():]
    return new_text, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_sadgs.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_sadgs",
        "\n# Anisotropic strip decoupled neck module\nfrom .yolo26_sadgs import SADGS  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".sadgs_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_sadgs",
        "\n# Anisotropic strip decoupled neck module for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_sadgs import SADGS  # noqa: E402,F401\n",
    )
    tk, base_changed = inject_base_modules(tk)
    tk, repeat_changed = inject_repeat_modules(tk)
    if imp_changed or base_changed or repeat_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)

    print("INSTALL_SADGS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
