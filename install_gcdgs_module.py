#!/usr/bin/env python
"""把「全局上下文解耦 neck 块」GCDGS 装进当前 ultralytics 包，供 YAML 引用。

设计动机（iter026 锁定的靶点，本轮 A 方向精修）：
- iter026 多尺度解耦 MSDGS 根治了 iter025 的"顾此失彼"——三强类首次同时守住
  （patches 0.589 / scratches 0.520 / pitted 0.469），证明"物理隔离多尺度分支"框架对。
- 但三候选全 < gsdown 0.4018（最好 135eq 0.3988，仅差 0.003），瓶颈锁定在两弱类
  （crazing 0.177 / rolled-in 0.258）。**关键负结果**：非对称偏 local（135local/13local）
  反而更差，证明弱类要的不是 local 容量。
- 机理：crazing（裂纹网）/rolled-in（氧化皮）是**弥漫纹理缺陷**，靠全图统计而非局部形状判别。
  depthwise 空洞卷积即便 d=5，有效感受野仍有限、且有网格伪影，喂不饱这类"全局统计型"纹理。

GCDGS = MSDGS + 一条**全局上下文分支**（GCNet 式：注意力池化聚全局上下文 -> 轻量 transform
-> 广播加回），让弥漫纹理支路每个空间位置都拿到图像级上下文。解耦（参数/梯度隔离）与轻量保持不变：

    x  -> cv1 (1x1, c1->c_=e*c2) = x1
    x1 -> split 成 (Kd 条 local + 1 条 global)，通道按权重分，物理隔离：
          local_k : depthwise 3x3 (dilation=dilations[k]) + 分支内残差   -> 局部结构缺陷
          global  : 注意力池化 -> 1x1 降维 -> LN -> SiLU -> 1x1 -> 广播加回 -> 弥漫纹理缺陷
    cat(所有分支) = feat (c_)
    cv2( cat(feat, x1) ) (1x1, 2*c_->c2)   # 与 MSDGS 同款 CSP 残差整合

与 VoVGSCSP 签名兼容（drop-in 替换 neck 里的 VoVGSCSP）：
    GCDGS(c1, c2, n=1, shortcut=True, g=1, e=0.5, dilations=(1,3), global_frac=1.0, r=4)

前置依赖：先跑 install_gsconv_modules.py（gsdown head 需 GSConv/VoVGSCSP 才能 build）。

安全：首次运行前备份 tasks.py 为 tasks.py.gcdgs_bak；幂等，重复运行不重复注入。

用法（远程）：
    python install_gsconv_modules.py
    python install_gcdgs_module.py
成功打印 INSTALL_GCDGS_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated global-context decoupled neck module (GCDGS) for YOLO26.
# Injected by install_gcdgs_module.py. Do not edit by hand.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["GCDGS"]


class _DWBranch(nn.Module):
    """一条瘦局部分支：depthwise(可空洞) 3x3 -> BN+SiLU，通道全程保持 c_ch。

    dilation=1 => 局部结构；dilation=d>1 => 多尺度局部纹理。与 MSDGS 同款。
    """

    def __init__(self, c_ch, k=3, dilation=1):
        super().__init__()
        p = dilation * (k - 1) // 2  # same padding for dilated conv
        self.dw = nn.Conv2d(c_ch, c_ch, k, 1, p, dilation=dilation, groups=c_ch, bias=False)
        self.bn = nn.BatchNorm2d(c_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class _GCBranch(nn.Module):
    """全局上下文分支（GCNet 式），专治弥漫纹理类（crazing / rolled-in）。

    注意力池化把整张特征图聚成一个全局上下文向量 -> 轻量 bottleneck transform
    -> 广播加回每个空间位置。让每个像素都拿到图像级统计，弥补 depthwise 空洞卷积
    有效感受野不足、且不引网格伪影。参数极省（1x1 卷积 + 全局池化）。
    """

    def __init__(self, c_ch, r=4):
        super().__init__()
        hidden = max(4, c_ch // r)
        self.att = nn.Conv2d(c_ch, 1, 1)  # 空间注意力权重 -> softmax 池化
        self.transform = nn.Sequential(
            nn.Conv2d(c_ch, hidden, 1),
            nn.LayerNorm([hidden, 1, 1]),
            nn.SiLU(),
            nn.Conv2d(hidden, c_ch, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        att = self.att(x).view(b, 1, h * w)          # [b,1,hw]
        att = torch.softmax(att, dim=-1)
        feat = x.view(b, c, h * w)                    # [b,c,hw]
        ctx = torch.bmm(feat, att.transpose(1, 2)).view(b, c, 1, 1)  # [b,c,1,1] 全局上下文
        return x + self.transform(ctx)               # 广播加回（分支内残差）


class GCDGS(nn.Module):
    """全局上下文解耦 neck 块，drop-in 替换 neck 里的 VoVGSCSP。

    Kd 条 depthwise 多尺度局部分支（dilation 各异）+ 1 条全局上下文分支并联，
    通道按权重非对称分配，物理隔离各支路参数与梯度。总预算对齐原 VoVGSCSP。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 dilations=(1, 3), global_frac=1.0, r=4):
        super().__init__()
        c_ = int(c2 * e)
        dils = [int(d) for d in dilations]
        self.Kd = len(dils)
        # 权重：Kd 条 local 各 1，global 分支 global_frac
        weights = [1.0] * self.Kd + [float(global_frac)]
        total = sum(weights)
        # 按权重把 c_ 通道分成 Kd+1 份，每份 >=2，余数补到第一份（局部分支）
        splits = [max(2, int(round(c_ * wt / total))) for wt in weights]
        diff = c_ - sum(splits)
        splits[0] += diff
        assert all(s > 0 for s in splits) and sum(splits) == c_, f"bad splits {splits} sum!=c_={c_}"
        self.splits = splits
        self.dils = dils
        self.n = int(n)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        # 每个 stage 有 Kd 条 local + 1 条 global（参数不共享）
        self.stages = nn.ModuleList()
        for _ in range(self.n):
            branches = nn.ModuleList(_DWBranch(splits[k], k=3, dilation=dils[k]) for k in range(self.Kd))
            branches.append(_GCBranch(splits[self.Kd], r=r))  # 末份给全局分支
            self.stages.append(branches)

    def forward(self, x):
        x1 = self.cv1(x)
        feat = x1
        for stage in self.stages:
            parts = torch.split(feat, self.splits, dim=1)  # 物理隔离：第 k 份 -> 第 k 支路
            outs = []
            for k in range(len(parts)):
                if k < self.Kd:
                    outs.append(parts[k] + stage[k](parts[k]))  # local 分支：外部残差
                else:
                    outs.append(stage[k](parts[k]))             # global 分支：内部已含残差
            feat = torch.cat(outs, dim=1)
        return self.cv2(torch.cat((feat, x1), dim=1))
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    """在 base_modules frozenset 内的 VoVGSCSP, 后插入 GCDGS。"""
    m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}", text, re.S)
    if not m:
        return text, False
    if "GCDGS" in m.group(0):
        return text, False
    new_text = text.replace(
        "            VoVGSCSP,\n",
        "            VoVGSCSP,\n            GCDGS,\n",
        1,
    )
    return new_text, new_text != text


def inject_repeat_modules(text: str) -> tuple[str, bool]:
    """在 repeat_modules frozenset 块内追加 GCDGS（需要 insert n）。"""
    m = re.search(r"(repeat_modules\s*=\s*frozenset\([^{]*\{)(.*?)(\n\s*\})", text, re.S)
    if not m:
        return text, False
    body = m.group(2)
    if "GCDGS" in body:
        return text, False
    new_block = m.group(1) + body.rstrip().rstrip(",") + ",\n            GCDGS," + m.group(3)
    new_text = text[: m.start()] + new_block + text[m.end():]
    return new_text, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_gcdgs.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_gcdgs",
        "\n# Global-context decoupled neck module\nfrom .yolo26_gcdgs import GCDGS  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".gcdgs_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_gcdgs",
        "\n# Global-context decoupled neck module for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_gcdgs import GCDGS  # noqa: E402,F401\n",
    )
    tk, base_changed = inject_base_modules(tk)
    tk, repeat_changed = inject_repeat_modules(tk)
    if imp_changed or base_changed or repeat_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)

    print("INSTALL_GCDGS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
