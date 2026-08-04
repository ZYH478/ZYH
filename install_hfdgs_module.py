#!/usr/bin/env python
"""把「异质频率解耦 neck 块」HFDGS 装进当前 ultralytics 包，供 YAML 引用。

设计动机（iter027 证伪 iter026「两弱类同类」前提，本轮 C 方向按新机理精修）：
- iter026 多尺度解耦 MSDGS 把强类全守住（patches 0.589/scratches 0.520/pitted 0.469），
  最好 135eq test 0.3988，瓶颈全压在两弱类 crazing(0.177)/rolled-in(0.258)。
- iter027 给弱类补全局上下文（GCDGS）反而全线变差，但逐类真值证伪了「两弱类同类」前提：
  全局加倍(13g_heavy)时 rolled-in 升到 0.296（全场最高）、crazing 跌到 0.133（全场最低）——
  **两弱类机理相反**：rolled-in（氧化皮，大面积周期统计）吃全局上下文；
  crazing（裂纹网，细密局部高频边缘）被全局池化抹平，越全局越糟。
- 真正卡整体的一直是 crazing（0.13-0.18，谁都救不动），它要的是**更精细的高频/局部分辨率**。

HFDGS = MSDGS（多尺度 local，强类底座）+ 一条**高频增强支路**（专治 crazing）+ 可选全局支路
（治 rolled-in）。异质分工，物理解耦：

    x  -> cv1 (1x1, c1->c_=e*c2) = x1
    x1 -> split 成 (Kd 条 local + [HF] + [global])，通道按权重分，物理隔离：
          local_k : depthwise 3x3 (dilation=dilations[k]) + 分支内残差   -> 局部结构缺陷（强类）
          HF      : x - avgpool_k(x) 取高频残差 -> dw3x3 学高频结构 -> 加回  -> crazing 细密边缘
          global  : 注意力池化 -> bottleneck -> 广播加回                  -> rolled-in 大面积统计
    cat(所有分支) = feat (c_)
    cv2( cat(feat, x1) ) (1x1, 2*c_->c2)   # 与 MSDGS 同款 CSP 残差整合

与 VoVGSCSP 签名兼容（drop-in 替换 neck 里的 VoVGSCSP）：
    HFDGS(c1, c2, n=1, shortcut=True, g=1, e=0.5,
          dilations=(1,3,5), hf_frac=1.0, global_frac=0.0, pool_k=3, r=4)

前置依赖：先跑 install_gsconv_modules.py（gsdown head 需 GSConv/VoVGSCSP 才能 build）。

安全：首次运行前备份 tasks.py 为 tasks.py.hfdgs_bak；幂等，重复运行不重复注入。

用法（远程）：
    python install_gsconv_modules.py
    python install_hfdgs_module.py
成功打印 INSTALL_HFDGS_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated heterogeneous-frequency decoupled neck module (HFDGS) for YOLO26.
# Injected by install_hfdgs_module.py. Do not edit by hand.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["HFDGS"]


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


class _HFBranch(nn.Module):
    """高频增强分支，专治 crazing（裂纹网，细密局部高频边缘）。

    x - avgpool_k(x) = 高频残差（high-pass / 类 unsharp mask），保留细密边缘；
    再 dw3x3 学高频结构，加回。全分辨率无空洞、不抹平细节——与全局池化方向相反，
    正对 iter027 发现「crazing 被全局池化抹平、越全局越糟」的机理。参数极省。
    """

    def __init__(self, c_ch, pool_k=3):
        super().__init__()
        # count_include_pad=False：padding 位置只对有效元素求平均，常数输入处处 hf=0，
        # 避免边界补零凭空激发高频响应（否则图像边缘会有伪高频）。
        self.pool = nn.AvgPool2d(pool_k, stride=1, padding=pool_k // 2, count_include_pad=False)
        self.dw = nn.Conv2d(c_ch, c_ch, 3, 1, 1, groups=c_ch, bias=False)
        self.bn = nn.BatchNorm2d(c_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        hf = x - self.pool(x)                    # 高频成分（high-pass）
        return x + self.act(self.bn(self.dw(hf)))  # 分支内残差


class _GCBranch(nn.Module):
    """全局上下文分支（GCNet 式），治 rolled-in（氧化皮，大面积周期统计）。

    注意力池化把整张特征图聚成全局上下文向量 -> 轻量 bottleneck transform
    -> 广播加回每个空间位置。iter027 已验证：全局加倍时 rolled-in 升到 0.296（全场最高）。
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


class HFDGS(nn.Module):
    """异质频率解耦 neck 块，drop-in 替换 neck 里的 VoVGSCSP。

    Kd 条 depthwise 多尺度局部分支（强类底座）+ 可选高频支路（治 crazing）
    + 可选全局上下文支路（治 rolled-in），通道按权重非对称分配，物理隔离各支路
    参数与梯度。总预算对齐原 VoVGSCSP。异质分工：不同缺陷类走机理匹配的支路。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
                 dilations=(1, 3, 5), hf_frac=1.0, global_frac=0.0, pool_k=3, r=4):
        super().__init__()
        c_ = int(c2 * e)
        dils = [int(d) for d in dilations]
        self.Kd = len(dils)
        self.has_hf = float(hf_frac) > 0
        self.has_gc = float(global_frac) > 0
        # 权重：Kd 条 local 各 1，HF 分支 hf_frac，global 分支 global_frac
        weights = [1.0] * self.Kd
        if self.has_hf:
            weights.append(float(hf_frac))
        if self.has_gc:
            weights.append(float(global_frac))
        total = sum(weights)
        # 按权重把 c_ 通道分成若干份，每份 >=2，余数补到第一份（局部分支）
        splits = [max(2, int(round(c_ * wt / total))) for wt in weights]
        diff = c_ - sum(splits)
        splits[0] += diff
        assert all(s > 0 for s in splits) and sum(splits) == c_, f"bad splits {splits} sum!=c_={c_}"
        self.splits = splits
        self.dils = dils
        self.n = int(n)
        self.pool_k = int(pool_k)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        # 每个 stage：Kd 条 local + [HF] + [global]（参数不共享）
        self.stages = nn.ModuleList()
        for _ in range(self.n):
            branches = nn.ModuleList(_DWBranch(splits[k], k=3, dilation=dils[k]) for k in range(self.Kd))
            idx = self.Kd
            if self.has_hf:
                branches.append(_HFBranch(splits[idx], pool_k=self.pool_k))
                idx += 1
            if self.has_gc:
                branches.append(_GCBranch(splits[idx], r=r))
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
                    outs.append(stage[k](parts[k]))             # HF/global 分支：内部已含残差
            feat = torch.cat(outs, dim=1)
        return self.cv2(torch.cat((feat, x1), dim=1))
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    """在 base_modules frozenset 内的 VoVGSCSP, 后插入 HFDGS。"""
    m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}", text, re.S)
    if not m:
        return text, False
    if "HFDGS" in m.group(0):
        return text, False
    new_text = text.replace(
        "            VoVGSCSP,\n",
        "            VoVGSCSP,\n            HFDGS,\n",
        1,
    )
    return new_text, new_text != text


def inject_repeat_modules(text: str) -> tuple[str, bool]:
    """在 repeat_modules frozenset 块内追加 HFDGS（需要 insert n）。"""
    m = re.search(r"(repeat_modules\s*=\s*frozenset\([^{]*\{)(.*?)(\n\s*\})", text, re.S)
    if not m:
        return text, False
    body = m.group(2)
    if "HFDGS" in body:
        return text, False
    new_block = m.group(1) + body.rstrip().rstrip(",") + ",\n            HFDGS," + m.group(3)
    new_text = text[: m.start()] + new_block + text[m.end():]
    return new_text, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_hfdgs.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_hfdgs",
        "\n# Heterogeneous-frequency decoupled neck module\nfrom .yolo26_hfdgs import HFDGS  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".hfdgs_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_hfdgs",
        "\n# Heterogeneous-frequency decoupled neck module for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_hfdgs import HFDGS  # noqa: E402,F401\n",
    )
    tk, base_changed = inject_base_modules(tk)
    tk, repeat_changed = inject_repeat_modules(tk)
    if imp_changed or base_changed or repeat_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)

    print("INSTALL_HFDGS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
