#!/usr/bin/env python
"""iter19：给 ultralytics 8.4.93 注入 backbone 特征提取增强模块（纯 PyTorch，免编译）。

设计对齐 install_yolo26_exp_modules.py（SPDConv/DySample 同款注入范式）：
- 写模块文件 yolo26_backbone.py 到 nn/modules/
- 注入 __init__.py 导出 + tasks.py import
- 改通道的模块（DCNv2Conv 做 stride-2 下采样、PKIC3k2/DWRC3k2 作 C3k2 替换）→ 进 base_modules
  （ultralytics 会自动传 (c1, c2, ...)，与 SPDConv/C3k2 一致）

三个模块契合本 goal 唯一有效规律「几何自适应/信息保真类有效、重加权类无效」：

1. DCNv2Conv（可变形卷积 v2，基于 torchvision.ops.deform_conv2d，官方算子免编译）：
   卷积采样点位置可学习偏移 + 调制权重，感受野贴合目标几何形状而非死板方形网格。
   对 scratches（细长）/inclusion（不规则形变）对症。作 stride-2 下采样替换（改通道）。
   偏移场需数据学习——1200 张小数据集上是灰色地带，只放深层（语义强、偏移易学）。

2. PKIC3k2（Poly Kernel Inception，参考 PKINet CVPR2024）：
   C3k2 的 bottleneck 内部并联多个 depthwise 大核（3/5/7/9），Inception 式多尺度感受野，
   无注意力/无重加权（纯并联卷积相加）。对钢铁缺陷天然多尺度（pitted 小点 + patches 大块共存）
   对症。放浅中层（尺度混杂最重）。改通道进 base_modules（与 C3k2 同）。

3. DWRC3k2（Dilation-Wise Residual，参考 DWRSeg）：
   C3k2 的 bottleneck 内部用多路空洞卷积（dilation 1/3/5）聚合多尺度上下文，两步式区域残差。
   depthwise+空洞极轻。对 crazing/rolled-in_scale（低对比度、大范围纹理，需更大上下文）对症。
   放深层（大感受野收益最大）。改通道进 base_modules。

全部从官方 yolo26n.pt 迁移，保持 end2end/reg_max=1。

远程用法：
    python install_backbone_modules.py    # 幂等
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


MODULE_SRC = '''# Auto-generated backbone feature-extraction modules for YOLO26 (NEU-DET iter19).
# Injected by install_backbone_modules.py. Do not edit by hand.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2, Bottleneck


class DCNv2Conv(nn.Module):
    """Deformable Conv v2 as a stride-2 downsampling replacement (info-preserving, geometry-adaptive).

    Uses torchvision.ops.deform_conv2d (official op, no CUDA compilation needed).
    A side branch predicts per-location sampling offsets (2*K) and modulation masks (K);
    the main 3x3 deformable conv then samples input at geometry-adaptive positions.
    Targets elongated / irregular defects (scratches, inclusion). YAML: [-1, 1, DCNv2Conv, [c2, 3, 2]].
    """

    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        super().__init__()
        from torchvision.ops import DeformConv2d
        self.k = int(k)
        self.s = int(s)
        self.pad = (self.k // 2) if p is None else int(p)
        # offset (2*k*k) + modulation mask (k*k) predictor, standard conv over input.
        self.offset_mask = nn.Conv2d(c1, 3 * self.k * self.k, self.k, stride=self.s, padding=self.pad)
        nn.init.constant_(self.offset_mask.weight, 0.0)
        nn.init.constant_(self.offset_mask.bias, 0.0)
        self.dcn = DeformConv2d(c1, c2, self.k, stride=self.s, padding=self.pad, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        om = self.offset_mask(x)
        o1, o2, mask = torch.chunk(om, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)
        y = self.dcn(x, offset, mask)
        return self.act(self.bn(y))


class _PKIConv(nn.Module):
    """Poly-Kernel Inception depthwise multi-branch conv (c-preserving inner op, no attention)."""

    def __init__(self, c, kernels=(3, 5, 7, 9)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False) for k in kernels]
        )
        # 1x1 fuse the summed multi-scale response back to c.
        self.fuse = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        y = 0
        for b in self.branches:
            y = y + b(x)
        return self.act(self.bn(self.fuse(y)))


class _PKIBottleneck(nn.Module):
    """Bottleneck whose 3x3 is replaced by poly-kernel Inception multi-scale conv."""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.pki = _PKIConv(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.pki(self.cv1(x)))
        return x + y if self.add else y


class PKIC3k2(C3k2):
    """C3k2 with poly-kernel Inception bottlenecks (multi-scale receptive field, no attention).

    Drop-in for a C3k2 in the backbone. Keeps CSP split/concat structure, only swaps
    inner Bottleneck for multi-scale _PKIBottleneck. YAML: [-1, n, PKIC3k2, [c2, shortcut]].
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(_PKIBottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))


class _DWRConv(nn.Module):
    """Dilation-wise residual multi-scale context (c-preserving inner op, no attention)."""

    def __init__(self, c, dilations=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(c, c, 3, padding=d, dilation=d, groups=c, bias=False) for d in dilations]
        )
        self.fuse = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        y = 0
        for b in self.branches:
            y = y + b(x)
        return self.act(self.bn(self.fuse(y)))


class _DWRBottleneck(nn.Module):
    """Bottleneck whose 3x3 is replaced by dilation-wise residual multi-scale conv."""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.dwr = _DWRConv(c_)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.dwr(self.cv1(x)))
        return x + y if self.add else y


class DWRC3k2(C3k2):
    """C3k2 with dilation-wise residual bottlenecks (large multi-scale context, no attention).

    Drop-in for a deep-stage C3k2. YAML: [-1, n, DWRC3k2, [c2, shortcut]].
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(_DWRBottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))


_BACKBONE_EXPORTS = {"DCNv2Conv": DCNv2Conv, "PKIC3k2": PKIC3k2, "DWRC3k2": DWRC3k2}
'''


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def install() -> dict:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    modules_dir = pkg / "nn" / "modules"
    module_path = modules_dir / "yolo26_backbone.py"
    tasks_path = pkg / "nn" / "tasks.py"
    init_path = modules_dir / "__init__.py"

    module_path.write_text(MODULE_SRC.lstrip(), encoding="utf-8")

    init_src = _read(init_path)
    init_changed = False
    if "yolo26_backbone import DCNv2Conv" not in init_src:
        init_src = init_src.rstrip() + (
            "\n# yolo26 iter19 backbone modules\n"
            "from .yolo26_backbone import DCNv2Conv, PKIC3k2, DWRC3k2  # noqa: E402,F401\n"
        )
        _write(init_path, init_src)
        init_changed = True

    tasks_src = _read(tasks_path)
    tasks_changed = False
    if "yolo26_backbone import DCNv2Conv" not in tasks_src:
        tasks_src = tasks_src.rstrip() + (
            "\n\n# YOLO26 iter19 backbone modules injected by install_backbone_modules.py\n"
            "from ultralytics.nn.modules.yolo26_backbone import DCNv2Conv, PKIC3k2, DWRC3k2\n"
        )
        tasks_changed = True

    # All three change channels -> must be in base_modules (like SPDConv / C3k2).
    base_match = re.search(r"base_modules = frozenset\(\s*\{(?P<body>.*?)\n\s*\}\n\s*\)\n\s*repeat_modules", tasks_src, re.S)
    if not base_match:
        raise RuntimeError("Could not locate parse_model base_modules block")
    body = base_match.group("body")
    to_add = [m for m in ("DCNv2Conv", "PKIC3k2", "DWRC3k2") if m not in body]
    if to_add:
        start, end = base_match.span("body")
        insert = "".join(f"\n            {m}," for m in to_add)
        # insert after SPDConv if present, else after A2C2f.
        body_new, n = re.subn(r"(?m)^(\s*SPDConv,\s*)$", r"\1" + insert, body, count=1)
        if n != 1:
            body_new, n = re.subn(r"(?m)^(\s*A2C2f,\s*)$", r"\1" + insert, body, count=1)
        if n != 1:
            raise RuntimeError("Could not insert backbone modules into base_modules")
        tasks_src = tasks_src[:start] + body_new + tasks_src[end:]
        tasks_changed = True

    # PKIC3k2 / DWRC3k2 subclass C3k2 and support the `n` repeat arg -> add to repeat_modules.
    rep_match = re.search(r"repeat_modules = frozenset\([^{]*\{(?P<body>.*?)\n\s*\}\n\s*\)", tasks_src, re.S)
    if not rep_match:
        raise RuntimeError("Could not locate parse_model repeat_modules block")
    rbody = rep_match.group("body")
    rto_add = [m for m in ("PKIC3k2", "DWRC3k2") if m not in rbody]
    if rto_add:
        start, end = rep_match.span("body")
        insert = "".join(f"\n            {m}," for m in rto_add)
        rbody_new, n = re.subn(r"(?m)^(\s*C3k2,\s*)$", r"\1" + insert, rbody, count=1)
        if n != 1:
            raise RuntimeError("Could not insert into repeat_modules (C3k2 anchor not found)")
        tasks_src = tasks_src[:start] + rbody_new + tasks_src[end:]
        tasks_changed = True

    if tasks_changed:
        _write(tasks_path, tasks_src)

    return {
        "module_path": str(module_path),
        "init_changed": init_changed,
        "tasks_changed": tasks_changed,
    }


def main() -> int:
    try:
        info = install()
    except Exception as exc:  # noqa: BLE001
        print(f"INSTALL_BACKBONE_MODULES_FAILED: {exc}", file=sys.stderr)
        return 1
    for k, v in info.items():
        print(f"{k}: {v}")
    print("INSTALL_BACKBONE_MODULES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
