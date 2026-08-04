#!/usr/bin/env python
"""iter37 方案D：给 ultralytics 8.4.93 注入轻量多核 backbone 增强模块 PKIC3k2（纯 PyTorch）。

立场说明（如实记录，不粉饰）：
- backbone 侧多尺度/信息保真增强在本 goal 已 13 次失败，iter29 把结论升级为
  「NEU-DET 小数据上 backbone 侧多尺度增强边际贡献≈0」。D 方案本质就是这一类。
- owner 明确要求试 D。按纪律：只替换 backbone 层4 C3k2、不动深层/下采样、不和 head 改动混跑、
  单独 seed0 验证。这是一枪定性的实验，不是主线。

设计（PKINet inception-style 多核思想的 nano 版）：
- 原版 PKINet 用 inception 式并联多尺度深度卷积（3/5/7/9/11）捕获不同尺度纹理。
- 这里做通道守恒 lite bottleneck：cv1(c1->c_) -> 并联 DW{3,5,7} 求和(各向同性多核) -> pw 折回 ->
  cv2(c_->c2)，shortcut 残差。比标准 Bottleneck 的单 3x3 多了 5/7 两个大核感受野，
  参数增量可控（DW 卷积，3x3+5x5+7x7 = 9+25+49=83 乘 c_ 个参数，仍是轻量级）。
- PKIC3k2 继承 C3k2，只替换 self.m 为 _PKILiteBottleneck 串。

放置：MSDGS neck 基座 + backbone 层4 C3k2 -> PKIC3k2（[-1, 2, C3k2, [128, false, 0.25]]，
P3/8 前的 80x80 分辨率，浅层多尺度纹理）。保持 end2end/reg_max=1，从官方 yolo26n.pt 迁移。

远程用法：
    python install_pki_module.py    # 幂等
成功打印 INSTALL_PKI_MODULE_OK。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


MODULE_SRC = r'''# Auto-generated lightweight poly-kernel backbone module for YOLO26 (NEU-DET iter37 方案D).
# Injected by install_pki_module.py. Do not edit by hand.
# Multi-kernel design inspired by PKINet (inception-style parallel depthwise conv), nano-lite variant.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2


class _PKILiteBottleneck(nn.Module):
    """Channel-preserving bottleneck with parallel depthwise multi-kernel (3/5/7) summation.

    cv1(c1->c_) -> [DW3 + DW5 + DW7](c_->c_, isotropic multi-scale) -> BN+act -> pw fold ->
    cv2(c_->c2). shortcut residual when c1==c2. Adds 5x5/7x7 receptive fields over the
    standard single-3x3 Bottleneck at low depthwise cost.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, kernels=(3, 5, 7)):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        # parallel depthwise multi-kernel (groups=c_), same padding per kernel
        self.dws = nn.ModuleList(
            nn.Conv2d(c_, c_, k, 1, k // 2, groups=c_, bias=False) for k in kernels
        )
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.pw = Conv(c_, c_, 1, 1)  # fuse multi-kernel responses
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        h = self.cv1(x)
        multi = self.dws[0](h)
        for dw in self.dws[1:]:
            multi = multi + dw(h)
        h = self.act(self.bn(multi))
        h = self.pw(h)
        y = self.cv2(h)
        return x + y if self.add else y


class PKIC3k2(C3k2):
    """C3k2 with lightweight poly-kernel bottlenecks (parallel DW 3/5/7).

    Drop-in for a backbone C3k2. YAML: [-1, n, PKIC3k2, [c2, shortcut]].
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(_PKILiteBottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))


_PKI_EXPORTS = {"PKIC3k2": PKIC3k2}
'''


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def install() -> dict:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    modules_dir = pkg / "nn" / "modules"
    module_path = modules_dir / "yolo26_pki.py"
    tasks_path = pkg / "nn" / "tasks.py"
    init_path = modules_dir / "__init__.py"

    module_path.write_text(MODULE_SRC.lstrip(), encoding="utf-8")

    init_src = _read(init_path)
    init_changed = False
    if "yolo26_pki import PKIC3k2" not in init_src:
        init_src = init_src.rstrip() + (
            "\n# yolo26 iter37 pki module\n"
            "from .yolo26_pki import PKIC3k2  # noqa: E402,F401\n"
        )
        _write(init_path, init_src)
        init_changed = True

    tasks_src = _read(tasks_path)
    tasks_changed = False
    if "yolo26_pki import PKIC3k2" not in tasks_src:
        tasks_src = tasks_src.rstrip() + (
            "\n\n# YOLO26 iter37 pki module injected by install_pki_module.py\n"
            "from ultralytics.nn.modules.yolo26_pki import PKIC3k2\n"
        )
        tasks_changed = True

    # PKIC3k2 keeps c1->c2 -> base_modules (like C3k2).
    base_match = re.search(
        r"base_modules = frozenset\(\s*\{(?P<body>.*?)\n\s*\}\n\s*\)\n\s*repeat_modules", tasks_src, re.S
    )
    if not base_match:
        raise RuntimeError("Could not locate parse_model base_modules block")
    body = base_match.group("body")
    if "PKIC3k2" not in body:
        start, end = base_match.span("body")
        insert = "\n            PKIC3k2,"
        body_new, n = re.subn(r"(?m)^(\s*C3k2,\s*)$", r"\1" + insert, body, count=1)
        if n != 1:
            raise RuntimeError("Could not insert PKIC3k2 into base_modules (C3k2 anchor not found)")
        tasks_src = tasks_src[:start] + body_new + tasks_src[end:]
        tasks_changed = True

    # PKIC3k2 subclasses C3k2 and supports the `n` repeat arg -> repeat_modules.
    rep_match = re.search(r"repeat_modules = frozenset\([^{]*\{(?P<body>.*?)\n\s*\}\n\s*\)", tasks_src, re.S)
    if not rep_match:
        raise RuntimeError("Could not locate parse_model repeat_modules block")
    rbody = rep_match.group("body")
    if "PKIC3k2" not in rbody:
        start, end = rep_match.span("body")
        insert = "\n            PKIC3k2,"
        rbody_new, n = re.subn(r"(?m)^(\s*C3k2,\s*)$", r"\1" + insert, rbody, count=1)
        if n != 1:
            raise RuntimeError("Could not insert PKIC3k2 into repeat_modules (C3k2 anchor not found)")
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
        print(f"INSTALL_PKI_MODULE_FAILED: {exc}", file=sys.stderr)
        return 1
    for k, v in info.items():
        print(f"{k}: {v}")
    print("INSTALL_PKI_MODULE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
