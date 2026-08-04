#!/usr/bin/env python
"""iter30：给 ultralytics 8.4.93 注入 Dynamic Snake Conv 骨干增强模块（纯 PyTorch，免编译）。

动机（承接 iter29 诊断）：
- crazing 是本 goal 万年最弱类（test mAP50 0.443），形态是弥散弯曲的裂纹网；
  map50-95/map50 比值 0.347 全场最低 → 定位吃亏于「弯曲细长结构」。
- Dynamic Snake Conv（DSCNet，ICCV2023）核心：卷积核沿蛇形曲线可学习变形，
  沿单轴蜿蜒采样，天生追踪细长/弯曲管状结构。对 crazing/scratches/rolled-in 对症。
- 本 goal 唯一有效规律「几何自适应/信息保真类有效」——DySnake 属几何自适应类，契合。

集成方式（通道守恒的 DySnake-C3k2，照 install_backbone_modules.py 的 DWRC3k2 范式）：
- 原版 DySnakeConv 输出 3×ouc（conv_0 + snake_x + snake_y concat），不能直接顶替一层。
- 这里做成通道守恒 bottleneck：cv1(c1→c_) → [DSConv_x, DSConv_y] concat 2c_
  → 1×1 折回 c_ → cv2(c_→c2)，shortcut 残差。捕获 x/y 双朝向蛇形（裂纹任意方向）。
- 变形核心（DSConv/DSC）从 LightYOLO 成熟实现原样搬入，只加 bottleneck 包装 + C3k2 子类。

放置：MSDGS neck 基座 + backbone P4 层6 C3k2 → DySnakeC3k2（保守分辨率甜点，40×40）。
保持 end2end/reg_max=1，从官方 yolo26n.pt 迁移。

远程用法：
    python install_dysnake_module.py    # 幂等
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


MODULE_SRC = r'''# Auto-generated Dynamic Snake Conv module for YOLO26 (NEU-DET iter30).
# Injected by install_dysnake_module.py. Do not edit by hand.
# Deformation core (DSConv / DSC) adapted from DSCNet (ICCV2023) / LightYOLO impl.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2


class DSConv(nn.Module):
    """Dynamic Snake Convolution single-orientation deformable conv.

    morph=0: snake along x-axis; morph=1: snake along y-axis. Channel-preserving-ish
    (out_ch configurable). Learns per-location offsets, iteratively accumulated along
    the kernel to mimic a snake's continuous swing, then samples via bilinear interp.
    """

    def __init__(self, in_ch, out_ch, morph, kernel_size=3, if_offset=True, extend_scope=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size
        self.dsc_conv_x = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1),
                                    stride=(kernel_size, 1), padding=0)
        self.dsc_conv_y = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size),
                                    stride=(1, kernel_size), padding=0)
        self.gn = nn.GroupNorm(max(out_ch // 4, 1), out_ch)
        self.act = Conv.default_act
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = _DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
        x = self.gn(x)
        x = self.act(x)
        return x


class _DSC(object):
    """Coordinate map + bilinear sampling core for Dynamic Snake Conv."""

    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2),
                               int(self.num_points))
            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)
            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)
            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)
            y_new = y_center + y_grid
            x_new = x_center + x_grid
            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)
            y_offset_new = y_offset.detach().clone()
            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))
            y_new = y_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([self.num_batch, self.num_points * self.width, 1 * self.height])
            x_new = x_new.reshape([self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([self.num_batch, self.num_points * self.width, 1 * self.height])
            return y_new, x_new
        else:
            y = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2),
                               int(self.num_points))
            x = torch.linspace(0, 0, 1)
            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)
            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)
            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)
            y_new = y_center + y_grid
            x_new = x_center + x_grid
            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)
            y_new = y_new.to(device)
            x_new = x_new.to(device)
            x_offset_new = x_offset.detach().clone()
            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))
            y_new = y_new.reshape([self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([self.num_batch, 1 * self.width, self.num_points * self.height])
            x_new = x_new.reshape([self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([self.num_batch, 1 * self.width, self.num_points * self.height])
            return y_new, x_new

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()
        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width
        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()
        repeat = torch.ones([self.num_points * self.width * self.height]).unsqueeze(0)
        repeat = repeat.float()
        base = torch.matmul(base, repeat)
        base = base.reshape([-1])
        base = base.to(device)
        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)
        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()
        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)
        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 + value_c1 * vol_c1)
        if self.morph == 0:
            outputs = outputs.reshape([self.num_batch, self.num_points * self.width,
                                       1 * self.height, self.num_channels])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([self.num_batch, 1 * self.width,
                                       self.num_points * self.height, self.num_channels])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


class _DySnakeBottleneck(nn.Module):
    """Bottleneck whose 3x3 is replaced by dual-orientation snake conv (x + y), channel-preserving.

    cv1(c1->c_) -> [DSConv_x(c_->c_), DSConv_y(c_->c_)] concat 2c_ -> 1x1 fold to c_ -> cv2(c_->c2).
    Captures crack networks of arbitrary orientation. shortcut residual when c1==c2.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.snake_x = DSConv(c_, c_, morph=0, kernel_size=3)
        self.snake_y = DSConv(c_, c_, morph=1, kernel_size=3)
        self.fold = Conv(2 * c_, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        h = self.cv1(x)
        h = torch.cat([self.snake_x(h), self.snake_y(h)], dim=1)
        h = self.fold(h)
        y = self.cv2(h)
        return x + y if self.add else y


class DySnakeC3k2(C3k2):
    """C3k2 with dual-orientation Dynamic Snake bottlenecks (geometry-adaptive, tracks curved thin structures).

    Drop-in for a backbone C3k2. YAML: [-1, n, DySnakeC3k2, [c2, shortcut]].
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(_DySnakeBottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))


_DYSNAKE_EXPORTS = {"DySnakeC3k2": DySnakeC3k2}
'''


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def install() -> dict:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    modules_dir = pkg / "nn" / "modules"
    module_path = modules_dir / "yolo26_dysnake.py"
    tasks_path = pkg / "nn" / "tasks.py"
    init_path = modules_dir / "__init__.py"

    module_path.write_text(MODULE_SRC.lstrip(), encoding="utf-8")

    init_src = _read(init_path)
    init_changed = False
    if "yolo26_dysnake import DySnakeC3k2" not in init_src:
        init_src = init_src.rstrip() + (
            "\n# yolo26 iter30 dysnake module\n"
            "from .yolo26_dysnake import DySnakeC3k2  # noqa: E402,F401\n"
        )
        _write(init_path, init_src)
        init_changed = True

    tasks_src = _read(tasks_path)
    tasks_changed = False
    if "yolo26_dysnake import DySnakeC3k2" not in tasks_src:
        tasks_src = tasks_src.rstrip() + (
            "\n\n# YOLO26 iter30 dysnake module injected by install_dysnake_module.py\n"
            "from ultralytics.nn.modules.yolo26_dysnake import DySnakeC3k2\n"
        )
        tasks_changed = True

    # DySnakeC3k2 changes hidden channels internally but keeps c1->c2 -> base_modules (like C3k2).
    base_match = re.search(
        r"base_modules = frozenset\(\s*\{(?P<body>.*?)\n\s*\}\n\s*\)\n\s*repeat_modules", tasks_src, re.S
    )
    if not base_match:
        raise RuntimeError("Could not locate parse_model base_modules block")
    body = base_match.group("body")
    if "DySnakeC3k2" not in body:
        start, end = base_match.span("body")
        insert = "\n            DySnakeC3k2,"
        body_new, n = re.subn(r"(?m)^(\s*C3k2,\s*)$", r"\1" + insert, body, count=1)
        if n != 1:
            raise RuntimeError("Could not insert DySnakeC3k2 into base_modules (C3k2 anchor not found)")
        tasks_src = tasks_src[:start] + body_new + tasks_src[end:]
        tasks_changed = True

    # DySnakeC3k2 subclasses C3k2 and supports the `n` repeat arg -> add to repeat_modules.
    rep_match = re.search(r"repeat_modules = frozenset\([^{]*\{(?P<body>.*?)\n\s*\}\n\s*\)", tasks_src, re.S)
    if not rep_match:
        raise RuntimeError("Could not locate parse_model repeat_modules block")
    rbody = rep_match.group("body")
    if "DySnakeC3k2" not in rbody:
        start, end = rep_match.span("body")
        insert = "\n            DySnakeC3k2,"
        rbody_new, n = re.subn(r"(?m)^(\s*C3k2,\s*)$", r"\1" + insert, rbody, count=1)
        if n != 1:
            raise RuntimeError("Could not insert DySnakeC3k2 into repeat_modules (C3k2 anchor not found)")
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
        print(f"INSTALL_DYSNAKE_MODULE_FAILED: {exc}", file=sys.stderr)
        return 1
    for k, v in info.items():
        print(f"{k}: {v}")
    print("INSTALL_DYSNAKE_MODULE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
