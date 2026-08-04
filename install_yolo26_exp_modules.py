#!/usr/bin/env python
"""Idempotently install YOLO26 experimental modules into the active Ultralytics env.

This script patches the *current Python environment* (normally the remote
`yolo26` conda env) rather than storing secrets or machine-specific paths in the
repo.  It is intentionally small and repeatable:

1. Write `ultralytics/nn/modules/yolo26_exp.py` with custom modules.
2. Expose those classes to `ultralytics.nn.tasks` so YAML can reference them.
3. Add `SPDConv` to `parse_model()`'s `base_modules`, because it changes channel
   count and needs Ultralytics to pass `(c1, c2, ...)`.

The c-preserving modules (`SimAM`, `EMA`, `CoordAtt`, `LSKA`, `DySample`) are
kept out of `base_modules` on purpose.  `parse_model()` then preserves their
input channel count automatically; they lazily build channel-dependent
submodules on the first dummy forward that Ultralytics already runs during model
construction.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import textwrap


MODULE_SOURCE = r'''
"""Experimental YOLO26 modules for steel-defect / small-object ablations.

The modules here are deliberately YAML-friendly in Ultralytics:
- SPDConv changes channels and is registered as a base module.
- SimAM, EMA, CoordAtt, LSKA and DySample preserve channels and can be inserted
  as `[-1, 1, ModuleName, args]` without changing the channel ledger.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class SPDConv(nn.Module):
    """Space-to-depth followed by non-strided Conv.

    Intended as a drop-in replacement for stride-2 Conv downsampling.  It keeps
    more fine-grained spatial information for low-resolution or small objects.
    YAML example: `[-1, 1, SPDConv, [128, 3, 2]]`.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p=None, g: int = 1, act: bool = True):
        super().__init__()
        if s != 2:
            raise ValueError(f"SPDConv is intended for stride=2 downsampling, got stride={s}")
        self.conv = Conv(c1 * 4, c2, k, 1, p, g=g, act=act)

    @staticmethod
    def space_to_depth(x: torch.Tensor) -> torch.Tensor:
        # Pad odd feature maps defensively; YOLO train sizes are normally even.
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        return torch.cat(
            (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.space_to_depth(x))


class SimAM(nn.Module):
    """Parameter-free 3D attention from SimAM (ICML 2021)."""

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1] * x.shape[-2] - 1
        if n <= 0:
            return x
        x_minus_mu_sq = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)


class _LazyModule(nn.Module):
    """Small helper for channel-preserving modules that parse_model treats as c2=ch[f]."""

    def __init__(self):
        super().__init__()
        self._built_channels: int | None = None

    def _ensure_built(self, x: torch.Tensor) -> None:
        c = int(x.shape[1])
        if self._built_channels != c:
            self._build(c, device=x.device, dtype=x.dtype)
            self._built_channels = c

    def _build(self, c: int, device=None, dtype=None) -> None:  # pragma: no cover - abstract helper
        raise NotImplementedError


class EMA(_LazyModule):
    """Efficient Multi-Scale Attention with cross-spatial learning (c-preserving)."""

    def __init__(self, factor: int = 32):
        super().__init__()
        self.factor = int(factor)
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _valid_groups(c: int, requested: int) -> int:
        g = max(1, min(int(requested), c))
        while c % g != 0:
            g -= 1
        return max(1, g)

    def _build(self, c: int, device=None, dtype=None) -> None:
        groups = self._valid_groups(c, self.factor)
        gc = c // groups
        self.groups = groups
        self.group_channels = gc
        self.conv1x1 = nn.Conv2d(gc, gc, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(gc, gc, 3, 1, 1)
        self.gn = nn.GroupNorm(1, gc)
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_built(x)
        b, c, h, w = x.shape
        g = self.groups
        gx = x.reshape(b * g, c // g, h, w)
        x_h = gx.mean(dim=3, keepdim=True)
        x_w = gx.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(gx * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(gx)
        x11 = self.softmax(self.agp(x1).reshape(b * g, 1, -1))
        x12 = x2.reshape(b * g, -1, h * w)
        x21 = self.softmax(self.agp(x2).reshape(b * g, 1, -1))
        x22 = x1.reshape(b * g, -1, h * w)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * g, 1, h, w)
        return (gx * weights.sigmoid()).reshape(b, c, h, w)


class CoordAtt(_LazyModule):
    """Coordinate Attention, preserving position-sensitive long-range cues."""

    def __init__(self, reduction: int = 32):
        super().__init__()
        self.reduction = int(reduction)
        self.act = nn.SiLU(inplace=True)

    def _build(self, c: int, device=None, dtype=None) -> None:
        mip = max(8, c // self.reduction)
        self.conv1 = nn.Conv2d(c, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, c, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, c, 1, 1, 0)
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_built(x)
        identity = x
        n, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class LSKA(_LazyModule):
    """Large Selective Kernel style attention for texture/context robustness."""

    def __init__(self, k1: int = 5, k2: int = 7, dilation: int = 3):
        super().__init__()
        self.k1 = int(k1)
        self.k2 = int(k2)
        self.dilation = int(dilation)

    def _build(self, c: int, device=None, dtype=None) -> None:
        p1 = self.k1 // 2
        p2 = self.dilation * (self.k2 // 2)
        self.conv0 = nn.Conv2d(c, c, self.k1, padding=p1, groups=c)
        self.conv_spatial = nn.Conv2d(c, c, self.k2, padding=p2, dilation=self.dilation, groups=c)
        self.conv1 = nn.Conv2d(c, c, 1)
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_built(x)
        attn = self.conv1(self.conv_spatial(self.conv0(x))).sigmoid()
        return x * attn


class DySample(_LazyModule):
    """Dynamic point-sampling upsampler. Channel-preserving replacement for nearest upsample."""

    def __init__(self, scale: int = 2, style: str = "lp", groups: int = 4, dyscope: bool = False):
        super().__init__()
        if style not in {"lp", "pl"}:
            raise ValueError(f"style must be 'lp' or 'pl', got {style}")
        self.scale = int(scale)
        self.style = style
        self.groups = int(groups)
        self.dyscope = bool(dyscope)

    @staticmethod
    def _valid_groups(c: int, requested: int) -> int:
        g = max(1, min(int(requested), c))
        while c % g != 0:
            g -= 1
        return max(1, g)

    def _init_pos(self, groups: int, device=None, dtype=None) -> torch.Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1, device=device, dtype=dtype) / self.scale
        pos = torch.stack(torch.meshgrid(h, h, indexing="ij")).transpose(1, 2)
        return pos.repeat(1, groups, 1).reshape(1, -1, 1, 1)

    def _build(self, c: int, device=None, dtype=None) -> None:
        groups = self._valid_groups(c, self.groups)
        self.groups = groups
        if self.style == "pl":
            if c % (self.scale ** 2) != 0:
                raise ValueError(f"DySample style='pl' requires channels divisible by scale^2, got c={c}")
            offset_in = c // (self.scale ** 2)
            offset_out = 2 * groups
        else:
            offset_in = c
            offset_out = 2 * groups * self.scale ** 2
        self.offset = nn.Conv2d(offset_in, offset_out, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)
        if self.dyscope:
            self.scope = nn.Conv2d(offset_in, offset_out, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)
        if "init_pos" in self._buffers:
            self._buffers["init_pos"] = self._init_pos(groups, device=device, dtype=dtype)
        else:
            self.register_buffer("init_pos", self._init_pos(groups, device=device, dtype=dtype))
        self.to(device=device, dtype=dtype)

    def sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="xy")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([w, h], device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w).permute(0, 2, 3, 4, 1).contiguous()
        coords = coords.flatten(0, 1)
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward_lp(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        if self.dyscope:
            offset = offset * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = offset * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x: torch.Tensor) -> torch.Tensor:
        x_ = F.pixel_shuffle(x, self.scale)
        offset = self.offset(x_)
        if self.dyscope:
            offset = F.pixel_unshuffle(offset * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(offset, self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_built(x)
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)
'''


def _append_once(path: Path, marker: str, text: str) -> bool:
    src = path.read_text(encoding="utf-8")
    if marker in src:
        return False
    path.write_text(src.rstrip() + "\n\n" + text.rstrip() + "\n", encoding="utf-8")
    return True


def install() -> dict[str, str | bool]:
    import ultralytics  # noqa: PLC0415

    pkg = Path(ultralytics.__file__).resolve().parent
    modules_dir = pkg / "nn" / "modules"
    module_path = modules_dir / "yolo26_exp.py"
    tasks_path = pkg / "nn" / "tasks.py"
    init_path = modules_dir / "__init__.py"

    module_path.write_text(MODULE_SOURCE.lstrip(), encoding="utf-8")

    init_changed = _append_once(
        init_path,
        "yolo26_exp import SimAM",
        "from .yolo26_exp import SimAM, EMA, CoordAtt, LSKA, SPDConv, DySample  # YOLO26 experimental modules",
    )

    tasks_src = tasks_path.read_text(encoding="utf-8")
    tasks_changed = False
    if "yolo26_exp import SimAM" not in tasks_src:
        tasks_src = tasks_src.rstrip() + (
            "\n\n# YOLO26 experimental modules injected by install_yolo26_exp_modules.py\n"
            "from ultralytics.nn.modules.yolo26_exp import SimAM, EMA, CoordAtt, LSKA, SPDConv, DySample\n"
        )
        tasks_changed = True

    base_match = re.search(r"base_modules = frozenset\(\s*\{(?P<body>.*?)\n\s*\}\n\s*\)\n\s*repeat_modules", tasks_src, re.S)
    if not base_match:
        raise RuntimeError("Could not locate parse_model base_modules block")
    if "SPDConv" not in base_match.group("body"):
        start, end = base_match.span("body")
        body = base_match.group("body")
        body_new, n = re.subn(r"(?m)^(\s*A2C2f,\s*)$", r"\1\n            SPDConv,", body, count=1)
        if n != 1:
            raise RuntimeError("Could not insert SPDConv after A2C2f in base_modules")
        tasks_src = tasks_src[:start] + body_new + tasks_src[end:]
        tasks_changed = True

    if tasks_changed:
        tasks_path.write_text(tasks_src, encoding="utf-8")

    return {
        "ultralytics_pkg": str(pkg),
        "module_path": str(module_path),
        "tasks_path": str(tasks_path),
        "init_changed": init_changed,
        "tasks_changed": tasks_changed,
    }


def main() -> int:
    try:
        info = install()
    except Exception as exc:  # noqa: BLE001
        print(f"INSTALL_YOLO26_EXP_MODULES_FAILED: {exc}", file=sys.stderr)
        return 1
    for k, v in info.items():
        print(f"{k}: {v}")
    print("INSTALL_YOLO26_EXP_MODULES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
