#!/usr/bin/env python
"""Install a crazing-protected hybrid DCNv2 block for the MSDGS P4 backbone.

The original iter058 candidate replaced every output channel of the inner 3x3
Bottleneck conv by DCNv2. It improved global test mAP50/mAP50-95 but hurt
crazing. This module preserves complementary feature families inside the same
conv output:

- regular branch: a fixed-grid 3x3 Conv2d for a configurable share of channels;
- deform branch: DCNv2 for the remaining channels, with bounded offsets and
  bounded modulation around one;
- concatenation keeps the original output channel count and order.

Pretrained transfer slices the original conv weight along output channels, so
at offset=0 and mask=1 the hybrid block is numerically equivalent to the
standard pretrained conv at initialization. The regular channels can never
lose their fixed sampling grid, providing the explicit crazing protection that
was missing in iter058.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = r'''# Auto-generated crazing-protected Hybrid DCNv2 for YOLO26 NEU-DET.
# Injected by install_msdgs_backbone_hybrid_dcnv2_module.py. Do not edit by hand.
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C3k2, Bottleneck
from ultralytics.nn.modules.conv import Conv

__all__ = ["C3k2HybridDCNv2", "HybridDCNv2Unit", "transfer_standard_to_hybrid_dcnv2"]


class HybridDCNv2Unit(nn.Module):
    """Channel-partitioned regular/deform 3x3 convolution.

    Output channels are split into a regular-grid branch and a deformable branch.
    The deform branch is deliberately constrained:
      offset = offset_limit * tanh(raw / offset_limit)
      mask   = 1 + mask_span * tanh(raw_mask)
    Zero-initialized predictors therefore give offset=0 and mask=1 exactly.
    """

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        g: int = 1,
        act=True,
        deform_ratio: float = 0.5,
        offset_limit: float = 1.0,
        mask_span: float = 0.5,
    ):
        super().__init__()
        from torchvision.ops import DeformConv2d

        self.k = int(k)
        self.s = int(s)
        self.pad = self.k // 2
        self.groups = int(g)
        self.deform_ratio = float(deform_ratio)
        self.offset_limit = float(offset_limit)
        self.mask_span = float(mask_span)
        if not (0.0 < self.deform_ratio < 1.0):
            raise ValueError(f"deform_ratio must be in (0,1), got {self.deform_ratio}")
        if self.offset_limit <= 0.0:
            raise ValueError(f"offset_limit must be >0, got {self.offset_limit}")
        if not (0.0 < self.mask_span <= 1.0):
            raise ValueError(f"mask_span must be in (0,1], got {self.mask_span}")

        # Keep both branches group-compatible. The MSDGS layer uses g=1, but
        # rounding by g keeps the unit correct for grouped Bottlenecks too.
        c_def = int(round(c2 * self.deform_ratio / self.groups)) * self.groups
        c_def = min(max(self.groups, c_def), c2 - self.groups)
        c_reg = c2 - c_def
        if c_reg <= 0 or c_reg % self.groups or c_def % self.groups:
            raise ValueError(f"invalid channel split c2={c2}, g={g}, reg={c_reg}, deform={c_def}")
        self.c_regular = int(c_reg)
        self.c_deform = int(c_def)

        self.regular = nn.Conv2d(
            c1, self.c_regular, self.k, stride=self.s, padding=self.pad,
            groups=self.groups, bias=False,
        )
        self.offset_mask = nn.Conv2d(
            c1, 3 * self.k * self.k, self.k, stride=self.s, padding=self.pad, bias=True
        )
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        self.deform = DeformConv2d(
            c1, self.c_deform, self.k, stride=self.s, padding=self.pad,
            groups=self.groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self._split = 2 * self.k * self.k

    def sampling_tensors(self, x: torch.Tensor):
        raw = self.offset_mask(x)
        raw_offset = raw[:, : self._split]
        raw_mask = raw[:, self._split :]
        # tanh(raw / L) has derivative 1 at zero after multiplying by L.
        offset = self.offset_limit * torch.tanh(raw_offset / self.offset_limit)
        mask = 1.0 + self.mask_span * torch.tanh(raw_mask)
        return offset, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        regular = self.regular(x)
        offset, mask = self.sampling_tensors(x)
        deform = self.deform(x, offset, mask)
        return self.act(self.bn(torch.cat((regular, deform), dim=1)))


def _swap_bottleneck_cv2(
    bottleneck: Bottleneck,
    deform_ratio: float,
    offset_limit: float,
    mask_span: float,
) -> bool:
    old = bottleneck.cv2
    if not isinstance(old, Conv):
        return False
    conv = old.conv
    if conv.kernel_size[0] != 3 or conv.stride[0] != 1:
        return False
    bottleneck.cv2 = HybridDCNv2Unit(
        conv.in_channels,
        conv.out_channels,
        k=3,
        s=1,
        g=conv.groups,
        act=True,
        deform_ratio=deform_ratio,
        offset_limit=offset_limit,
        mask_span=mask_span,
    )
    return True


class C3k2HybridDCNv2(C3k2):
    """Official C3k2 with channel-partitioned regular/deform inner 3x3 convs."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        deform_ratio: float = 0.5,
        offset_limit: float = 1.0,
        mask_span: float = 0.5,
    ):
        super().__init__(c1, c2, n=n, c3k=c3k, e=e, attn=attn, g=g, shortcut=shortcut)
        self.deform_ratio = float(deform_ratio)
        self.offset_limit = float(offset_limit)
        self.mask_span = float(mask_span)
        self.hybrid_units = 0
        for block in self.m:
            for module in block.modules():
                if isinstance(module, Bottleneck) and _swap_bottleneck_cv2(
                    module, self.deform_ratio, self.offset_limit, self.mask_span
                ):
                    self.hybrid_units += 1


def transfer_standard_to_hybrid_dcnv2(hybrid_model, standard_state_dict) -> int:
    """Slice each pretrained standard conv into regular/deform output channels.

    Concatenating the two branch outputs recreates the original channel order.
    With zero offsets and unit masks this is exactly the pretrained regular-grid
    convolution, up to the tiny numeric epsilon of deform_conv2d.
    """
    target_state = hybrid_model.state_dict()
    transferred_units = 0
    regular_keys = [k for k in target_state if k.endswith(".cv2.regular.weight")]
    for regular_key in regular_keys:
        prefix = regular_key[: -len(".regular.weight")]
        deform_key = prefix + ".deform.weight"
        source_key = prefix + ".conv.weight"
        if source_key not in standard_state_dict:
            raise KeyError(f"missing source weight for {regular_key}: {source_key}")
        if deform_key not in target_state:
            raise KeyError(f"missing paired deform weight: {deform_key}")
        src = standard_state_dict[source_key]
        c_reg = target_state[regular_key].shape[0]
        c_def = target_state[deform_key].shape[0]
        if src.shape[0] != c_reg + c_def or src.shape[1:] != target_state[regular_key].shape[1:]:
            raise ValueError(
                f"shape mismatch source={source_key}:{tuple(src.shape)} "
                f"regular={tuple(target_state[regular_key].shape)} "
                f"deform={tuple(target_state[deform_key].shape)}"
            )
        target_state[regular_key] = src[:c_reg].clone()
        target_state[deform_key] = src[c_reg:c_reg + c_def].clone()
        transferred_units += 1
    hybrid_model.load_state_dict(target_state, strict=True)
    return transferred_units
'''


def append_once(text: str, marker: str, addition: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text.rstrip() + "\n" + addition.lstrip(), True


def inject_frozenset_member(text: str, set_name: str, anchor: str, member: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"({re.escape(set_name)}\s*=\s*frozenset\([^{{}}]*\{{)(.*?)(\n\s*\}}\s*\))",
        re.S,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"cannot locate {set_name} frozenset")
    body = match.group(2)
    if re.search(rf"^\s*{re.escape(member)},\s*$", body, re.M):
        return text, False
    anchor_pattern = re.compile(rf"(^\s*{re.escape(anchor)},\s*$)", re.M)
    if not anchor_pattern.search(body):
        raise RuntimeError(f"cannot locate {anchor} in {set_name}")
    new_body = anchor_pattern.sub(rf"\1\n            {member},", body, count=1)
    return text[: match.start(2)] + new_body + text[match.end(2) :], True


def verify(tasks_text: str, init_text: str) -> None:
    assert "from .yolo26_msdgs_backbone_hybrid_dcnv2 import C3k2HybridDCNv2" in init_text
    assert (
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_hybrid_dcnv2 import C3k2HybridDCNv2"
        in tasks_text
    )
    for set_name in ("base_modules", "repeat_modules"):
        pattern = re.compile(
            rf"{set_name}\s*=\s*frozenset\([^{{}}]*\{{(.*?)\n\s*\}}\s*\)", re.S
        )
        match = pattern.search(tasks_text)
        assert match and re.search(r"^\s*C3k2HybridDCNv2,\s*$", match.group(1), re.M)


def main() -> int:
    import ultralytics

    package = Path(ultralytics.__file__).resolve().parent
    module_file = package / "nn" / "modules" / "yolo26_msdgs_backbone_hybrid_dcnv2.py"
    init_file = package / "nn" / "modules" / "__init__.py"
    tasks_file = package / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("package", package)
    module_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = append_once(
        init_text,
        "yolo26_msdgs_backbone_hybrid_dcnv2",
        "# MSDGS P4 backbone crazing-protected Hybrid DCNv2\n"
        "from .yolo26_msdgs_backbone_hybrid_dcnv2 import C3k2HybridDCNv2  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_backbone_hybrid_dcnv2_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = append_once(
        tasks_text,
        "yolo26_msdgs_backbone_hybrid_dcnv2",
        "# MSDGS P4 backbone crazing-protected Hybrid DCNv2\n"
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_hybrid_dcnv2 import C3k2HybridDCNv2  # noqa: E402,F401\n",
    )
    tasks_text, base_changed = inject_frozenset_member(
        tasks_text, "base_modules", "C3k2", "C3k2HybridDCNv2"
    )
    tasks_text, repeat_changed = inject_frozenset_member(
        tasks_text, "repeat_modules", "C3k2", "C3k2HybridDCNv2"
    )
    if import_changed or base_changed or repeat_changed:
        tasks_file.write_text(tasks_text, encoding="utf-8")

    verify(tasks_file.read_text(encoding="utf-8"), init_file.read_text(encoding="utf-8"))
    print("init_changed", init_changed)
    print("tasks_import_changed", import_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)
    print("INSTALL_MSDGS_BACKBONE_HYBRID_DCNV2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
