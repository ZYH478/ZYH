#!/usr/bin/env python
"""Install Stable-DCNv2 modules for the MSDGS P4 backbone experiment (iter062).

This installer adds new class names and does not modify the legacy C3k2DCNv2
implementation, so all existing checkpoints remain loadable.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = r'''# Auto-generated Stable-DCNv2 modules for YOLO26 NEU-DET (iter062).
# Injected by install_msdgs_backbone_stable_dcnv2_module.py. Do not edit by hand.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C3k2, Bottleneck
from ultralytics.nn.modules.conv import Conv

__all__ = [
    "DCNv2StableUnit",
    "C3k2StableDCNv2",
    "iter_stable_dcnv2_units",
    "set_stable_deform_strength",
    "stable_deform_strength_for_epoch",
    "register_stable_offset_mask_grad_hooks",
    "transfer_standard_to_stable_dcnv2",
]


class DCNv2StableUnit(nn.Module):
    """Bounded, progressively enabled modulated deformable 3x3 convolution.

    At deform_strength=0 the operation is exactly the regular-grid convolution:
    offset=0 and mask=1. During training, strength is warmed from 0 to 1 while
    tanh bounds keep the final offset in [-1.5, 1.5] and mask in [0.5, 1.5].
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
        offset_limit: float = 1.5,
    ):
        super().__init__()
        from torchvision.ops import DeformConv2d

        self.k = int(k)
        self.s = int(s)
        self.pad = self.k // 2
        self.groups = int(g)
        self.offset_limit = float(offset_limit)
        self.offset_mask = nn.Conv2d(
            c1, 3 * self.k * self.k, self.k, stride=self.s, padding=self.pad, bias=True
        )
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        self.deform = DeformConv2d(
            c1, c2, self.k, stride=self.s, padding=self.pad, groups=self.groups, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self._split = 2 * self.k * self.k
        self.register_buffer("deform_strength", torch.tensor(0.0, dtype=torch.float32))

    def compute_offset_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        om = self.offset_mask(x)
        raw_offset = om[:, : self._split]
        raw_mask = om[:, self._split :]
        limit = self.offset_limit
        bounded_offset = limit * torch.tanh(raw_offset / limit)
        mask_delta = 0.5 * torch.tanh(raw_mask)
        strength = self.deform_strength.to(dtype=x.dtype)
        offset = strength * bounded_offset
        mask = 1.0 + strength * mask_delta
        return offset, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A zero-offset deform kernel can still differ from cuDNN conv2d by ~1e-3
        # because it uses a different CUDA accumulation path. During the protected
        # strength=0 phase, execute the exact regular convolution with the shared
        # deform weight. Once strength becomes non-zero, switch to deform_conv2d.
        if self.deform_strength.item() == 0.0:
            y = F.conv2d(
                x,
                self.deform.weight,
                bias=None,
                stride=self.s,
                padding=self.pad,
                groups=self.groups,
            )
        else:
            offset, mask = self.compute_offset_mask(x)
            y = self.deform(x, offset, mask)
        return self.act(self.bn(y))


def _swap_bottleneck_cv2(bottleneck: Bottleneck) -> None:
    """Replace an eligible Bottleneck 3x3 cv2 while preserving shape/BN/activation."""
    old = bottleneck.cv2
    conv = old.conv
    if conv.kernel_size[0] != 3 or conv.stride[0] != 1:
        return
    bottleneck.cv2 = DCNv2StableUnit(
        conv.in_channels,
        conv.out_channels,
        k=3,
        s=1,
        g=conv.groups,
        act=True,
    )


class C3k2StableDCNv2(C3k2):
    """C3k2 with only inner Bottleneck.cv2 replaced by Stable-DCNv2 units."""

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
    ):
        super().__init__(c1, c2, n=n, c3k=c3k, e=e, attn=attn, g=g, shortcut=shortcut)
        self.deform_units = 0
        for block in self.m:
            for module in block.modules():
                if (
                    isinstance(module, Bottleneck)
                    and isinstance(module.cv2, Conv)
                    and module.cv2.conv.kernel_size[0] == 3
                    and module.cv2.conv.stride[0] == 1
                ):
                    _swap_bottleneck_cv2(module)
                    self.deform_units += 1


def iter_stable_dcnv2_units(module):
    for child in module.modules():
        if isinstance(child, DCNv2StableUnit):
            yield child


def set_stable_deform_strength(module, strength: float) -> int:
    value = min(1.0, max(0.0, float(strength)))
    count = 0
    for unit in iter_stable_dcnv2_units(module):
        unit.deform_strength.fill_(value)
        count += 1
    return count


def stable_deform_strength_for_epoch(
    epoch: int,
    hold_epochs: int = 20,
    full_epoch: int = 80,
) -> float:
    """Epochs 0..20 use 0; epochs 20..80 ramp linearly; >=80 use 1."""
    epoch = int(epoch)
    if epoch <= hold_epochs:
        return 0.0
    if epoch >= full_epoch:
        return 1.0
    return float(epoch - hold_epochs) / float(full_epoch - hold_epochs)


def register_stable_offset_mask_grad_hooks(module, scale: float = 0.25):
    """Scale only offset/mask predictor gradients; caller keeps returned handles alive."""
    scale = float(scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"gradient scale must be in (0, 1], got {scale}")
    handles = []
    for unit in iter_stable_dcnv2_units(module):
        for parameter in unit.offset_mask.parameters():
            handles.append(parameter.register_hook(lambda grad, factor=scale: grad * factor))
    return handles


def transfer_standard_to_stable_dcnv2(stable_model, standard_state_dict) -> int:
    """Copy each replaced standard Conv2d weight into its deformable main weight."""
    target_state = stable_model.state_dict()
    transferred = 0
    for key in list(target_state.keys()):
        if key.endswith(".cv2.deform.weight"):
            source_key = key[: -len(".deform.weight")] + ".conv.weight"
            if source_key not in standard_state_dict:
                raise KeyError(f"missing source weight for {key}: {source_key}")
            src = standard_state_dict[source_key]
            if src.shape != target_state[key].shape:
                raise ValueError(
                    f"shape mismatch {key} {tuple(target_state[key].shape)} vs "
                    f"{source_key} {tuple(src.shape)}"
                )
            target_state[key] = src.clone()
            transferred += 1
    stable_model.load_state_dict(target_state, strict=True)
    return transferred
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
    assert "from .yolo26_msdgs_backbone_stable_dcnv2 import C3k2StableDCNv2" in init_text
    assert (
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_stable_dcnv2 import C3k2StableDCNv2"
        in tasks_text
    )
    for set_name in ("base_modules", "repeat_modules"):
        pattern = re.compile(
            rf"{set_name}\s*=\s*frozenset\([^{{}}]*\{{(.*?)\n\s*\}}\s*\)", re.S
        )
        match = pattern.search(tasks_text)
        assert match and re.search(r"^\s*C3k2StableDCNv2,\s*$", match.group(1), re.M)


def main() -> int:
    import ultralytics

    package = Path(ultralytics.__file__).resolve().parent
    module_file = package / "nn" / "modules" / "yolo26_msdgs_backbone_stable_dcnv2.py"
    init_file = package / "nn" / "modules" / "__init__.py"
    tasks_file = package / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("package", package)
    module_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = append_once(
        init_text,
        "yolo26_msdgs_backbone_stable_dcnv2",
        "# MSDGS P4 backbone Stable-DCNv2 replacement (iter062)\n"
        "from .yolo26_msdgs_backbone_stable_dcnv2 import C3k2StableDCNv2  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_backbone_stable_dcnv2_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = append_once(
        tasks_text,
        "yolo26_msdgs_backbone_stable_dcnv2",
        "# MSDGS P4 backbone Stable-DCNv2 replacement (iter062)\n"
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_stable_dcnv2 import C3k2StableDCNv2  # noqa: E402,F401\n",
    )
    tasks_text, base_changed = inject_frozenset_member(
        tasks_text, "base_modules", "C3k2", "C3k2StableDCNv2"
    )
    tasks_text, repeat_changed = inject_frozenset_member(
        tasks_text, "repeat_modules", "C3k2", "C3k2StableDCNv2"
    )
    if import_changed or base_changed or repeat_changed:
        tasks_file.write_text(tasks_text, encoding="utf-8")

    verify(tasks_file.read_text(encoding="utf-8"), init_file.read_text(encoding="utf-8"))
    print("init_changed", init_changed)
    print("tasks_import_changed", import_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)
    print("INSTALL_MSDGS_BACKBONE_STABLE_DCNV2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
