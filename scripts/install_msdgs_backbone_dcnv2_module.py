#!/usr/bin/env python
"""Install C3k2DCNv2 for the MSDGS P4 backbone single-module replacement (iter057).

Design (main-path replacement, NOT a residual branch — per iter056 closure):
- C3k2DCNv2 subclasses the official C3k2 and keeps the complete CSP split / cv1 /
  cv2 / C3k structure so almost all pretrained yolo26n.pt weights still transfer.
- Only the inner 3x3 feature-extraction conv (`cv2`) of every innermost Bottleneck
  is replaced by a modulated deformable conv (DCNv2, torchvision.ops.deform_conv2d,
  official op, no CUDA compilation).
- Init is grid-equivalent: offset predictor zero-init -> zero offsets -> regular
  3x3 grid sampling; modulation mask = 2*sigmoid(0) = 1 -> no scaling. The deform
  conv main weight uses standard Conv init, so at init the block behaves like a
  fresh standard 3x3 conv (the intended geometry-adaptive experiment).

Rationale: this is the only 'information-preserving / geometry-adaptive' backbone
module (the goal's single proven-effective family, same as SPDConv +2.56pp) that
has never been trained on the MSDGS line. Sampling points adapt to defect geometry
(elongated scratches, diffuse crazing) instead of a rigid square grid.

The installer is idempotent and registers the class in Ultralytics exports plus
parse_model's base/repeat module sets. Because C3k2DCNv2 keeps the exact C3k2
constructor signature, the YAML entry stays [512, True] (c3k=True).

Remote usage:
    python install_msdgs_backbone_dcnv2_module.py   # idempotent
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = r'''# Auto-generated C3k2DCNv2 for YOLO26 NEU-DET (iter057).
# Injected by install_msdgs_backbone_dcnv2_module.py. Do not edit by hand.
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C3k2, C3k, Bottleneck
from ultralytics.nn.modules.conv import Conv

__all__ = ["C3k2DCNv2", "DCNv2Unit"]


class DCNv2Unit(nn.Module):
    """Modulated deformable conv (DCNv2) as a drop-in for a stride-1 3x3 Conv.

    Uses torchvision.ops.deform_conv2d (official op, no compilation). A side
    predictor produces per-location sampling offsets (2*k*k) and modulation
    logits (k*k). Both predictors are zero-initialized so at init:
        offset = 0                  -> regular 3x3 grid (identical geometry to a
                                       standard conv)
        mask   = 2*sigmoid(0) = 1   -> no per-point scaling
    The main deform weight uses standard Conv (kaiming) init, so the unit starts
    as a well-behaved fresh 3x3 conv and learns geometry-adaptive sampling.

    Mirrors ultralytics Conv's (conv -> bn -> act) shape so it slots in where a
    3x3 Conv was, but it is intentionally NOT fusable (kept separate at inference;
    params/latency accounted for explicitly).
    """

    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, g: int = 1, act=True):
        super().__init__()
        from torchvision.ops import DeformConv2d

        self.k = int(k)
        self.s = int(s)
        self.pad = self.k // 2
        self.groups = int(g)
        # Offset (2*k*k) + modulation logits (k*k) predictor over the input.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        om = self.offset_mask(x)
        offset = om[:, : self._split]
        mask = 2.0 * torch.sigmoid(om[:, self._split :])
        y = self.deform(x, offset, mask)
        return self.act(self.bn(y))


def _swap_bottleneck_cv2(bottleneck: Bottleneck) -> None:
    """Replace a Bottleneck's 3x3 cv2 with a stride-1 DCNv2Unit (same channels)."""
    old = bottleneck.cv2
    conv = old.conv
    c1 = conv.in_channels
    c2 = conv.out_channels
    k = conv.kernel_size[0]
    g = conv.groups
    if k != 3 or conv.stride[0] != 1:
        # Only the 3x3 stride-1 feature-extraction conv is deformable-replaced.
        return
    bottleneck.cv2 = DCNv2Unit(c1, c2, k=3, s=1, g=g, act=True)


class C3k2DCNv2(C3k2):
    """Official C3k2 whose inner Bottleneck 3x3 convs are modulated deformable convs.

    Keeps the CSP split / cv1 / cv2 outer structure and the C3k nesting; only the
    innermost Bottleneck.cv2 (3x3, stride 1) is replaced by DCNv2Unit. Constructor
    signature is identical to C3k2 so the YAML entry is unchanged ([c2, c3k]).
    """

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
                if isinstance(module, Bottleneck):
                    if isinstance(module.cv2, Conv) and module.cv2.conv.kernel_size[0] == 3 \
                            and module.cv2.conv.stride[0] == 1:
                        _swap_bottleneck_cv2(module)
                        self.deform_units += 1


def transfer_standard_to_dcnv2(dcnv2_model, standard_state_dict) -> int:
    """Copy pretrained standard 3x3 conv weights into DCNv2Unit deform weights.

    A DeformConv2d weight has the exact same shape as the standard Conv2d weight
    it replaces, and with offset=0 + mask=1 (both zero-init here) it computes the
    identical regular-grid convolution. So copying `...cv2.conv.weight` from a
    standard model into the DCNv2 model's `...cv2.deform.weight` makes the block
    numerically equivalent to the pretrained conv at init (information-preserving,
    grid-equivalent). The batchnorm keys share names and are loaded normally by
    ultralytics; only these deform weights need the explicit rename-transfer.

    Returns the number of deform weights transferred.
    """
    target_state = dcnv2_model.state_dict()
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
    dcnv2_model.load_state_dict(target_state, strict=True)
    return transferred
'''


def append_once(text: str, marker: str, addition: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text.rstrip() + "\n" + addition.lstrip(), True


def inject_frozenset_member(
    text: str, set_name: str, anchor: str, member: str
) -> tuple[str, bool]:
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
    assert "from .yolo26_msdgs_backbone_dcnv2 import C3k2DCNv2" in init_text
    assert (
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_dcnv2 import C3k2DCNv2"
        in tasks_text
    )
    for set_name in ("base_modules", "repeat_modules"):
        pattern = re.compile(
            rf"{set_name}\s*=\s*frozenset\([^{{}}]*\{{(.*?)\n\s*\}}\s*\)", re.S
        )
        match = pattern.search(tasks_text)
        assert match and re.search(r"^\s*C3k2DCNv2,\s*$", match.group(1), re.M)


def main() -> int:
    import ultralytics

    package = Path(ultralytics.__file__).resolve().parent
    module_file = package / "nn" / "modules" / "yolo26_msdgs_backbone_dcnv2.py"
    init_file = package / "nn" / "modules" / "__init__.py"
    tasks_file = package / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("package", package)
    module_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = append_once(
        init_text,
        "yolo26_msdgs_backbone_dcnv2",
        "# MSDGS P4 backbone DCNv2 single-module replacement (iter057)\n"
        "from .yolo26_msdgs_backbone_dcnv2 import C3k2DCNv2  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_backbone_dcnv2_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = append_once(
        tasks_text,
        "yolo26_msdgs_backbone_dcnv2",
        "# MSDGS P4 backbone DCNv2 single-module replacement (iter057)\n"
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_dcnv2 import C3k2DCNv2  # noqa: E402,F401\n",
    )
    tasks_text, base_changed = inject_frozenset_member(
        tasks_text, "base_modules", "C3k2", "C3k2DCNv2"
    )
    tasks_text, repeat_changed = inject_frozenset_member(
        tasks_text, "repeat_modules", "C3k2", "C3k2DCNv2"
    )
    if import_changed or base_changed or repeat_changed:
        tasks_file.write_text(tasks_text, encoding="utf-8")

    verify(tasks_file.read_text(encoding="utf-8"), init_file.read_text(encoding="utf-8"))
    print("init_changed", init_changed)
    print("tasks_import_changed", import_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)
    print("INSTALL_MSDGS_BACKBONE_DCNV2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
