#!/usr/bin/env python
"""Install C3k2DirectionalResidual for the MSDGS P4 backbone ablation.

The module preserves the complete official C3k2 main path and adds a narrow,
zero-initialized, spatially adaptive directional residual. The installer is
idempotent and registers the class in Ultralytics exports plus parse_model's
base/repeat module sets.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = r'''# Auto-generated C3k2DirectionalResidual for YOLO26 NEU-DET.
# Injected by install_msdgs_backbone_dirres_module.py. Do not edit by hand.
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv

__all__ = ["C3k2DirectionalResidual"]


class C3k2DirectionalResidual(C3k2):
    """Official C3k2 plus a zero-init, per-location directional residual."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        branch_channels: int = 32,
        direction_kernel: int = 5,
        residual_scale: float = 0.1,
    ):
        super().__init__(c1, c2, n=n, c3k=c3k, e=0.5, attn=False, g=1, shortcut=True)
        branch_channels = int(branch_channels)
        direction_kernel = int(direction_kernel)
        residual_scale = float(residual_scale)
        if branch_channels < 1 or branch_channels > int(c2):
            raise ValueError(
                f"branch_channels must be in [1, c2], got {branch_channels} for c2={c2}"
            )
        if direction_kernel < 3 or direction_kernel % 2 == 0:
            raise ValueError(
                f"direction_kernel must be odd and >=3, got {direction_kernel}"
            )
        if not 0.0 <= residual_scale <= 0.1:
            raise ValueError(f"residual_scale must be in [0, 0.1], got {residual_scale}")

        self.branch_channels = branch_channels
        self.direction_kernel = direction_kernel
        self.residual_scale = residual_scale
        self.dir_reduce = Conv(c2, branch_channels, 1, 1)
        self.dir_h = Conv(
            branch_channels,
            branch_channels,
            (1, direction_kernel),
            1,
            g=branch_channels,
        )
        self.dir_v = Conv(
            branch_channels,
            branch_channels,
            (direction_kernel, 1),
            1,
            g=branch_channels,
        )
        self.dir_local = Conv(
            branch_channels,
            branch_channels,
            3,
            1,
            g=branch_channels,
        )
        self.dir_gate = nn.Conv2d(branch_channels, 3, 1, 1, 0, bias=True)
        self.dir_project = nn.Conv2d(branch_channels, c2, 1, 1, 0, bias=True)
        nn.init.zeros_(self.dir_gate.weight)
        nn.init.zeros_(self.dir_gate.bias)
        nn.init.zeros_(self.dir_project.weight)
        nn.init.zeros_(self.dir_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        z = self.dir_reduce(y)
        branches = torch.stack(
            (self.dir_h(z), self.dir_v(z), self.dir_local(z)),
            dim=1,
        )
        weights = torch.softmax(self.dir_gate(z), dim=1).unsqueeze(2)
        residual = self.dir_project((branches * weights).sum(dim=1))
        return y + self.residual_scale * residual
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
    assert (
        "from .yolo26_msdgs_backbone_dirres import C3k2DirectionalResidual"
        in init_text
    )
    assert (
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_dirres import "
        "C3k2DirectionalResidual"
        in tasks_text
    )
    for set_name in ("base_modules", "repeat_modules"):
        pattern = re.compile(
            rf"{set_name}\s*=\s*frozenset\([^{{}}]*\{{(.*?)\n\s*\}}\s*\)", re.S
        )
        match = pattern.search(tasks_text)
        assert match and re.search(
            r"^\s*C3k2DirectionalResidual,\s*$", match.group(1), re.M
        )


def main() -> int:
    import ultralytics

    package = Path(ultralytics.__file__).resolve().parent
    module_file = package / "nn" / "modules" / "yolo26_msdgs_backbone_dirres.py"
    init_file = package / "nn" / "modules" / "__init__.py"
    tasks_file = package / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("package", package)
    module_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = append_once(
        init_text,
        "yolo26_msdgs_backbone_dirres",
        "# MSDGS P4 content-adaptive directional residual\n"
        "from .yolo26_msdgs_backbone_dirres import C3k2DirectionalResidual  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_backbone_dirres_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = append_once(
        tasks_text,
        "yolo26_msdgs_backbone_dirres",
        "# MSDGS P4 content-adaptive directional residual\n"
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_dirres import "
        "C3k2DirectionalResidual  # noqa: E402,F401\n",
    )
    tasks_text, base_changed = inject_frozenset_member(
        tasks_text, "base_modules", "C3k2", "C3k2DirectionalResidual"
    )
    tasks_text, repeat_changed = inject_frozenset_member(
        tasks_text, "repeat_modules", "C3k2", "C3k2DirectionalResidual"
    )
    if import_changed or base_changed or repeat_changed:
        tasks_file.write_text(tasks_text, encoding="utf-8")

    verify(tasks_file.read_text(encoding="utf-8"), init_file.read_text(encoding="utf-8"))
    print("init_changed", init_changed)
    print("tasks_import_changed", import_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)
    print("INSTALL_MSDGS_BACKBONE_DIRRES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
