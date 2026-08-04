#!/usr/bin/env python
"""Install the MSDGS P3-to-P4 anti-alias residual downsampling module.

The custom layer inherits the official Ultralytics Conv, preserving its complete
conv/bn/act main path and state-dict keys. A fixed binomial blur plus a
zero-initialized 1x1 projection adds a tightly bounded low-frequency residual.
The installer is idempotent and registers the class only in parse_model's
base_modules set because this candidate is a single non-repeat downsampling layer.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = r'''# Auto-generated AntiAliasResidualConv for YOLO26 NEU-DET.
# Injected by install_msdgs_backbone_antialias_module.py. Do not edit by hand.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["AntiAliasResidualConv"]


class AntiAliasResidualConv(Conv):
    """Official stride-2 Conv plus a zero-init local anti-alias residual."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        residual_scale: float = 0.05,
    ):
        if int(k) != 3 or int(s) != 2:
            raise ValueError(f"AntiAliasResidualConv requires k=3,s=2; got k={k}, s={s}")
        if not 0.0 <= float(residual_scale) <= 0.05:
            raise ValueError(
                f"residual_scale must be in [0, 0.05], got {residual_scale}"
            )
        super().__init__(c1, c2, k=3, s=2)
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = torch.outer(kernel_1d, kernel_1d).div_(16.0)
        self.register_buffer(
            "aa_kernel",
            kernel_2d.view(1, 1, 3, 3).repeat(int(c1), 1, 1, 1),
            persistent=False,
        )
        self.aa_project = nn.Conv2d(int(c1), int(c2), 1, 1, 0, bias=False)
        nn.init.zeros_(self.aa_project.weight)
        self.residual_scale = float(residual_scale)

    def _anti_alias_residual(self, x: torch.Tensor) -> torch.Tensor:
        low = F.conv2d(
            x,
            self.aa_kernel,
            stride=2,
            padding=1,
            groups=x.shape[1],
        )
        return self.aa_project(low)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.residual_scale * self._anti_alias_residual(x)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward_fuse(x) + self.residual_scale * self._anti_alias_residual(x)
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


def frozenset_body(text: str, set_name: str) -> str:
    pattern = re.compile(
        rf"{set_name}\s*=\s*frozenset\([^{{}}]*\{{(.*?)\n\s*\}}\s*\)", re.S
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"cannot locate {set_name} frozenset")
    return match.group(1)


def verify(tasks_text: str, init_text: str) -> None:
    assert "from .yolo26_msdgs_backbone_antialias import AntiAliasResidualConv" in init_text
    assert (
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_antialias import AntiAliasResidualConv"
        in tasks_text
    )
    base_body = frozenset_body(tasks_text, "base_modules")
    assert re.search(r"^\s*AntiAliasResidualConv,\s*$", base_body, re.M)
    repeat_body = frozenset_body(tasks_text, "repeat_modules")
    assert not re.search(r"^\s*AntiAliasResidualConv,\s*$", repeat_body, re.M)


def main() -> int:
    import ultralytics

    package = Path(ultralytics.__file__).resolve().parent
    module_file = package / "nn" / "modules" / "yolo26_msdgs_backbone_antialias.py"
    init_file = package / "nn" / "modules" / "__init__.py"
    tasks_file = package / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("package", package)
    module_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = append_once(
        init_text,
        "yolo26_msdgs_backbone_antialias",
        "# MSDGS P3-to-P4 local anti-alias residual\n"
        "from .yolo26_msdgs_backbone_antialias import AntiAliasResidualConv  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_backbone_antialias_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = append_once(
        tasks_text,
        "yolo26_msdgs_backbone_antialias",
        "# MSDGS P3-to-P4 local anti-alias residual\n"
        "from ultralytics.nn.modules.yolo26_msdgs_backbone_antialias import AntiAliasResidualConv  # noqa: E402,F401\n",
    )
    tasks_text, base_changed = inject_frozenset_member(
        tasks_text, "base_modules", "Conv", "AntiAliasResidualConv"
    )
    if import_changed or base_changed:
        tasks_file.write_text(tasks_text, encoding="utf-8")

    verify(tasks_file.read_text(encoding="utf-8"), init_file.read_text(encoding="utf-8"))
    print("init_changed", init_changed)
    print("tasks_import_changed", import_changed)
    print("tasks_base_modules_changed", base_changed)
    print("INSTALL_MSDGS_BACKBONE_ANTIALIAS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
