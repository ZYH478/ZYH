#!/usr/bin/env python
"""Install the paper-derived EMSC and C3k2EMSC modules into Ultralytics 8.4.93.

EMSC keeps half of the channels untouched, splits the other half into two groups,
processes them with depthwise/grouped 3x3 and 5x5 convolutions, concatenates all
channels, and applies a 1x1 fusion convolution. C3k2EMSC preserves the original
C3k2 parameter names and pretrained transfer, then applies one EMSC refinement at
the stage output. It is used only at MSDGS backbone layer 8 (P5 stage).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

MODULE_SRC = r'''# Auto-generated EMSC modules for YOLO26 MSDGS iteration 045.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2

__all__ = ["EMSC", "C3k2EMSC"]


class EMSC(nn.Module):
    """Efficient Multi-Scale Convolution from MSAF-YOLO, channel preserving.

    Half of the channels are an identity path. The other half is rearranged into
    two channel groups and processed independently with 3x3 and 5x5 depthwise
    grouped convolutions. A final 1x1 convolution mixes the concatenated output.
    """

    def __init__(self, channels: int):
        super().__init__()
        channels = int(channels)
        if channels < 4:
            raise ValueError(f"EMSC requires at least 4 channels, got {channels}")
        self.c_identity = channels // 2
        c_multi = channels - self.c_identity
        self.c_k3 = c_multi // 2
        self.c_k5 = c_multi - self.c_k3
        self.k3 = Conv(self.c_k3, self.c_k3, 3, 1, g=self.c_k3)
        self.k5 = Conv(self.c_k5, self.c_k5, 5, 1, g=self.c_k5)
        self.fuse = Conv(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_identity, x_multi = torch.split(x, [self.c_identity, self.c_k3 + self.c_k5], dim=1)
        x_k3, x_k5 = torch.split(x_multi, [self.c_k3, self.c_k5], dim=1)
        return self.fuse(torch.cat((x_identity, self.k3(x_k3), self.k5(x_k5)), dim=1))


class C3k2EMSC(C3k2):
    """C3k2 with one EMSC refinement at its output.

    The inherited C3k2 remains structurally and name-compatible with official
    pretrained weights. Only the added ``emsc`` branch is newly initialized.
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
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.emsc = EMSC(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emsc(super().forward(x))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        return self.emsc(super().forward_split(x))
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text.rstrip() + import_line, True


def inject_into_set(text: str, set_name: str, anchor: str, symbol: str) -> tuple[str, bool]:
    match = re.search(rf"({set_name}\s*=\s*frozenset\([^{{]*\{{)(.*?)(\n\s*\}})", text, re.S)
    if not match:
        raise RuntimeError(f"could not locate {set_name}")
    body = match.group(2)
    if re.search(rf"\b{re.escape(symbol)}\b", body):
        return text, False
    updated = body.replace(f"{anchor},\n", f"{anchor},\n            {symbol},\n", 1)
    if updated == body:
        raise RuntimeError(f"anchor {anchor} not found in {set_name}")
    return text[:match.start()] + match.group(1) + updated + match.group(3) + text[match.end():], True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    modules_dir = pkg / "nn" / "modules"
    module_file = modules_dir / "yolo26_emsc.py"
    init_file = modules_dir / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)
    module_file.write_text(MODULE_SRC.lstrip(), encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = inject_import(
        init_text, "yolo26_emsc import EMSC",
        "\n\n# Efficient Multi-Scale Convolution (MSAF-YOLO)\n"
        "from .yolo26_emsc import EMSC, C3k2EMSC  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".emsc_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = inject_import(
        tasks_text, "yolo26_emsc import EMSC",
        "\n\n# Efficient Multi-Scale Convolution for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_emsc import EMSC, C3k2EMSC  # noqa: E402,F401\n",
    )
    tasks_text, base_changed = inject_into_set(tasks_text, "base_modules", "C3k2", "C3k2EMSC")
    tasks_text, repeat_changed = inject_into_set(tasks_text, "repeat_modules", "C3k2", "C3k2EMSC")
    if import_changed or base_changed or repeat_changed:
        tasks_file.write_text(tasks_text, encoding="utf-8")

    print("init_changed", init_changed)
    print("tasks_import_changed", import_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)
    print("INSTALL_EMSC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
