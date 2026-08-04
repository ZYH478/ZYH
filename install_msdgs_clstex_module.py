#!/usr/bin/env python
"""Install MSDGSClsTexDetect: P4 texture evidence added only to P5 classification logits.

The candidate leaves backbone, MSDGS neck, box towers, anchor features, and P3/P4
classification outputs unchanged. A tiny zero-initialized residual converts P4 local
high-frequency evidence into six P5 class-logit corrections. Separate one2many and
one2one residual modules avoid conflicting supervision; the one2one input remains the
official detached feature list.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = r'''# Auto-generated MSDGSClsTexDetect for YOLO26 NEU-DET.
# Injected by install_msdgs_clstex_module.py. Do not edit by hand.
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect

__all__ = ["MSDGSClsTexDetect"]


class _P4TextureToP5Logits(nn.Module):
    """Convert full-resolution P4 local contrast into a P5 class-logit residual.

    The final 1x1 projection is zero initialized, so the complete Detect output is
    exactly identical to the standard head before optimization. The fixed residual
    scale bounds early perturbation while preserving gradients to the projection.
    """

    def __init__(self, channels: int, nc: int, pool_kernel: int = 3):
        super().__init__()
        if pool_kernel < 3 or pool_kernel % 2 == 0:
            raise ValueError(f"pool_kernel must be odd and >=3, got {pool_kernel}")
        self.pool = nn.AvgPool2d(
            pool_kernel, stride=1, padding=pool_kernel // 2, count_include_pad=False
        )
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.proj = nn.Conv2d(channels, nc, 1, 1, 0, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, p4: torch.Tensor) -> torch.Tensor:
        high = p4 - self.pool(p4)
        return self.proj(self.down(high))


class MSDGSClsTexDetect(Detect):
    """Standard Detect plus a P4->P5 classification-logit texture residual.

    Invariants:
    - cv2 / one2one_cv2 and all box logits are untouched;
    - cv3 inputs at P3/P4 are untouched;
    - only P5 class logits receive the residual;
    - feats remains the original input list;
    - one2one uses its own residual module on the official detached feature list.
    """

    def __init__(
        self,
        nc: int = 80,
        logit_scale: float = 0.1,
        pool_kernel: int = 3,
        reg_max: int = 1,
        end2end: bool = True,
        ch: tuple = (),
    ):
        if len(ch) != 3:
            raise ValueError(f"MSDGSClsTexDetect expects P3/P4/P5 channels, got ch={ch}")
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.input_channels = tuple(int(v) for v in ch)
        self.logit_scale = float(logit_scale)
        self.pool_kernel = int(pool_kernel)
        self.p5_cls_texture = _P4TextureToP5Logits(ch[1], nc, self.pool_kernel)
        self.one2one_p5_cls_texture = copy.deepcopy(self.p5_cls_texture) if end2end else None

    def _texture_module(self, cls_head: nn.Module):
        if self.end2end and cls_head is self.one2one_cv3:
            return self.one2one_p5_cls_texture
        return self.p5_cls_texture

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:  # fused one2many path
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1
        )
        texture = self._texture_module(cls_head)(x[1])
        score_parts = []
        for i in range(self.nl):
            logits = cls_head[i](x[i])
            if i == 2:
                if logits.shape[-2:] != texture.shape[-2:]:
                    raise RuntimeError(
                        f"P4 texture/P5 logits shape mismatch: {texture.shape} vs {logits.shape}"
                    )
                logits = logits + self.logit_scale * texture
            score_parts.append(logits.view(bs, self.nc, -1))
        scores = torch.cat(score_parts, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def fuse(self) -> None:
        super().fuse()
        # one2many is removed by Detect.fuse(); remove its now-unused residual too.
        self.p5_cls_texture = None
'''


def append_once(text: str, marker: str, addition: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text.rstrip() + "\n" + addition.lstrip(), True


def inject_detect_set(text: str) -> tuple[str, bool]:
    for match in re.finditer(r"elif m in frozenset\(\s*\{(.*?)\n\s*\}\s*\n\s*\):", text, re.S):
        body = match.group(1)
        if re.search(r"^\s*Detect,\s*$", body, re.M):
            if "MSDGSClsTexDetect" in body:
                return text, False
            new_body = re.sub(
                r"(^\s*Detect,\s*$)", r"\1\n                MSDGSClsTexDetect,", body, count=1, flags=re.M
            )
            return text[: match.start(1)] + new_body + text[match.end(1) :], True
    raise RuntimeError("cannot locate Detect parse_model frozenset")


def inject_legacy_set(text: str) -> tuple[str, bool]:
    pattern = re.compile(r"(if m in \{)([^\n]*\bDetect\b[^\n]*)(\}:\s*\n\s*m\.legacy = legacy)")
    match = pattern.search(text)
    if not match:
        raise RuntimeError("cannot locate Detect legacy set")
    if "MSDGSClsTexDetect" in match.group(2):
        return text, False
    replacement = match.group(1) + "MSDGSClsTexDetect, " + match.group(2) + match.group(3)
    return text[: match.start()] + replacement + text[match.end() :], True


def verify(tasks_text: str, init_text: str) -> None:
    assert "from .yolo26_msdgs_clstex import MSDGSClsTexDetect" in init_text
    assert "from ultralytics.nn.modules.yolo26_msdgs_clstex import MSDGSClsTexDetect" in tasks_text
    assert "MSDGSClsTexDetect" in tasks_text
    assert "args.extend([reg_max, end2end, [ch[x] for x in f]])" in tasks_text


def main() -> int:
    import ultralytics

    package = Path(ultralytics.__file__).resolve().parent
    module_file = package / "nn" / "modules" / "yolo26_msdgs_clstex.py"
    init_file = package / "nn" / "modules" / "__init__.py"
    tasks_file = package / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("package", package)
    module_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = append_once(
        init_text,
        "yolo26_msdgs_clstex",
        "# MSDGS classification texture Detect\n"
        "from .yolo26_msdgs_clstex import MSDGSClsTexDetect  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_clstex_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = append_once(
        tasks_text,
        "yolo26_msdgs_clstex",
        "# MSDGS classification texture Detect\n"
        "from ultralytics.nn.modules.yolo26_msdgs_clstex import MSDGSClsTexDetect  # noqa: E402,F401\n",
    )
    tasks_text, set_changed = inject_detect_set(tasks_text)
    tasks_text, legacy_changed = inject_legacy_set(tasks_text)
    if import_changed or set_changed or legacy_changed:
        tasks_file.write_text(tasks_text, encoding="utf-8")

    verify(tasks_file.read_text(encoding="utf-8"), init_file.read_text(encoding="utf-8"))
    print("init_changed", init_changed)
    print("tasks_import_changed", import_changed)
    print("tasks_detect_set_changed", set_changed)
    print("tasks_legacy_set_changed", legacy_changed)
    print("INSTALL_MSDGS_CLSTEX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
