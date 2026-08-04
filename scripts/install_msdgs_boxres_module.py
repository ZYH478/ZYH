#!/usr/bin/env python
"""安装并注册 MSDGSBoxResDetect：保留原 P5 box tower 的零初始化输出残差头。"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

MODULE_SRC = r'''# Auto-generated MSDGSBoxResDetect for YOLO26.
# Injected by install_msdgs_boxres_module.py. Do not edit by hand.
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect

__all__ = ["MSDGSBoxResDetect"]


class MSDGSBoxResDetect(Detect):
    """MSDGS P5 定向 box-logit 残差头。

    原始 P3/P4/P5 box tower 与全部分类 tower 保持不变；新增分支只给 P5 的
    4*reg_max 个 box logits 提供零初始化残差。Detect.forward 仍负责 one2many、
    detached one2one、解码和 postprocess。
    """

    def __init__(
        self,
        nc: int = 80,
        residual_width: int = 16,
        use_p4_detail: bool = False,
        reg_max: int = 1,
        end2end: bool = True,
        ch: tuple = (),
    ):
        if len(ch) != 3:
            raise ValueError(f"MSDGSBoxResDetect expects P3/P4/P5 channels, got ch={ch}")
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.input_channels = tuple(int(x) for x in ch)
        self.residual_width = int(residual_width)
        self.use_p4_detail = bool(use_p4_detail)
        if self.residual_width <= 0:
            raise ValueError(f"residual_width must be positive, got {residual_width}")

        if self.use_p4_detail:
            residual = nn.Sequential(
                DWConv(ch[1], ch[1], 3, 2),
                Conv(ch[1], self.residual_width, 1, 1),
                nn.Conv2d(self.residual_width, 4 * self.reg_max, 1),
            )
        else:
            residual = nn.Sequential(
                Conv(ch[2], self.residual_width, 3, 1),
                Conv(self.residual_width, self.residual_width, 3, 1),
                nn.Conv2d(self.residual_width, 4 * self.reg_max, 1),
            )
        nn.init.zeros_(residual[-1].weight)
        nn.init.zeros_(residual[-1].bias)
        self.box_res_o2m = residual
        self.box_res_o2o = copy.deepcopy(residual) if end2end else None

    def _box_residual(self, x: list[torch.Tensor], one2one: bool) -> torch.Tensor:
        branch = self.box_res_o2o if one2one else self.box_res_o2m
        source = x[1] if self.use_p4_detail else x[2]
        return branch(source)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        one2one = self.end2end and box_head is self.one2one_cv2
        box_outputs = []
        for i in range(self.nl):
            box = box_head[i](x[i])
            if i == 2:
                box = box + self._box_residual(x, one2one=one2one)
            box_outputs.append(box.view(bs, 4 * self.reg_max, -1))
        boxes = torch.cat(box_outputs, dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def fuse(self) -> None:
        super().fuse()
        # end2end 融合推理只保留 one2one，删除不再使用的 one2many 残差参数。
        self.box_res_o2m = None
'''


def append_once(text: str, marker: str, addition: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text.rstrip() + "\n" + addition.lstrip(), True


def inject_detect_set(text: str) -> tuple[str, bool]:
    pattern = re.compile(r"(elif m in frozenset\(\s*\{\s*\n\s*Detect,\n)")
    match = pattern.search(text)
    if not match:
        raise RuntimeError("cannot locate Detect parse_model frozenset")
    block_end = text.find("        ):", match.start())
    if block_end < 0:
        raise RuntimeError("cannot locate end of Detect parse_model frozenset")
    if "MSDGSBoxResDetect" in text[match.start():block_end]:
        return text, False
    replacement = match.group(1) + "                MSDGSBoxResDetect,\n"
    return text[:match.start()] + replacement + text[match.end():], True


def inject_legacy_set(text: str) -> tuple[str, bool]:
    pattern = re.compile(r"(if m in \{)([^\n]*\bDetect\b[^\n]*)(\}:\s*\n\s*m\.legacy = legacy)")
    match = pattern.search(text)
    if not match:
        raise RuntimeError("cannot locate Detect legacy set")
    if "MSDGSBoxResDetect" in match.group(2):
        return text, False
    replacement = match.group(1) + "MSDGSBoxResDetect, " + match.group(2) + match.group(3)
    return text[:match.start()] + replacement + text[match.end():], True


def verify(tasks_text: str, init_text: str) -> None:
    assert "from .yolo26_msdgs_boxres import MSDGSBoxResDetect" in init_text
    assert "from ultralytics.nn.modules.yolo26_msdgs_boxres import MSDGSBoxResDetect" in tasks_text
    match = re.search(r"elif m in frozenset\(\s*\{(.*?)\n\s*\}\s*\n\s*\):", tasks_text, re.S)
    assert match and "MSDGSBoxResDetect" in match.group(1)
    legacy = re.search(r"if m in \{([^\n]+)\}:\s*\n\s*m\.legacy = legacy", tasks_text)
    assert legacy and "MSDGSBoxResDetect" in legacy.group(1)
    assert "args.extend([reg_max, end2end, [ch[x] for x in f]])" in tasks_text


def main() -> int:
    import ultralytics

    package = Path(ultralytics.__file__).resolve().parent
    module_file = package / "nn" / "modules" / "yolo26_msdgs_boxres.py"
    init_file = package / "nn" / "modules" / "__init__.py"
    tasks_file = package / "nn" / "tasks.py"
    print("ultralytics_version", ultralytics.__version__)
    print("package", package)
    module_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = append_once(
        init_text,
        "yolo26_msdgs_boxres",
        "# MSDGS P5 box-logit residual Detect\n"
        "from .yolo26_msdgs_boxres import MSDGSBoxResDetect  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_boxres_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = append_once(
        tasks_text,
        "yolo26_msdgs_boxres",
        "# MSDGS P5 box-logit residual Detect\n"
        "from ultralytics.nn.modules.yolo26_msdgs_boxres import MSDGSBoxResDetect  # noqa: E402,F401\n",
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
    print("INSTALL_MSDGS_BOXRES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
