#!/usr/bin/env python
"""安装并注册 MSDGSDetect：MSDGS 基座的 P5 定向 Detect 精度增强头。

YAML 构造参数（parse_model 会在末尾继续注入 reg_max/end2end/ch）：
    MSDGSDetect(nc, p5_box_width, use_p4p5_fusion, gate_scale, use_p5_cafm,
                reg_max, end2end, ch)

实现的干净消融：
- P4->P5 零初始化有界残差细节融合；
- 仅扩宽 P5 box tower；
- P5 检测入口零初始化 Gated CAFM。

安全：幂等修改 site-packages，并首次备份 tasks.py。凭证和实验配置不写入本脚本。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = r'''# Auto-generated MSDGSDetect for YOLO26.
# Injected by install_msdgs_detect_module.py. Do not edit by hand.
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect

__all__ = ["MSDGSDetect"]


class _CAKA(nn.Module):
    """CAFM 的 channel-aware kernel adaptor。"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        strip = 3 * kernel_size + 2
        self.channels = channels
        self.dw_square = nn.Conv2d(
            channels, channels, kernel_size, 1, kernel_size // 2, groups=channels, bias=False
        )
        self.dw_horizontal = nn.Conv2d(
            channels, channels, (1, strip), 1, (0, strip // 2), groups=channels, bias=False
        )
        self.dw_vertical = nn.Conv2d(
            channels, channels, (strip, 1), 1, (strip // 2, 0), groups=channels, bias=False
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Conv2d(channels, 3 * channels, 1, bias=True)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (self.dw_square(x), self.dw_horizontal(x), self.dw_vertical(x))
        batch = x.shape[0]
        weights = self.gate(self.pool(x)).view(batch, 3, self.channels, 1, 1)
        weights = torch.softmax(weights, dim=1)
        fused = sum(weights[:, i] * branch for i, branch in enumerate(branches))
        return self.act(self.bn(fused))


class _P5CAFM(nn.Module):
    """只服务于 P5 检测入口的 CAFM，不替换 backbone C2PSA。"""

    def __init__(self, channels: int):
        super().__init__()
        left = channels // 2
        right = channels - left
        self.split = (left, right)
        self.big = _CAKA(left, kernel_size=5)
        self.small = _CAKA(right, kernel_size=3)
        self.fuse = Conv(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = torch.split(x, self.split, dim=1)
        return self.fuse(torch.cat((self.big(left), self.small(right)), dim=1))


class MSDGSDetect(Detect):
    """MSDGS 的 P5 定向 Detect。

    官方 Detect 的 one2many/one2one、detach、解码、postprocess、fuse 全部沿用。
    本类只在进入官方 forward 前生成新的 P5，并按配置仅重建 P5 box tower。
    """

    def __init__(
        self,
        nc: int = 80,
        p5_box_width: int = 16,
        use_p4p5_fusion: bool = False,
        gate_scale: float = 0.1,
        use_p5_cafm: bool = False,
        reg_max: int = 1,
        end2end: bool = True,
        ch: tuple = (),
    ):
        if len(ch) != 3:
            raise ValueError(f"MSDGSDetect expects P3/P4/P5 channels, got ch={ch}")
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.input_channels = tuple(int(x) for x in ch)
        self.p5_box_width = int(p5_box_width)
        self.use_p4p5_fusion = bool(use_p4p5_fusion)
        self.gate_scale = float(gate_scale)
        self.use_p5_cafm = bool(use_p5_cafm)

        if self.p5_box_width < 4 * self.reg_max:
            raise ValueError(
                f"p5_box_width={self.p5_box_width} must be >= 4*reg_max={4 * self.reg_max}"
            )

        if self.use_p4p5_fusion:
            self.p4_to_p5 = nn.Sequential(
                DWConv(ch[1], ch[1], 3, 2),
                Conv(ch[1], ch[2], 1, 1),
            )
            self.alpha = nn.Parameter(torch.zeros(()))
        else:
            self.p4_to_p5 = None
            self.register_parameter("alpha", None)

        if self.use_p5_cafm:
            self.p5_cafm = _P5CAFM(ch[2])
            self.beta = nn.Parameter(torch.zeros(()))
        else:
            self.p5_cafm = None
            self.register_parameter("beta", None)

        standard_width = max(16, ch[0] // 4, self.reg_max * 4)
        if self.p5_box_width != standard_width:
            p5_box = nn.Sequential(
                Conv(ch[2], self.p5_box_width, 3),
                Conv(self.p5_box_width, self.p5_box_width, 3),
                nn.Conv2d(self.p5_box_width, 4 * self.reg_max, 1),
            )
            self.cv2[2] = p5_box
            if end2end:
                self.one2one_cv2[2] = copy.deepcopy(p5_box)

    def _prepare_features(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(x) != 3:
            raise ValueError(f"MSDGSDetect expects 3 feature maps, got {len(x)}")
        p5 = x[2]
        if self.p4_to_p5 is not None:
            detail = self.p4_to_p5(x[1])
            p5 = p5 + self.gate_scale * torch.tanh(self.alpha) * detail
        if self.p5_cafm is not None:
            p5 = p5 + self.gate_scale * torch.tanh(self.beta) * self.p5_cafm(p5)
        return [x[0], x[1], p5]

    def forward(self, x: list[torch.Tensor]):
        # Detect.forward 会从这里生成 one2many 正常特征和 one2one 的 detached 副本。
        return super().forward(self._prepare_features(x))
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
    if "MSDGSDetect" in text[match.start():block_end]:
        return text, False
    replacement = match.group(1) + "                MSDGSDetect,\n"
    return text[:match.start()] + replacement + text[match.end():], True


def inject_legacy_set(text: str) -> tuple[str, bool]:
    pattern = re.compile(r"(if m in \{)([^\n]*\bDetect\b[^\n]*)(\}:\s*\n\s*m\.legacy = legacy)")
    match = pattern.search(text)
    if not match:
        raise RuntimeError("cannot locate Detect legacy set")
    if "MSDGSDetect" in match.group(2):
        return text, False
    replacement = match.group(1) + "MSDGSDetect, " + match.group(2) + match.group(3)
    return text[:match.start()] + replacement + text[match.end():], True


def verify(tasks_text: str, init_text: str) -> None:
    assert "from .yolo26_msdgs_detect import MSDGSDetect" in init_text
    assert "from ultralytics.nn.modules.yolo26_msdgs_detect import MSDGSDetect" in tasks_text
    match = re.search(r"elif m in frozenset\(\s*\{(.*?)\n\s*\}\s*\n\s*\):", tasks_text, re.S)
    assert match and "MSDGSDetect" in match.group(1), "MSDGSDetect absent from Detect parser set"
    legacy = re.search(r"if m in \{([^\n]+)\}:\s*\n\s*m\.legacy = legacy", tasks_text)
    assert legacy and "MSDGSDetect" in legacy.group(1), "MSDGSDetect absent from legacy set"
    assert "args.extend([reg_max, end2end, [ch[x] for x in f]])" in tasks_text


def main() -> int:
    import ultralytics

    package = Path(ultralytics.__file__).resolve().parent
    module_file = package / "nn" / "modules" / "yolo26_msdgs_detect.py"
    init_file = package / "nn" / "modules" / "__init__.py"
    tasks_file = package / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("package", package)
    module_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", module_file)

    init_text = init_file.read_text(encoding="utf-8")
    init_text, init_changed = append_once(
        init_text,
        "yolo26_msdgs_detect",
        "# MSDGS P5-directed Detect\nfrom .yolo26_msdgs_detect import MSDGSDetect  # noqa: E402,F401\n",
    )
    if init_changed:
        init_file.write_text(init_text, encoding="utf-8")

    backup = tasks_file.with_name(tasks_file.name + ".msdgs_detect_bak")
    if not backup.exists():
        shutil.copy2(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tasks_text = tasks_file.read_text(encoding="utf-8")
    tasks_text, import_changed = append_once(
        tasks_text,
        "yolo26_msdgs_detect",
        "# MSDGS P5-directed Detect\n"
        "from ultralytics.nn.modules.yolo26_msdgs_detect import MSDGSDetect  # noqa: E402,F401\n",
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
    print("INSTALL_MSDGS_DETECT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())