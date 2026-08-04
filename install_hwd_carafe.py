#!/usr/bin/env python
"""给 ultralytics 8.4.93 注入两个信息保真类结构改进模块：HWD + CARAFE。

设计对齐 install_yolo26_exp_modules.py（SPDConv/DySample 同款注入范式）：
- 写模块文件 yolo26_struct.py 到 nn/modules/
- 注入 __init__.py 导出 + tasks.py import
- HWD 改通道 → 进 base_modules frozenset（同 SPDConv）
- CARAFE 通道保持（c-preserving）→ 不进 base_modules（同 DySample）

两模块都契合本 goal 唯一有效规律「信息保真类有效、重加权类无效」：

1. HWD（Haar 小波下采样，Xu et al. PR2023）：
   用固定 Haar 小波变换替代 stride-2 有损卷积做下采样。输入经 DWT 分解为
   4 个子带（LL 低频 + LH/HL/HH 高频边缘/纹理），空间减半、通道 4×C，拼接后
   接 Conv 降回目标通道。与已验证有效的 SPDConv 同族（无损下采样），但显式保留
   高频子带，直击 crazing/rolled-in_scale 弱纹理缺陷（判别信息集中在高频）。
   固定 Haar 核（register_buffer，无需 pytorch_wavelets 依赖，免 AutoDL 环境事故）。

2. CARAFE（内容感知上采样，ICCV2019）：
   根据输入内容动态预测重组核做上采样，大感受野聚合上下文。与已验证有效的
   DySample 同族（内容感知上采样）。纯 PyTorch 实现（unfold + 加权求和，
   免 mmcv CUDA 依赖）。通道保持，作 nn.Upsample 的对照替换。

远程用法：
    python install_hwd_carafe.py    # 幂等
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


MODULE_SRC = '''# Auto-generated info-preserving structural modules for YOLO26 (NEU-DET iter18).
# Injected by install_hwd_carafe.py. Do not edit by hand.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class HWD(nn.Module):
    """Haar Wavelet Downsampling. Info-preserving stride-2 replacement.

    DWT decomposes input into 4 subbands (LL/LH/HL/HH) at half resolution via a
    fixed Haar filter bank, concatenates to 4*c1 channels, then a Conv projects to c2.
    Unlike strided conv (low-pass, lossy on high freq), HWD explicitly keeps the
    LH/HL/HH high-frequency edge/texture subbands -- directly targets weak-texture
    steel defects (crazing / rolled-in_scale). Same family as validated SPDConv.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        # After Haar DWT: channels become 4*c1 at half res; Conv projects to c2 (stride 1).
        self.conv = Conv(4 * c1, c2, k, 1, p, g, act=act)
        self.register_buffer("haar", self._haar_kernel(), persistent=False)

    @staticmethod
    def _haar_kernel():
        # 4 Haar 2x2 analysis filters (LL, LH, HL, HH), normalized by 0.5.
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        # (4,1,2,2)
        return torch.stack([ll, lh, hl, hh], 0).unsqueeze(1)

    def forward(self, x):
        b, c, h, w = x.shape
        # depthwise Haar DWT: each channel -> 4 subbands via grouped stride-2 conv.
        k = self.haar.to(dtype=x.dtype, device=x.device)  # (4,1,2,2)
        k = k.repeat(c, 1, 1, 1)  # (4c,1,2,2), groups=c
        # pad odd H/W to even so 2x2 stride-2 tiling is exact.
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2))
        y = F.conv2d(x, k, stride=2, groups=c)  # (b, 4c, h/2, w/2)
        # reorder from [c0_ll,c0_lh,c0_hl,c0_hh, c1_ll,...] grouping to [all_ll, all_lh, ...]
        y = y.view(b, c, 4, y.shape[-2], y.shape[-1]).permute(0, 2, 1, 3, 4).reshape(b, 4 * c, y.shape[-2], y.shape[-1])
        return self.conv(y)


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures upsampling (ICCV2019), pure-PyTorch.

    Predicts a content-dependent reassembly kernel per output location and does a
    large-receptive-field weighted reassembly. Same family as validated DySample
    (content-aware upsampling). Channel-preserving. No mmcv/CUDA dependency.

    Lazy-built (like DySample): the channel-compression conv is created on the first
    forward from the actual input channels, so parse_model treats it as c-preserving
    (c2=ch[f]) and the YAML only needs to pass [scale] -- no c1 needed.
    """

    def __init__(self, scale=2, k_up=5, k_enc=3, c_mid=64):
        super().__init__()
        self.scale = int(scale)
        self.k_up = int(k_up)
        self.k_enc = int(k_enc)
        self.c_mid = int(c_mid)
        self.pix_shf = nn.PixelShuffle(self.scale)
        self.softmax = nn.Softmax(dim=1)
        self._built = False
        self.comp = None
        self.enc = None

    def _build(self, c, device=None, dtype=None):
        self.comp = nn.Conv2d(c, self.c_mid, 1)
        self.enc = nn.Conv2d(self.c_mid, (self.scale ** 2) * (self.k_up ** 2), self.k_enc, padding=self.k_enc // 2)
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1], device=x.device, dtype=x.dtype)
            self._built = True
        b, c, h, w = x.shape
        s, ku = self.scale, self.k_up
        # kernel prediction: (b, s^2*ku^2, h, w) -> pixelshuffle -> (b, ku^2, s*h, s*w)
        W = self.enc(self.comp(x))
        W = self.pix_shf(W)
        W = self.softmax(W)  # normalize over ku^2 taps
        # unfold input into ku^2 neighborhood, upsample-nearest to output res, weighted-sum.
        X = F.unfold(x, kernel_size=ku, padding=ku // 2)  # (b, c*ku^2, h*w)
        X = X.view(b, c, ku * ku, h, w)
        X = F.interpolate(X.view(b, c * ku * ku, h, w), scale_factor=s, mode="nearest")
        X = X.view(b, c, ku * ku, s * h, s * w)
        out = (X * W.unsqueeze(1)).sum(dim=2)  # (b, c, s*h, s*w)
        return out


_STRUCT_EXPORTS = {"HWD": HWD, "CARAFE": CARAFE}
'''


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def install() -> dict:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    modules_dir = pkg / "nn" / "modules"
    module_path = modules_dir / "yolo26_struct.py"
    tasks_path = pkg / "nn" / "tasks.py"
    init_path = modules_dir / "__init__.py"

    module_path.write_text(MODULE_SRC.lstrip(), encoding="utf-8")

    init_src = _read(init_path)
    init_changed = False
    if "yolo26_struct import HWD" not in init_src:
        init_src = init_src.rstrip() + (
            "\n# yolo26 iter18 structural modules\n"
            "from .yolo26_struct import HWD, CARAFE  # noqa: E402,F401\n"
        )
        _write(init_path, init_src)
        init_changed = True

    tasks_src = _read(tasks_path)
    tasks_changed = False
    if "yolo26_struct import HWD" not in tasks_src:
        tasks_src = tasks_src.rstrip() + (
            "\n\n# YOLO26 iter18 structural modules injected by install_hwd_carafe.py\n"
            "from ultralytics.nn.modules.yolo26_struct import HWD, CARAFE\n"
        )
        tasks_changed = True

    # HWD changes channels -> must be in base_modules (like SPDConv). CARAFE is c-preserving -> not added.
    base_match = re.search(r"base_modules = frozenset\(\s*\{(?P<body>.*?)\n\s*\}\n\s*\)\n\s*repeat_modules", tasks_src, re.S)
    if not base_match:
        raise RuntimeError("Could not locate parse_model base_modules block")
    if "HWD" not in base_match.group("body"):
        start, end = base_match.span("body")
        body = base_match.group("body")
        # insert HWD after SPDConv if present, else after A2C2f.
        body_new, n = re.subn(r"(?m)^(\s*SPDConv,\s*)$", r"\1\n            HWD,", body, count=1)
        if n != 1:
            body_new, n = re.subn(r"(?m)^(\s*A2C2f,\s*)$", r"\1\n            HWD,", body, count=1)
        if n != 1:
            raise RuntimeError("Could not insert HWD into base_modules")
        tasks_src = tasks_src[:start] + body_new + tasks_src[end:]
        tasks_changed = True

    # RepConv (native, already imported) used as a stride-2 downsample must be in
    # base_modules so parse_model injects c1 (args[0] -> c2). Without this,
    # RepConv(256,3,2) is mis-parsed as c1=256,c2=3,k=2 -> assert k==3 fails.
    # Re-locate the block since HWD insertion above may have shifted it.
    base_match2 = re.search(
        r"base_modules = frozenset\(\s*\{(?P<body>.*?)\n\s*\}\n\s*\)\n\s*repeat_modules",
        tasks_src, re.S,
    )
    if not base_match2:
        raise RuntimeError("Could not locate parse_model base_modules block (RepConv)")
    if not re.search(r"(?m)^\s*RepConv,\s*$", base_match2.group("body")):
        start, end = base_match2.span("body")
        body = base_match2.group("body")
        # insert RepConv right after the plain 'Conv,' entry.
        body_new, n = re.subn(r"(?m)^(\s*Conv,\s*)$", r"\1\n            RepConv,", body, count=1)
        if n != 1:
            raise RuntimeError("Could not insert RepConv into base_modules")
        tasks_src = tasks_src[:start] + body_new + tasks_src[end:]
        tasks_changed = True

    if tasks_changed:
        _write(tasks_path, tasks_src)

    return {
        "module_path": str(module_path),
        "init_changed": init_changed,
        "tasks_changed": tasks_changed,
    }


def main() -> int:
    try:
        info = install()
    except Exception as exc:  # noqa: BLE001
        print(f"INSTALL_HWD_CARAFE_FAILED: {exc}", file=sys.stderr)
        return 1
    for k, v in info.items():
        print(f"{k}: {v}")
    print("INSTALL_HWD_CARAFE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
