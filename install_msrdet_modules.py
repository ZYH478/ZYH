#!/usr/bin/env python
"""iter23：把 MSR-Det 论文（Bian et al., RCIM 2026）的三个模块注入 ultralytics 8.4.93。

论文 = "A lightweight detection network integrating multi-scale semantic refinement
for steel strip defects"（MSR-Det）。owner 指定把其中三个模块叠到轻量交付模型
vovgscsp_gsdown 上做对比。三个模块：

1. LLKBM（Lightweight Large-Kernel Bottleneck Module，论文 4.5 / Fig.6）：
   从 C2f 派生，把 bottleneck 重构成三段 `1x1 -> 大核 depthwise -> 1x1`。
   论文只在 backbone 最后两个 C2f 用，大核 {7,9} 最优（Table 4）。
   -> gsdown backbone 层 6(k=7) / 层 8(k=9) 的 C3k2 替换。
   继承 C2f，需进 base_modules + repeat_modules。

2. AGSPP（Adaptive Gating Spatial Pyramid Pooling，论文 4.6，图内写 AGSP）：
   替换 SPPF。1x1 压缩 -> 3 个级联同核 MaxPool -> 可学习门控权重
   w_hat_i = SiLU(w_i)/(sum SiLU(w_j)+eps) 加权 -> concat(y0,w1y1,w2y2,w3y3) -> 1x1 投影。
   签名对齐 SPPF(c1,c2,k=5,n=3,shortcut=False)，进 base_modules（单输入，不 insert n）。
   -> gsdown backbone 层 9 的 SPPF 替换。

3. FEM（Feature Enhancement Module，论文 4.4 / Fig.5）：
   双输入 (F_L=P3 浅层, F_H=C5 深层语义)。用 C5 经 reshape 到 P3 空间尺寸后，
   做通道注意力 A=softmax(F_L' . F_H'^T) in R^{CxC}，再 F_L~=A . F_L''，
   残差 P3' = T(F_L~) + F_L。深->浅语义引导精炼浅层。
   双输入模块，parse_model 需专门分支：c1=[ch[16],ch[10]]，c2=args[0]（=P3 通道）。

注入范式对齐 install_backbone_modules.py：
- 写模块文件 yolo26_msrdet.py 到 nn/modules/
- 注入 __init__.py 导出 + tasks.py import
- LLKBM -> base_modules + repeat_modules
- AGSPP -> base_modules（单输入不 insert n）
- FEM   -> 需在 parse_model 里加双输入分支（本脚本会打补丁）

幂等：重复运行不重复注入。首次改 tasks.py 前备份 tasks.py.msrdet_bak。

远程用法：
    python install_msrdet_modules.py    # 幂等
成功打印 INSTALL_MSRDET_MODULES_OK。
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


MODULE_SRC = '''# Auto-generated MSR-Det modules (LLKBM / AGSPP / FEM) for YOLO26 iter23.
# Paper: Bian et al., "A lightweight detection network integrating multi-scale
# semantic refinement for steel strip defects", RCIM 2026.
# Injected by install_msrdet_modules.py. Do not edit by hand.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f

__all__ = ["LLKBM", "AGSPP", "FEM"]


class _LLKBMBottleneck(nn.Module):
    """Large-kernel bottleneck: 1x1 cross-channel fuse -> large-kernel depthwise
    spatial extraction -> 1x1 remap (paper 4.5, Fig.6). Residual when c1==c2.
    """

    def __init__(self, c1, c2, k=7, e=0.5, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)                       # 1x1 cross-channel fusion
        # large-kernel depthwise conv for efficient spatial feature extraction
        self.dw = Conv(c_, c_, k, 1, g=c_)
        self.cv2 = Conv(c_, c2, 1, 1)                       # 1x1 feature remap
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.dw(self.cv1(x)))
        return x + y if self.add else y


class LLKBM(C2f):
    """Lightweight Large-Kernel Bottleneck Module (paper 4.5).

    Derived from C2f: keeps the CSP split/concat structure, only replaces the inner
    Bottleneck with a three-stage large-kernel-depthwise bottleneck. Drop-in for a
    backbone C3k2/C2f. YAML: [-1, n, LLKBM, [c2, k, shortcut]] (k defaults 7).
    """

    def __init__(self, c1, c2, n=1, k=7, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            _LLKBMBottleneck(self.c, self.c, k=k, e=1.0, shortcut=shortcut) for _ in range(n)
        )


class AGSPP(nn.Module):
    """Adaptive Gating Spatial Pyramid Pooling (paper 4.6).

    Replaces SPPF. 1x1 channel compression -> 3 cascaded same-kernel MaxPools ->
    learnable gating rebalances the 3 pooled maps -> concat(y0, w1*y1, w2*y2, w3*y3)
    -> 1x1 projection to c2. Signature mirrors SPPF(c1, c2, k, n, shortcut).
    """

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.n = int(n)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * (self.n + 1), c2, 1, 1)
        # learnable gating weights over the n pooled branches (paper Eq.19).
        self.gate = nn.Parameter(torch.ones(self.n))
        self.eps = 1e-4
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(self.m(y[-1]))
        # normalized SiLU gating over the pooled branches (y[1:]); y[0] kept as-is.
        g = F.silu(self.gate)
        g = g / (g.sum() + self.eps)
        fused = [y[0]] + [g[i] * y[i + 1] for i in range(self.n)]
        out = self.cv2(torch.cat(fused, 1))
        return out + x if self.add else out


class FEM(nn.Module):
    """Feature Enhancement Module (paper 4.4, Fig.5).

    Dual input (F_L = shallow P3, F_H = deep C5). Uses deep semantic C5 to refine
    shallow P3 via channel-wise attention A = softmax(F_L' . F_H'^T) in R^{CxC},
    then F_L_tilde = A . F_L'', residual P3' = proj(F_L_tilde) + F_L.

    parse_model passes c1 as a list [ch_PL, ch_PH]; c2 is the output (=P3) channels.
    YAML: [[P3_idx, C5_idx], 1, FEM, [c2]].
    """

    def __init__(self, c1, c2, mid=None):
        super().__init__()
        cl, ch = (c1[0], c1[1]) if isinstance(c1, (list, tuple)) else (c1, c1)
        cm = mid or c2
        self.cm = cm
        # scale for dot-product attention (1/sqrt(N)); softmax over C-dim affinity
        # of length-N inner products, so normalize by sqrt(N) to keep logits bounded.
        # N is spatial (H*W) and is large (e.g. 6400 at P3), so an unscaled bmm
        # overflows and diverges to nan within a few epochs (iter23 first attempt).
        # project F_L and F_H into a shared cm-dim latent (aligned to P3 spatial size).
        self.proj_l = Conv(cl, cm, 1, 1)         # F_L' = G(F_L)
        self.proj_h = Conv(ch, cm, 1, 1)         # F_H' = G(R(F_H)) (R = reshape via interp)
        self.proj_l2 = Conv(cl, cm, 1, 1)        # F_L'' = G(F_L)
        self.out = Conv(cm, c2, 1, 1)            # restore / project to c2
        # residual path aligns F_L channels to c2 when they differ.
        self.res = Conv(cl, c2, 1, 1) if cl != c2 else nn.Identity()
        # learnable residual gain, init 0 so training starts from clean gsdown+AGSPP
        # and the attention branch ramps in gradually (stabilizes early epochs).
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f_l, f_h = x[0], x[1]
        h, w = f_l.shape[-2:]
        # reshape deep feature to shallow spatial dims (paper R(.)).
        if f_h.shape[-2:] != (h, w):
            f_h = F.interpolate(f_h, size=(h, w), mode="bilinear", align_corners=False)
        b, n = f_l.shape[0], h * w
        # L2-normalize along the N (spatial) dim + 1/sqrt(N) scale so the C x C
        # affinity logits stay O(1) regardless of resolution (prevents nan).
        fl = F.normalize(self.proj_l(f_l).reshape(b, self.cm, -1), dim=2)   # C x N
        fh = F.normalize(self.proj_h(f_h).reshape(b, self.cm, -1), dim=2)   # C x N
        # channel-wise affinity A in R^{C x C}, prioritized by deep feature.
        logits = torch.bmm(fl, fh.transpose(1, 2)) / (self.cm ** 0.5)       # b, C, C
        attn = torch.softmax(logits, dim=-1)
        fl2 = self.proj_l2(f_l).reshape(b, self.cm, -1)   # C x N
        enhanced = torch.bmm(attn, fl2).reshape(b, self.cm, h, w)        # A . F_L''
        # gated residual: P3' = gamma * proj(A . F_L'') + F_L  (gamma init 0)
        return self.gamma * self.out(enhanced) + self.res(f_l)
'''


FEM_BRANCH = (
    "        elif m is FEM:  # MSR-Det dual-input feature enhancement (paper 4.4)\n"
    "            c2 = args[0]\n"
    "            c2 = make_divisible(min(c2, max_channels) * width, 8) if c2 != nc else c2\n"
    "            args = [[ch[x] for x in f], c2, *args[1:]]\n"
)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text.rstrip() + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    """在 base_modules frozenset 内追加 LLKBM / AGSPP。

    FEM 是双输入模块，绝不能进 base_modules：parse_model 的 `if m in base_modules`
    分支会先跑 `c1, c2 = ch[f], args[0]`，而 FEM 的 f 是 list [P3,C5] 会 TypeError。
    FEM 只由专门的 `elif m is FEM` 分支处理（见 inject_fem_branch）。
    """
    m = re.search(r"base_modules = frozenset\(\s*\{(?P<body>.*?)\n\s*\}\n\s*\)\n\s*repeat_modules", text, re.S)
    if not m:
        raise RuntimeError("Could not locate parse_model base_modules block")
    body = m.group("body")
    to_add = [x for x in ("LLKBM", "AGSPP") if x not in body]
    if not to_add:
        return text, False
    start, end = m.span("body")
    insert = "".join(f"\n            {x}," for x in to_add)
    # 锚定 SPDConv 后插入（gsconv/backbone 注入已保证其存在）。
    body_new, n = re.subn(r"(?m)^(\s*SPDConv,\s*)$", r"\1" + insert, body, count=1)
    if n != 1:
        body_new, n = re.subn(r"(?m)^(\s*A2C2f,\s*)$", r"\1" + insert, body, count=1)
    if n != 1:
        raise RuntimeError("Could not insert MSR-Det modules into base_modules")
    return text[:start] + body_new + text[end:], True


def inject_repeat_modules(text: str) -> tuple[str, bool]:
    """LLKBM 继承 C2f 且吃 n -> 进 repeat_modules。AGSPP/FEM 不进（单输入/双输入无 n）。"""
    m = re.search(r"repeat_modules = frozenset\([^{]*\{(?P<body>.*?)\n\s*\}\n\s*\)", text, re.S)
    if not m:
        raise RuntimeError("Could not locate parse_model repeat_modules block")
    body = m.group("body")
    if "LLKBM" in body:
        return text, False
    start, end = m.span("body")
    body_new, n = re.subn(r"(?m)^(\s*C3k2,\s*)$", r"\1\n            LLKBM,", body, count=1)
    if n != 1:
        raise RuntimeError("Could not insert LLKBM into repeat_modules")
    return text[:start] + body_new + text[end:], True


def inject_fem_branch(text: str) -> tuple[str, bool]:
    """在 parse_model 里为双输入 FEM 加专门分支（放在 Concat 分支之后）。"""
    if "m is FEM:" in text:
        return text, False
    # 锚点：Concat 的 c2 = sum(ch[x] for x in f) 分支之后插入 FEM 分支。
    anchor = "        elif m is Concat:\n            c2 = sum(ch[x] for x in f)\n"
    if anchor not in text:
        raise RuntimeError("Could not locate Concat branch anchor for FEM injection")
    return text.replace(anchor, anchor + FEM_BRANCH, 1), True


def install() -> dict:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    modules_dir = pkg / "nn" / "modules"
    module_path = modules_dir / "yolo26_msrdet.py"
    init_path = modules_dir / "__init__.py"
    tasks_path = pkg / "nn" / "tasks.py"

    module_path.write_text(MODULE_SRC.lstrip(), encoding="utf-8")

    init_src = _read(init_path)
    init_src, init_changed = inject_import(
        init_src, "yolo26_msrdet import",
        "\n# MSR-Det iter23 modules\nfrom .yolo26_msrdet import LLKBM, AGSPP, FEM  # noqa: E402,F401\n",
    )
    if init_changed:
        _write(init_path, init_src)

    backup = tasks_path.with_name(tasks_path.name + ".msrdet_bak")
    if not backup.exists():
        shutil.copy(tasks_path, backup)
        print("backup_created", backup)

    tk = _read(tasks_path)
    tk, imp_changed = inject_import(
        tk, "yolo26_msrdet import",
        "\n\n# MSR-Det iter23 modules injected by install_msrdet_modules.py\n"
        "from ultralytics.nn.modules.yolo26_msrdet import LLKBM, AGSPP, FEM\n",
    )
    tk, base_changed = inject_base_modules(tk)
    tk, rep_changed = inject_repeat_modules(tk)
    tk, fem_changed = inject_fem_branch(tk)
    if imp_changed or base_changed or rep_changed or fem_changed:
        _write(tasks_path, tk)

    return {
        "module_path": str(module_path),
        "init_changed": init_changed,
        "tasks_import_changed": imp_changed,
        "base_modules_changed": base_changed,
        "repeat_modules_changed": rep_changed,
        "fem_branch_changed": fem_changed,
    }


def main() -> int:
    try:
        info = install()
    except Exception as exc:  # noqa: BLE001
        print(f"INSTALL_MSRDET_MODULES_FAILED: {exc}", file=sys.stderr)
        return 1
    for k, v in info.items():
        print(f"{k}: {v}")
    print("INSTALL_MSRDET_MODULES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
