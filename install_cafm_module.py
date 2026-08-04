#!/usr/bin/env python
"""iter39-A：把 Channel-Aware Feature Mixer (CAFM) 装进 ultralytics 包，drop-in 替换 C2PSA。

来源：MSAF-YOLO (Measurement 2026, S0263224125019992)，在 YOLOv11n(1.9M) 上验证有效
(+CAFM 使 mAP0.5 从 80.5→81.2)。CAFM 放 backbone 末端第 10 层，替换原 C2PSA。

为什么这条路对本 goal 有意义（不是重走死路）：
- CAFM 的核心 CAKA = 三条并行 depthwise 分支：方形 KxK + 横条 1xM + 竖条 Mx1（M=3K+2），
  再用「通道自适应动态门控」(GAP→1x1→softmax) 加权融合。
- 这正是 owner iter35 SADGS「各向异性条带解耦」的**升级版**：SADGS 是静态并联相加，
  CAFM 多了动态通道门控这一关键机制，且在 1.9M 量级验证过。方向性/各向异性是 neck/backbone
  末端从未真正试过的正交轴，针对 rolled-in(带状)/scratches(细长) 这类可恢复的方向性缺陷。
- C2PSA 是各向同性全局自注意力；CAFM 是各向异性条带 + 通道门控。干净的单变量替换消融。

CAFM 与 C2PSA 签名兼容(drop-in)：CAFM(c1, c2, n=1, e=0.5)，通道保持 c1==c2。
注册到 C2PSA 所在的两个 frozenset(base_modules + repeat_modules)，锚点 `C2PSA,`。

前置依赖：先跑 install_gsconv_modules.py + install_msdgs_module.py。
安全：首次运行前备份 tasks.py 为 tasks.py.cafm_bak；幂等。成功打印 INSTALL_CAFM_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated Channel-Aware Feature Mixer (CAFM) for YOLO26.
# Injected by install_cafm_module.py. Source: MSAF-YOLO (Measurement 2026).
# Do not edit by hand.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["CAFM"]


class _CAKA(nn.Module):
    """Channel-Aware Kernel Adaptor：三条并联 depthwise 分支（方形 KxK / 横条 1xM / 竖条 Mx1），
    用全局通道描述子动态生成三组 softmax 权重加权融合，再 BN+SiLU。

    - 方形分支：常规局部感受野。
    - 横条 1xM (M=3K+2)：沿宽度长程，捕捉横向裂纹/带状纹理。
    - 竖条 Mx1：沿高度长程，捕捉纵向划痕。
    - 动态门控：G=GAP(X) -> 1x1 conv 出 3C -> reshape(3,B,C,1,1) -> softmax(dim=0)，
      各分支通道级自适应权重，和 SADGS 静态相加相比多了内容自适应。
    """

    def __init__(self, c_ch, k=3):
        super().__init__()
        m = 3 * k + 2  # 条带核长，保证 k 小时条带仍够长
        ph_sq = k // 2
        ph_m = m // 2
        self.dw_sq = nn.Conv2d(c_ch, c_ch, k, 1, ph_sq, groups=c_ch, bias=False)
        self.dw_hor = nn.Conv2d(c_ch, c_ch, (1, m), 1, (0, ph_m), groups=c_ch, bias=False)
        self.dw_ver = nn.Conv2d(c_ch, c_ch, (m, 1), 1, (ph_m, 0), groups=c_ch, bias=False)
        # 通道门控：GAP -> 1x1 conv 输出 3*c_ch 通道
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Conv2d(c_ch, 3 * c_ch, 1, 1, 0, bias=True)
        self.bn = nn.BatchNorm2d(c_ch)
        self.act = nn.SiLU()
        self.c_ch = c_ch

    def forward(self, x):
        f_sq = self.dw_sq(x)
        f_hor = self.dw_hor(x)
        f_ver = self.dw_ver(x)
        g = self.gate(self.gap(x))  # (B, 3C, 1, 1)
        b = x.shape[0]
        g = g.view(3, b, self.c_ch, 1, 1)
        a = torch.softmax(g, dim=0)  # 三分支通道级权重，和为 1
        fused = a[0] * f_sq + a[1] * f_hor + a[2] * f_ver
        return self.act(self.bn(fused))


class CAFM(nn.Module):
    """Channel-Aware Feature Mixer：通道分半后走两个不同核 CAKA(K=5 / K=3)，
    concat 回来再 1x1 卷积做跨通道交互。drop-in 替换 C2PSA（通道保持 c1==c2）。

    - K=5 分支：长程连续缺陷（宽皱褶、大氧化斑）。
    - K=3 分支：微缺陷（细裂纹、点蚀），避免大核引入背景干扰。
    n/e 仅为签名兼容（C2PSA 有 n/e），本模块结构不依赖 n。
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, f"CAFM expects c1==c2 (drop-in C2PSA), got {c1}!={c2}"
        c = c1
        c_half = c // 2
        c_rem = c - c_half
        self.c_half = c_half
        self.c_rem = c_rem
        self.caka_big = _CAKA(c_half, k=5)   # 大核分支
        self.caka_small = _CAKA(c_rem, k=3)  # 小核分支
        self.fuse = Conv(c, c2, 1, 1)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.c_half, self.c_rem], dim=1)
        y1 = self.caka_big(x1)
        y2 = self.caka_small(x2)
        return self.fuse(torch.cat((y1, y2), dim=1))
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_after_c2psa(text: str, frozenset_name: str) -> tuple[str, bool]:
    """在指定 frozenset 块内的 `C2PSA,` 后插入 `CAFM,`（仅该 frozenset 内首个匹配）。"""
    m = re.search(rf"({frozenset_name}\s*=\s*frozenset\([^{{]*\{{)(.*?)(\n\s*\}})", text, re.S)
    if not m:
        return text, False
    body = m.group(2)
    if "CAFM" in body:
        return text, False
    new_body = body.replace("C2PSA,\n", "C2PSA,\n            CAFM,\n", 1)
    if new_body == body:
        return text, False
    new_text = text[: m.start()] + m.group(1) + new_body + m.group(3) + text[m.end():]
    return new_text, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_cafm.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_cafm",
        "\n# Channel-Aware Feature Mixer (MSAF-YOLO)\nfrom .yolo26_cafm import CAFM  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".cafm_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_cafm",
        "\n# Channel-Aware Feature Mixer for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_cafm import CAFM  # noqa: E402,F401\n",
    )
    tk, base_changed = inject_after_c2psa(tk, "base_modules")
    tk, repeat_changed = inject_after_c2psa(tk, "repeat_modules")
    if imp_changed or base_changed or repeat_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)

    print("INSTALL_CAFM_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
