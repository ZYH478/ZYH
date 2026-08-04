#!/usr/bin/env python
"""iter39-B：把 Triplet Attention 装进 ultralytics 包，供 YAML 引用（channel-preserving）。

来源：MSAF-YOLO (Measurement 2026, S0263224125019992) 引用的 Triplet Attention
(Misra et al., WACV 2021, "Rotate to Attend")。在 MSAF 的消融里它是唯一「涨精度又降参」
的模块（+1.3 mAP 且 params 2.46M→1.9M）。

机理：三分支 rotate-to-attend。
- 分支1(H)：绕 H 轴旋转 -> Z-Pool(max||avg) -> 7x7 conv -> BN -> sigmoid -> 加权 -> 转回。
- 分支2(W)：绕 W 轴旋转，同上。
- 分支3(C)：不旋转，直接在原维度上 Z-Pool + conv + sigmoid 加权。
- 等权(1/3)融合三分支，捕捉 (C,H)/(C,W)/(H,W) 三对维度的跨维交互。
几乎零参数（仅 3 个 7x7 单通道卷积核），对方向性缺陷(scratches/rolled-in)友好。

落点（MSDGS 的 neck 全是 MSDGS 模块、无 C3k2，故不能像 MSAF 插进 C3k2）：
- 改为在三个检测分支输入前（layer 16/19/22 的 MSDGS 输出后）各追加一个 TripletAttention，
  Detect 从新的索引取。channel-preserving，走 parse_model 的 else 分支 c2=ch[f]，YAML args=[]。

TripletAttention 签名：TripletAttention(c1)。channel-preserving，注册到 base_modules。
前置依赖：先跑 install_gsconv_modules.py + install_msdgs_module.py。
安全：首次运行前备份 tasks.py 为 tasks.py.triplet_bak；幂等。成功打印 INSTALL_TRIPLET_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated Triplet Attention for YOLO26 (Misra et al., WACV 2021).
# Injected by install_triplet_module.py. Do not edit by hand.
import torch
import torch.nn as nn

__all__ = ["TripletAttention"]


class _ZPool(nn.Module):
    """沿通道维拼接 max 和 mean，得到 2 通道描述子。"""

    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)), dim=1
        )


class _AttnGate(nn.Module):
    """Z-Pool -> k x k conv(2->1) -> BN -> sigmoid，产生空间注意力图并加权。"""

    def __init__(self, k=7):
        super().__init__()
        self.pool = _ZPool()
        self.conv = nn.Conv2d(2, 1, k, 1, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        a = torch.sigmoid(self.bn(self.conv(self.pool(x))))
        return x * a


class TripletAttention(nn.Module):
    """Rotate-to-attend 三分支跨维注意力，channel-preserving，几乎零参。

    分支1：绕 H 轴 permute (B,C,H,W)->(B,H,C,W)，在 (C,W) 面算注意力后转回。
    分支2：绕 W 轴 permute ->(B,W,H,C)，在 (H,C) 面算注意力后转回。
    分支3：原维度，在 (H,W) 面算通道注意力。
    等权 1/3 融合。
    """

    def __init__(self, k=7):
        super().__init__()
        self.gate_h = _AttnGate(k)
        self.gate_w = _AttnGate(k)
        self.gate_c = _AttnGate(k)

    def forward(self, x):
        # 分支1：绕 H 轴（交换 C 与 H）
        x1 = x.permute(0, 2, 1, 3).contiguous()
        x1 = self.gate_h(x1).permute(0, 2, 1, 3).contiguous()
        # 分支2：绕 W 轴（交换 C 与 W）
        x2 = x.permute(0, 3, 2, 1).contiguous()
        x2 = self.gate_w(x2).permute(0, 3, 2, 1).contiguous()
        # 分支3：原维度通道注意力
        x3 = self.gate_c(x)
        return (x1 + x2 + x3) / 3.0
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    """在 base_modules frozenset 内插入 TripletAttention（channel-preserving，只需进 base）。"""
    m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}", text, re.S)
    if not m:
        return text, False
    if "TripletAttention" in m.group(0):
        return text, False
    # 锚在 C2PSA, 后插入（C2PSA 也是 channel-preserving，同类）
    new_text = text.replace(
        "            C2PSA,\n",
        "            C2PSA,\n            TripletAttention,\n",
        1,
    )
    return new_text, new_text != text


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_triplet.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_triplet",
        "\n# Triplet Attention (rotate-to-attend, WACV 2021)\n"
        "from .yolo26_triplet import TripletAttention  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".triplet_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_triplet",
        "\n# Triplet Attention for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_triplet import TripletAttention  # noqa: E402,F401\n",
    )
    # NOTE: TripletAttention is channel-preserving (args=[]), must go through
    # parse_model's final `else: c2 = ch[f]` branch instantiated as m() with no args.
    # It must NOT be in base_modules (that branch requires args[0] for c2). Import only.
    if imp_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", "SKIPPED (channel-preserving, else branch)")

    print("INSTALL_TRIPLET_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
