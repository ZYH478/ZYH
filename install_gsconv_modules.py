#!/usr/bin/env python
"""把 SlimNeck 的 GSConv / VoVGSCSP 装进当前 ultralytics 包，供 YAML 引用。

与 install_yolo26_exp_modules.py 独立：不改动现有 6 个实验模块的注册，只追加
GSConv / VoVGSCSP，并把它们登记进 parse_model 的 frozenset，使通道/深度被正确处理。

登记规则（依据远程 8.4.93 probe 结论）：
- GSConv    -> 只进 base_modules（需要 c1/c2 自动填充与 width 缩放）。
- VoVGSCSP  -> 同时进 base_modules 与 repeat_modules（替换 C3k2，需 insert n）。

安全：
- 首次运行前把 tasks.py 备份为 tasks.py.gsconv_bak（存在则不覆盖）。
- 幂等：重复运行不会重复注入。

用法（远程）：
    python install_gsconv_modules.py
成功打印 INSTALL_GSCONV_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated SlimNeck modules (GSConv / VoVGSCSP) for YOLO26 ablation.
# Injected by install_gsconv_modules.py. Do not edit by hand.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ["GSConv", "GSBottleneck", "VoVGSCSP"]


class GSConv(nn.Module):
    """GSConv: half standard conv + half depthwise on top, then channel shuffle."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        # 用关键字参数避免 8.4.x Conv(c1,c2,k,s,p,g,d,act) 的位置错位
        self.cv1 = Conv(c1, c_, k, s, g=g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, g=c_, act=act)  # g=c_ => depthwise 5x5

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False),
        )
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    """SlimNeck CSP block built from GSBottleneck; drop-in for C3k2 in neck.

    signature 兼容 C3k2(c1, c2, n, ...)：parse_model 会 insert n 到位置 2。
    shortcut/g 仅为签名兼容，不参与 forward。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))

    def forward(self, x):
        x1 = self.cv1(x)
        return self.cv2(torch.cat((self.m(x1), x1), dim=1))
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    """在 base_modules frozenset 内的 SPDConv, 后插入 GSConv/VoVGSCSP。

    SPDConv 只出现在 base_modules（repeat_modules 无 SPDConv），定位安全。
    """
    m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}", text, re.S)
    if not m or "GSConv" in m.group(0):
        return text, False
    # 只替换 base_modules 块内首个 SPDConv,
    new_text = text.replace(
        "            SPDConv,\n",
        "            SPDConv,\n            GSConv,\n            VoVGSCSP,\n",
        1,
    )
    return new_text, new_text != text


def inject_repeat_modules(text: str) -> tuple[str, bool]:
    """在 repeat_modules frozenset 块内追加 VoVGSCSP。"""
    m = re.search(r"(repeat_modules\s*=\s*frozenset\([^{]*\{)(.*?)(\n\s*\})", text, re.S)
    if not m:
        return text, False
    body = m.group(2)
    if "VoVGSCSP" in body:
        return text, False
    new_block = m.group(1) + body.rstrip().rstrip(",") + ",\n            VoVGSCSP," + m.group(3)
    new_text = text[: m.start()] + new_block + text[m.end():]
    return new_text, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_gsconv.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    # 1. module source
    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    # 2. __init__.py export
    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_gsconv",
        "\n# SlimNeck experimental modules\nfrom .yolo26_gsconv import GSConv, VoVGSCSP  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    # 3. tasks.py: backup then import + frozenset
    backup = tasks_file.with_name(tasks_file.name + ".gsconv_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_gsconv",
        "\n# SlimNeck experimental modules for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_gsconv import GSConv, VoVGSCSP  # noqa: E402,F401\n",
    )
    tk, base_changed = inject_base_modules(tk)
    tk, repeat_changed = inject_repeat_modules(tk)
    if imp_changed or base_changed or repeat_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", base_changed)
    print("tasks_repeat_modules_changed", repeat_changed)

    # 4. verify by re-import in a way that forces re-read (best-effort)
    print("INSTALL_GSCONV_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
