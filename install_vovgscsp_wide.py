#!/usr/bin/env python
"""iter24：注入可配隐藏通道比的 VoVGSCSPW（松开 gsdown neck 压缩瓶颈）。

背景：iter22 诊断 gsdown 掉点集中在强类（inclusion -3.5pp / patches -1.5pp），
根因 = VoVGSCSP/GSConv 把 neck 通道压到 0.5 太狠，丰富特征传不过去。前 12 次
失败全是"往紧 neck 上叠料"。本轮是首个直接松开瓶颈本身的方案：不叠新模块，
只把已有 VoVGSCSP 的隐藏通道比从固定 0.5 放宽为可配 ratio。

VoVGSCSPW 与 install_gsconv_modules.py 的 VoVGSCSP 结构完全一致，只把
`c_ = int(c2 * 0.5)` 改成 `c_ = int(c2 * ratio)`，ratio 由 YAML 传入。
不改动已验证的 VoVGSCSP（隔离风险）。

登记规则（对齐 VoVGSCSP）：进 base_modules + repeat_modules。
YAML: [-1, n, VoVGSCSPW, [c2, ratio]]
  args=[c2, ratio] -> base_module 取 args[0]=c2 -> args=[c1,c2,ratio]
  -> repeat_modules insert n 到位置2 -> [c1,c2,n,ratio]，匹配签名。

依赖：install_gsconv_modules.py 必须先跑（VoVGSCSPW 复用其 GSConv/GSBottleneck）。

幂等：重复运行不重复注入。首次改 tasks.py 前备份 tasks.py.vovwide_bak。

远程用法：
    python install_gsconv_modules.py     # 先装 GSConv/GSBottleneck/VoVGSCSP
    python install_vovgscsp_wide.py       # 本轮
成功打印 INSTALL_VOVGSCSP_WIDE_OK。
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


MODULE_SRC = '''# Auto-generated widened VoVGSCSP (configurable hidden ratio) for YOLO26 iter24.
# Loosens gsdown neck channel compression. Injected by install_vovgscsp_wide.py.
# Do not edit by hand.
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.yolo26_gsconv import GSBottleneck

__all__ = ["VoVGSCSPW"]


class VoVGSCSPW(nn.Module):
    """VoVGSCSP with configurable hidden channel ratio (default 0.75).

    Identical structure to VoVGSCSP except the CSP hidden width `c_` = int(c2*ratio)
    is exposed instead of fixed 0.5. Higher ratio loosens neck compression so richer
    features (strong defect classes) pass through. Drop-in for a neck VoVGSCSP.
    signature 兼容 repeat_modules：parse_model insert n 到位置 2 -> (c1,c2,n,ratio).
    """

    def __init__(self, c1, c2, n=1, ratio=0.75, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * ratio)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))

    def forward(self, x):
        x1 = self.cv1(x)
        return self.cv2(torch.cat((self.m(x1), x1), dim=1))
'''


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text.rstrip() + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    m = re.search(r"base_modules = frozenset\(\s*\{(?P<body>.*?)\n\s*\}\n\s*\)\n\s*repeat_modules", text, re.S)
    if not m:
        raise RuntimeError("Could not locate parse_model base_modules block")
    body = m.group("body")
    if "VoVGSCSPW" in body:
        return text, False
    start, end = m.span("body")
    # 锚定 VoVGSCSP 后插入（install_gsconv_modules 已保证其存在）。
    body_new, n = re.subn(r"(?m)^(\s*VoVGSCSP,\s*)$", r"\1\n            VoVGSCSPW,", body, count=1)
    if n != 1:
        raise RuntimeError("Could not insert VoVGSCSPW into base_modules (VoVGSCSP anchor not found)")
    return text[:start] + body_new + text[end:], True


def inject_repeat_modules(text: str) -> tuple[str, bool]:
    m = re.search(r"repeat_modules = frozenset\([^{]*\{(?P<body>.*?)\n\s*\}\n\s*\)", text, re.S)
    if not m:
        raise RuntimeError("Could not locate parse_model repeat_modules block")
    body = m.group("body")
    if "VoVGSCSPW" in body:
        return text, False
    start, end = m.span("body")
    body_new, n = re.subn(r"(?m)^(\s*VoVGSCSP,\s*)$", r"\1\n            VoVGSCSPW,", body, count=1)
    if n != 1:
        raise RuntimeError("Could not insert VoVGSCSPW into repeat_modules (VoVGSCSP anchor not found)")
    return text[:start] + body_new + text[end:], True


def install() -> dict:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    modules_dir = pkg / "nn" / "modules"
    module_path = modules_dir / "yolo26_vovwide.py"
    init_path = modules_dir / "__init__.py"
    tasks_path = pkg / "nn" / "tasks.py"

    module_path.write_text(MODULE_SRC.lstrip(), encoding="utf-8")

    init_src = _read(init_path)
    init_src, init_changed = inject_import(
        init_src, "yolo26_vovwide import",
        "\n# VoVGSCSPW iter24 widened neck\nfrom .yolo26_vovwide import VoVGSCSPW  # noqa: E402,F401\n",
    )
    if init_changed:
        _write(init_path, init_src)

    backup = tasks_path.with_name(tasks_path.name + ".vovwide_bak")
    if not backup.exists():
        shutil.copy(tasks_path, backup)
        print("backup_created", backup)

    tk = _read(tasks_path)
    tk, imp_changed = inject_import(
        tk, "yolo26_vovwide import",
        "\n\n# VoVGSCSPW iter24 injected by install_vovgscsp_wide.py\n"
        "from ultralytics.nn.modules.yolo26_vovwide import VoVGSCSPW\n",
    )
    tk, base_changed = inject_base_modules(tk)
    tk, rep_changed = inject_repeat_modules(tk)
    if imp_changed or base_changed or rep_changed:
        _write(tasks_path, tk)

    return {
        "module_path": str(module_path),
        "init_changed": init_changed,
        "tasks_import_changed": imp_changed,
        "base_modules_changed": base_changed,
        "repeat_modules_changed": rep_changed,
    }


def main() -> int:
    try:
        info = install()
    except Exception as exc:  # noqa: BLE001
        print(f"INSTALL_VOVGSCSP_WIDE_FAILED: {exc}", file=sys.stderr)
        return 1
    for k, v in info.items():
        print(f"{k}: {v}")
    print("INSTALL_VOVGSCSP_WIDE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
