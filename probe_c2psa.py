#!/usr/bin/env python
"""探明 C2PSA 的签名与 frozenset 归属，确认 CAFM 注入点（替换 backbone 末端 layer10 的 C2PSA）。"""
import inspect
import re
from pathlib import Path

import ultralytics
from ultralytics.nn.modules import C2PSA

print("ultralytics", ultralytics.__version__)
print("C2PSA sig:", inspect.signature(C2PSA.__init__))

tk = (Path(ultralytics.__file__).resolve().parent / "nn" / "tasks.py").read_text(encoding="utf-8")
for name in ["base_modules", "repeat_modules"]:
    m = re.search(name + r"\s*=\s*frozenset\(.*?\}", tk, re.S)
    if m:
        print(name, "| C2PSA in block:", "C2PSA" in m.group(0), "| PSA, in block:", "PSA," in m.group(0))

print("--- lines mentioning C2PSA/PSA ---")
for i, l in enumerate(tk.splitlines()):
    s = l.strip()
    if "C2PSA" in s or re.search(r"\bPSA\b", s):
        print(i, s[:100])
