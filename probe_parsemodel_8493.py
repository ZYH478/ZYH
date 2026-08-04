#!/usr/bin/env python
"""探查远程 8.4.93 的 parse_model：base/repeat frozenset、多输入模块(Detect等)通道解析分支，
为注入 GatedSPPF/C2fk/FEM 找准注入点。只读，不改任何文件。"""
from __future__ import annotations

import os
import re

import ultralytics

p = os.path.join(os.path.dirname(ultralytics.__file__), "nn", "tasks.py")
t = open(p, encoding="utf-8").read()
print("VER", ultralytics.__version__)
print("TASKS", p)

# 1. base_modules frozenset
m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}\s*\)", t, re.S)
print("=== base_modules ===")
print(m.group(0) if m else "NOT FOUND")

# 2. repeat_modules frozenset
m = re.search(r"repeat_modules\s*=\s*frozenset\(.*?\)", t, re.S)
print("=== repeat_modules ===")
print(m.group(0) if m else "NOT FOUND")

# 3. 多输入模块通道解析：找含 'ch[x] for x in f' 的分支（Detect / Concat 等）
print("=== multi-input branches (ch[x] for x in f) ===")
for mm in re.finditer(r".{0,120}ch\[x\] for x in f.{0,120}", t):
    print("...", mm.group(0).replace("\n", "\\n"))

# 4. Detect args 追加处 & end2end 处理
print("=== Detect append region ===")
i = t.find("args.append([ch[x] for x in f])")
if i > 0:
    print(t[i-400:i+200])

# 5. GSConv/VoVGSCSP 是否已注入
print("=== gsconv present ===", "yolo26_gsconv" in t)
