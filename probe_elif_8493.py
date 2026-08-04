#!/usr/bin/env python
"""抓取 8.4.93 parse_model 的 elif 分支链，找 FEM 多输入分支的注入锚点。只读。"""
from __future__ import annotations

import os
import re

import ultralytics

p = os.path.join(os.path.dirname(ultralytics.__file__), "nn", "tasks.py")
t = open(p, encoding="utf-8").read()

# 定位 parse_model 里 base_modules 大 if 结束后到 Detect 之间的 elif 链
i = t.find("c1, c2 = ch[f], args[0]")
j = t.find("m.legacy = legacy")
if i > 0 and j > i:
    seg = t[i:j+200]
    # 只打印每个 elif/if 行 + Concat/上采样等关键分支
    for line in seg.split("\n"):
        s = line.strip()
        if s.startswith("elif ") or s.startswith("if m is") or "ch[x] for x in f" in s or "c2 = sum" in s or s.startswith("args ="):
            print(repr(line))
print("=== END ===")
