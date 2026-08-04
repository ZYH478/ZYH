#!/usr/bin/env python
"""只读 dump parse_model 的分支逻辑 + frozen_c1_first 定义，供本地分析注入点。"""
import os
import ultralytics

p = os.path.join(os.path.dirname(ultralytics.__file__), "nn", "tasks.py")
s = open(p, encoding="utf-8").read()

i = s.find("if m in base_modules")
print("===PARSE_MODEL_BRANCHES===")
print(s[i - 100:i + 2600])

print("\n===FROZEN_C1_FIRST_DEF===")
j = s.find("frozen_c1_first")
while j != -1:
    print(f"--- at {j} ---")
    print(s[j - 120:j + 200])
    j = s.find("frozen_c1_first", j + 1)
