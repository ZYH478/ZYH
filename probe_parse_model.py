#!/usr/bin/env python
"""探明远程 ultralytics 的 parse_model 如何处理自定义模块的 c1/c2/n。

重点回答：
1. 现有 SPDConv / DySample 是否在 base_modules / repeat_modules frozenset 里？
2. parse_model 里 base_modules、repeat_modules 的定义位置与成员。
3. C3k2 是否两个集合都在。

只读，不训练。
"""
import inspect
import re
from pathlib import Path

import ultralytics
from ultralytics.nn import tasks

print("ultralytics_version", ultralytics.__version__)
tasks_path = Path(tasks.__file__)
print("tasks_path", tasks_path)

src = inspect.getsource(tasks.parse_model)
# 找出 base_modules / repeat_modules 定义块
for key in ("base_modules", "repeat_modules"):
    print(f"\n===== {key} occurrences =====")
    for m in re.finditer(rf"{key}\s*=\s*frozenset", src):
        start = m.start()
        # 打印该定义到下一个 ) 结束的片段
        snippet = src[start:start + 1200]
        # 截到第一个独立的 })  或 )\n
        end = snippet.find("}")
        print(snippet[:end + 2] if end > 0 else snippet[:600])
        break

# 检查关键模块名是否出现在 parse_model 源码里
print("\n===== module name mentions in parse_model =====")
for name in ("SPDConv", "DySample", "C3k2", "GSConv", "VoVGSCSP", "A2C2f"):
    print(f"{name:<12} in_parse_model={name in src}")

# base_modules / repeat_modules 里逐个模块是否出现（粗匹配）
print("\n===== membership hints (regex around the frozenset) =====")
bm = re.search(r"base_modules\s*=\s*frozenset\((.*?)\)\s*\n", src, re.S)
rm = re.search(r"repeat_modules\s*=\s*frozenset\((.*?)\)\s*\n", src, re.S)
for label, mm in (("base_modules", bm), ("repeat_modules", rm)):
    if mm:
        body = mm.group(1)
        for name in ("SPDConv", "DySample", "C3k2", "GSConv", "VoVGSCSP", "Conv"):
            print(f"{label:<16}{name:<12}{name in body}")
    else:
        print(f"{label}: NOT FOUND via regex (may be built differently)")
