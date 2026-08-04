#!/usr/bin/env python
"""Dump full parse_model source + base/repeat frozensets to design FEM (dual-input) module correctly.

只读。重点：多输入模块(from=[a,b])时 c1 如何取；base_modules 里 c2 如何算；args 插入 n 的规则。
"""
import inspect
from pathlib import Path

import ultralytics
from ultralytics.nn import tasks

print("ultralytics_version", ultralytics.__version__)
print("tasks_path", Path(tasks.__file__))
print("=" * 60)
src = inspect.getsource(tasks.parse_model)
print(src)
