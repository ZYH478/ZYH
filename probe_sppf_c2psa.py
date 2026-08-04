#!/usr/bin/env python
"""iter23 探针：确认 SPPF / C2PSA / Conv 的构造签名，为 AGSPP/LLKBM/FEM 设计接口。

远程用法：
    python probe_sppf_c2psa.py
"""
from __future__ import annotations

import inspect

from ultralytics.nn.modules.block import SPPF, C2PSA, C2f, C3k2
from ultralytics.nn.modules.conv import Conv

for name, cls in [("SPPF", SPPF), ("C2PSA", C2PSA), ("C2f", C2f), ("C3k2", C3k2), ("Conv", Conv)]:
    print(f"=== {name} sig ===")
    print(inspect.signature(cls.__init__))

print("=== SPPF source ===")
print(inspect.getsource(SPPF))
print("=== C3k2 source ===")
print(inspect.getsource(C3k2))
