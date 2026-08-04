#!/usr/bin/env python
"""核实 ultralytics 8.4.93 C3k2 的 __init__ 签名 + self.c/self.m 属性，供 iter19 子类对齐。"""
import inspect
from ultralytics.nn.modules.block import C3k2

print("SIG", inspect.signature(C3k2.__init__))
import torch
m = C3k2(64, 128, 2, False, 0.25)
print("HAS_c", hasattr(m, "c"), "c=", getattr(m, "c", None))
print("HAS_m", hasattr(m, "m"), "len_m=", len(m.m) if hasattr(m, "m") else None)
x = torch.randn(1, 64, 32, 32)
print("OUT", tuple(m(x).shape))
