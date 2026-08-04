#!/usr/bin/env python
"""探查 8.4.93 的 SPPF / C3k2 __init__ 签名，确认 gsdown 的 SPPF[1024,5,3,true] 如何解析，
为移植 GatedSPPF(仿 SPPF drop-in) / C2fk(仿 C3k2) 找准签名。只读。"""
from __future__ import annotations

import inspect

from ultralytics.nn.modules.block import SPPF, C3k2
from ultralytics.nn.modules.conv import Conv

print("SPPF.__init__", inspect.signature(SPPF.__init__))
print("SPPF source:")
print(inspect.getsource(SPPF.__init__))
print("C3k2.__init__", inspect.signature(C3k2.__init__))
print("Conv.__init__", inspect.signature(Conv.__init__))
