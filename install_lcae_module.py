#!/usr/bin/env python
"""把局部对比度自适应增强块 LCAE 装进当前 ultralytics 包，供 YAML 引用。

设计动机（iter32 混淆矩阵诊断坐实、五连败后换赛道）：
crazing 病根 = 前景/背景不可分：test 上 52% 的 GT crazing 判成背景（recall 0.379 全场最低）。
crazing 是低对比度弥散裂纹网络，前景相对局部背景的灰度差异微弱但存在。此前所有攻法
(iter29 backbone-DWR/SPD、iter30 DySnake、iter31 UBHead、iter32 FocalCW/FBCon) 均放在
**深层**(P3/P4/head/检测头)——而 crazing 的微弱前景信号在深层已被下采样平滑掉。

LCAE 与两条已证伪死路的关键分野：
- vs iter28 HFDGS(x - avgpool 全局高通，证伪：无差别放大噪声)：LCAE 多除一个局部标准差
  σ，做 divisive normalization。σ 小的低对比度区(crazing 前景)被相对放大，σ 大的高对比度区
  (inclusion/scratches 强边界)几乎不动 —— 只喂 crazing 需要的信号，不拖累强类。
- vs iter29 backbone(DWR/SPD，证伪)：LCAE 放在 backbone **最浅层 P2/4(layer 1 后)**，
  stride=4，crazing 微弱前景信号尚未被平滑；iter29 全放深层，在信号已丢失处做文章。

算子（借视觉皮层 divisive normalization）：
    mu    = avgpool_k(x)                    # 局部均值
    var   = avgpool_k(x^2) - mu^2           # 局部方差(clamp>=0)
    c     = (x - mu) / sqrt(var + eps)      # 归一化局部对比度(自适应)
    out   = x + gamma * c                   # gamma 可学习标量, 初始 0 = 恒等起步

参数量：pooling 无参 + 1 个 gamma 标量。近零开销，红线内。单输入单输出、通道不变
(drop-in 插入 backbone 任意位置)。

集成：backbone layer 1(第二个 Conv, P2/4 stride=4)之后插入一层 LCAE，其后所有层索引 +1，
head 里所有绝对引用(Concat from、Detect from)由 train 脚本 YAML 生成逻辑同步偏移。

parse_model 兼容：LCAE 不改通道数(c2=c1)，只需注册进 base_modules。args 仅 [k]（pooling
核大小，可空），c1 由 parse_model 自动传入。

安全：首次运行前备份 tasks.py 为 tasks.py.lcae_bak；幂等，重复运行不重复注入。

用法（远程）：
    python install_lcae_module.py
成功打印 INSTALL_LCAE_OK。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated local contrast adaptive enhancement module (LCAE) for YOLO26.
# Injected by install_lcae_module.py. Do not edit by hand.
import torch
import torch.nn as nn

__all__ = ["LCAE"]


class LCAE(nn.Module):
    """局部对比度自适应增强块（divisive normalization），drop-in 插入 backbone 任意层。

    单输入单输出，通道不变。对低对比度区(crazing 弥散前景)自适应放大，对高对比度区
    (强边界)几乎不动。参数量 = 1 个可学习标量 gamma(初始 0 = 恒等起步)。

        mu   = avgpool_k(x)
        var  = avgpool_k(x^2) - mu^2   (clamp >= 0)
        c    = (x - mu) / sqrt(var + eps)
        out  = x + gamma * c
    """

    def __init__(self, c1, c2=None, k=5, eps=1e-4):
        super().__init__()
        # c2 由 parse_model 传入(=c1)，此处不改通道；仅保留签名兼容
        self.k = int(k)
        self.eps = float(eps)
        self.pool = nn.AvgPool2d(self.k, stride=1, padding=self.k // 2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu = self.pool(x)
        var = self.pool(x * x) - mu * mu
        var = var.clamp_(min=0.0)
        c = (x - mu) / torch.sqrt(var + self.eps)
        return x + self.gamma * c
'''


def inject_import(text: str, marker: str, import_line: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + import_line, True


def inject_base_modules(text: str) -> tuple[str, bool]:
    """在 base_modules frozenset 内插入 LCAE。"""
    m = re.search(r"base_modules\s*=\s*frozenset\(.*?\}", text, re.S)
    if not m:
        return text, False
    if "LCAE" in m.group(0):
        return text, False
    new_text = text.replace(
        "            VoVGSCSP,\n",
        "            VoVGSCSP,\n            LCAE,\n",
        1,
    )
    if new_text == text:
        # 回退：VoVGSCSP 可能尚未注入，退而挂在 Conv 后
        new_text = text.replace(
            "            Conv,\n",
            "            Conv,\n            LCAE,\n",
            1,
        )
    return new_text, new_text != text


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_lcae.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_import(
        it, "yolo26_lcae",
        "\n# Local contrast adaptive enhancement module\nfrom .yolo26_lcae import LCAE  # noqa: E402,F401\n",
    )
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".lcae_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_import(
        tk, "yolo26_lcae",
        "\n# Local contrast adaptive enhancement module for custom YAMLs\n"
        "from ultralytics.nn.modules.yolo26_lcae import LCAE  # noqa: E402,F401\n",
    )
    tk, base_changed = inject_base_modules(tk)
    if imp_changed or base_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_base_modules_changed", base_changed)

    print("INSTALL_LCAE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
