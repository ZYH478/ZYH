#!/usr/bin/env python
"""远程构建校验：只实例化 stage2 winner-combo 的每个 YAML，不训练。

目的：在启动 250e 训练前，确认 SPD 双层插入后 backbone 层数、Concat 索引、
Detect 输入都正确，且模型能 build + 一次前向，避免 iter7/iter11 那种
"训练跑起来才崩" 的浪费。

对每个候选：
1. 生成 YAML（复用 sweep.dump_yaml）。
2. YOLO(yaml).load(weights) 实例化并加载预训练权重（验证权重迁移不报错）。
3. 断言 end2end=True / reg_max=1。
4. 单次 dummy forward（640x640），打印输出结构、参数量、检测尺度数。
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

import train_yolo26_module_sweep as sweep
import train_yolo26_stage2_winner_combo as wc


def main() -> int:
    wc.set_sweep_outputs()
    sweep.ensure_modules()
    specs = wc.winner_combo_specs()
    weights = str(sweep.WEIGHTS)

    from ultralytics import YOLO

    all_ok = True
    for name, spec in specs.items():
        print(f"\n===== BUILD_CHECK {name} =====")
        try:
            cfg = sweep.dump_yaml(name, spec["doc"])
            model = YOLO(str(cfg)).load(weights)
            core = model.model
            det = core.model[-1]
            end2end = getattr(det, "end2end", None)
            reg_max = getattr(det, "reg_max", None)
            nl = getattr(det, "nl", None)
            n_layers = len(core.model)
            params = int(sum(p.numel() for p in core.parameters()))
            core.eval()
            with torch.no_grad():
                out = core(torch.zeros(1, 3, 640, 640))
            # end2end 推理输出结构：dict 或 tensor
            if isinstance(out, dict):
                shapes = {k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in out.items()}
            elif isinstance(out, (list, tuple)):
                shapes = [x.shape if hasattr(x, "shape") else type(x).__name__ for x in out]
            else:
                shapes = out.shape
            assert end2end is True, f"end2end != True (got {end2end})"
            assert reg_max == 1, f"reg_max != 1 (got {reg_max})"
            print(f"OK end2end={end2end} reg_max={reg_max} nl={nl} n_layers={n_layers} params={params}")
            print(f"   forward_out={shapes}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"FAIL {name}: {exc!r}")

    print("\nBUILD_CHECK_ALL_OK" if all_ok else "\nBUILD_CHECK_HAS_FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
