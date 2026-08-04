#!/usr/bin/env python
"""诊断 iter36 方案A 崩盘根因：核对 yolo26n.pt 的 reg_max、以及 MSDGS 基线权重的 reg_max。

方案A(MSDGS + reg_max=1->4)训完 test map50 只有 0.237(vs MSDGS 基线 0.732)，
且 box/mAP 全程钉死不涨、cls_loss 正常下降。怀疑：
1. yolo26n.pt 官方权重 reg_max=? 若=16，则 .load() 时 DFL head 形状(16 vs 4)不匹配被丢弃；
2. 关键对照：MSDGS135eq 基线(reg_max=1)当初也是从 yolo26n.pt 迁移、也 250e，却能到 0.732。
   两者唯一差别就是 reg_max。若 reg_max=1 时 box head 输出 4 通道能从 yolo26n 迁移，
   而 reg_max=4 时输出 16 通道无法迁移 => 说明 reg_max=4 的 box head 完全从零学。
3. 更要命的可能：DFL proj 冻结 + 小数据，reg_max=4 分布回归学不动。

本脚本只读，不训练。
"""
from __future__ import annotations

import torch

ROOT = "/root/autodl-tmp/neu-det-yolo26"


def inspect(path, tag):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] load failed: {exc!r}")
        return
    m = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    head = m.model[-1]
    print(f"[{tag}] class={type(head).__name__} reg_max={getattr(head, 'reg_max', '?')} "
          f"nc={getattr(head, 'nc', '?')} no={getattr(head, 'no', '?')}")
    # box head 最后一层输出通道
    try:
        cv2_last = head.cv2[0][-1]
        print(f"[{tag}] cv2[0] last conv: in={cv2_last.in_channels} out={cv2_last.out_channels}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] cv2 inspect failed: {exc!r}")
    # DFL
    dfl = getattr(head, "dfl", None)
    print(f"[{tag}] dfl type={type(dfl).__name__}")


def main():
    inspect(f"{ROOT}/yolo26n.pt", "yolo26n_official")
    inspect(f"{ROOT}/runs_msdgs_gsdown_e250/y26n_gsdown_msdgs_135eq_e250/weights/best.pt",
            "msdgs135eq_baseline")
    inspect(f"{ROOT}/runs_dfl4_gsdown_e250/msdgs_dfl4/weights/best.pt", "A_dfl4_result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
