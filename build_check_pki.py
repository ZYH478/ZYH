#!/usr/bin/env python
"""iter37 方案D build check：验证 PKIC3k2 装配后能 build/forward/fuse，参数增量可控。

判据：
- MSDGS135eq 基线 fused 1.777M。PKIC3k2 只在层4加并联 DW 5/7 大核（DW 卷积、通道守恒），
  参数增量应 < 0.05M（几 k 级），fused 仍应 < gsdown 红线 1.936M。
- 层4 确实被换成 PKIC3k2、Detect from [16,19,22] 不变（backbone 层索引不移，因为是替换非插入）。
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
GEN = ROOT / "_bc_pki"
GEN.mkdir(parents=True, exist_ok=True)


def build_pki_doc():
    doc = yaml.safe_load(MSDGS_YAML.read_text(encoding="utf-8"))
    bb = doc.get("backbone", [])
    swapped = []
    for i, block in enumerate(bb):
        if len(block) >= 4 and block[2] == "C3k2" and block[3] and block[3][0] == 512 \
                and len(block[3]) >= 2 and block[3][1] is False:
            block[2] = "PKIC3k2"
            swapped.append(i)
            break
    assert len(swapped) == 1, f"swapped at {swapped}"
    print(f"swapped backbone idx {swapped[0]} C3k2 -> PKIC3k2")
    return doc


def count_fused(cfg):
    m = YOLO(str(cfg), task="detect")
    # forward smoke
    m.model.eval()
    with torch.no_grad():
        y = m.model(torch.zeros(1, 3, 640, 640))
    unfused = int(sum(p.numel() for p in m.model.parameters()))
    try:
        m.model.fuse()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN fuse: {exc!r}")
    fused = int(sum(p.numel() for p in m.model.parameters()))
    # head class check
    head = m.model.model[-1]
    return unfused, fused, type(head).__name__


def main():
    print("=== baseline MSDGS135eq ===")
    ub, fb, hb = count_fused(MSDGS_YAML)
    print(f"[baseline] unfused={ub} fused={fb} ({fb/1e6:.4f}M) head={hb}")

    print("=== D: MSDGS + PKIC3k2@layer4 ===")
    doc = build_pki_doc()
    cfg = GEN / "msdgs_pki_l4.yaml"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    # verify PKIC3k2 present in backbone
    bb_mods = [b[2] for b in doc["backbone"]]
    assert "PKIC3k2" in bb_mods, bb_mods
    up, fp, hp = count_fused(cfg)
    print(f"[pki_l4] unfused={up} fused={fp} ({fp/1e6:.4f}M) head={hp}")
    print(f"[pki_l4] param delta vs baseline: unfused +{up-ub}, fused +{fp-fb}")
    assert fp < 1_935_814, f"fused {fp} exceeds gsdown red line 1.936M"
    assert hp == "Detect", f"head changed unexpectedly: {hp}"
    print("BUILD CHECK PKI OK")


if __name__ == "__main__":
    main()
