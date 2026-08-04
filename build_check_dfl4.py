#!/usr/bin/env python
"""iter36 build check：验证方案 A(DFL4) / B(BoxHead32+DFL4) 能 build/forward/fuse，
并核对 fused 参数量、reg_max、DFL 是否激活、box head 宽度。

用法（远程）：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py
    python install_msdgs_module.py
    python install_boxhead32_module.py
    python build_check_dfl4.py
成功打印 BUILD_CHECK_DFL4_OK。
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
MSDGS_YAML = ROOT / "generated_models_multiseed_msdgs_vs_gsdown" / "msdgs135eq.yaml"
GEN = ROOT / "generated_models_dfl4_gsdown_e250"


def _load_doc() -> dict:
    return yaml.safe_load(MSDGS_YAML.read_text(encoding="utf-8"))


def build_A() -> dict:
    doc = _load_doc()
    doc["reg_max"] = 4
    return doc


def build_B() -> dict:
    doc = _load_doc()
    doc["reg_max"] = 4
    for block in doc["head"]:
        if len(block) >= 3 and block[2] == "Detect":
            block[2] = "BoxHead32Detect"
    return doc


def fused_params(cfg_path: Path) -> tuple[int, object]:
    model = YOLO(str(cfg_path), task="detect")
    m = model.model
    m.eval()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        _ = m(x)
    m.fuse()
    with torch.no_grad():
        _ = m(x)
    params = int(sum(p.numel() for p in m.parameters()))
    return params, m


def report(tag: str, doc: dict) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    cfg = GEN / f"_bc_{tag}.yaml"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    # 先用未 fuse 的模型看结构
    model = YOLO(str(cfg), task="detect")
    det = model.model.model[-1]
    assert isinstance(det, Detect), f"{tag}: head not Detect subclass: {type(det)}"
    reg_max = det.reg_max
    dfl_active = det.dfl.__class__.__name__
    # box head 宽度 = cv2[0] 第一个 Conv 的输出通道
    box_w = det.cv2[0][0].conv.out_channels
    box_out = det.cv2[0][-1].out_channels  # 应 = 4*reg_max
    print(f"[{tag}] head_class={det.__class__.__name__} reg_max={reg_max} "
          f"dfl={dfl_active} box_head_width={box_w} box_out={box_out}")
    assert reg_max == 4, f"{tag}: reg_max != 4"
    assert dfl_active == "DFL", f"{tag}: DFL not active (got {dfl_active})"
    assert box_out == 4 * reg_max, f"{tag}: box_out {box_out} != 4*reg_max"
    params, _ = fused_params(cfg)
    print(f"[{tag}] fused_params={params} ({params/1e6:.4f}M)")
    cfg.unlink(missing_ok=True)


def main() -> int:
    print("=== MSDGS baseline (reg_max=1) reference ===")
    doc0 = _load_doc()
    print(f"baseline reg_max={doc0.get('reg_max')}")

    print("=== A: MSDGS + DFL4 (reg_max=1->4, nothing else) ===")
    docA = build_A()
    report("A_dfl4", docA)

    print("=== B: MSDGS + BoxHead32 + DFL4 ===")
    docB = build_B()
    report("B_boxhead32_dfl4", docB)

    print("BUILD_CHECK_DFL4_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
