#!/usr/bin/env python
"""iter35 build check（不训练）：验证 SADGS 三候选能 build/forward/fuse，核对参数量。

检查项：
1. YAML 能被 parse_model 正确构建（SADGS 已注册进 base_modules）。
2. Detect head 的 from 索引不变（纯 neck 替换，backbone/结构拓扑不动）。
3. forward 一张假图不报错，输出 shape 合理。
4. fuse 后能 forward（条带分支 dw_h/dw_v + bn 可正常合并/推理）。
5. fused 参数量打印，主候选应对齐 MSDGS ~1.777M。
6. 打印每个 SADGS 块的 splits，确认通道均等三/四分。

用法（远程）：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py
    python install_sadgs_module.py
    python build_check_sadgs.py
成功打印 BUILD_CHECK_SADGS_OK。
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
GSDOWN_YAML = ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"
GEN_DIR = ROOT / "generated_models_sadgs_build_check"

CANDIDATES = {
    "gsdown_sadgs_13s7": {"dilations": [1, 3], "strip_k": 7, "fracs": [1, 1, 1]},
    "gsdown_sadgs_13s11": {"dilations": [1, 3], "strip_k": 11, "fracs": [1, 1, 1]},
    "gsdown_sadgs_135s7": {"dilations": [1, 3, 5], "strip_k": 7, "fracs": [1, 1, 1, 1]},
}


def build_doc(dilations, strip_k, fracs) -> dict:
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    n_swapped = 0
    for block in doc.get("head", []):
        if len(block) >= 4 and block[2] == "VoVGSCSP":
            c2 = block[3][0]
            block[2] = "SADGS"
            block[3] = [c2, True, 1, 0.5, list(dilations), int(strip_k), list(fracs)]
            n_swapped += 1
    assert n_swapped == 4, f"expected 4 VoVGSCSP, got {n_swapped}"
    return doc


def check_one(name: str, spec: dict) -> None:
    print(f"\n===== {name} spec={spec} =====")
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    doc = build_doc(spec["dilations"], spec["strip_k"], spec["fracs"])
    cfg = GEN_DIR / f"{name}.yaml"
    cfg.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    model = YOLO(str(cfg), task="detect")
    net = model.model

    # Detect head from-index（纯 neck 替换应与 gsdown 一致：[16,19,22]）
    detect = net.model[-1]
    print("detect_from", detect.f, "detect_class", type(detect).__name__)

    # 打印每个 SADGS 块的 splits
    from ultralytics.nn.modules.yolo26_sadgs import SADGS
    for i, m in enumerate(net.model):
        if isinstance(m, SADGS):
            print(f"  layer[{i}] SADGS splits={m.splits} dils={m.dils} n_iso={m.n_iso} strip_k={m.strip_k}")

    # forward（unfused）
    net.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = net(x)
    print("forward_ok unfused")

    params_unfused = sum(p.numel() for p in net.parameters())

    # fuse + forward
    model.fuse()
    with torch.no_grad():
        _ = net(x)
    params_fused = sum(p.numel() for p in net.parameters())
    print("forward_ok fused")
    print(f"params_unfused={params_unfused} params_fused={params_fused} "
          f"({params_fused/1e6:.4f}M)  vs MSDGS 1.777M / gsdown 1.936M")


def main() -> int:
    print("GSDOWN_YAML", GSDOWN_YAML, "exists", GSDOWN_YAML.exists())
    for name, spec in CANDIDATES.items():
        check_one(name, spec)
    print("\nBUILD_CHECK_SADGS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
