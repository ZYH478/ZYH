#!/usr/bin/env python
"""iter20：在 vovgscsp_gsdown 的 backbone 上叠加 DWR 改进（owner 指定方向）。

背景：iter19 发现 DWRC3k2（深层 C3k2 空洞残差）在 base backbone 上把精度提到 0.4246
（追平精度赢家 winner）且更轻（2.217M）、crazing 弱类新高 0.2024。owner 要求把这个
DWR 改进叠到 gsdown 上，看能否在保持 gsdown 极致轻量（1.936M/-18.5%）的同时吃进
DWR 的精度增益。

关键结构事实（已核实）：
- gsdown 的 backbone（层 0-10）与 base 完全相同，其轻量化改动全在 head
  （VoVGSCSP neck 层 13/16/19/22 + GSConv 下采样 层 17/20）。
- DWR 改的是 backbone 深层 C3k2（层 6/8）。两者改动区域不重叠 → 可干净正交叠加。

做法：直接加载已验证正确的 gsdown YAML，只把 backbone 层 6/8 的 C3k2 换成 DWRC3k2，
其余一字不动，保证 head 结构绝对正确。从 yolo26n.pt 迁移，保 end2end/reg_max=1。

对照：gsdown 独立真值 val mAP50-95 0.4077 / 1.936M（轻量化轴交付模型）。
判定：若 gsdown_dwr 真值 > 0.4077 且 fused params 守住 ~1.936M（DWR 比标准 C3k2 更省参，
应该守得住）→ 在极致轻量前提下提精度，轻量+精度双升单模型。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py
    python install_backbone_modules.py    # DCNv2Conv/PKIC3k2/DWRC3k2
    python -u train_gsdown_dwr.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
GSDOWN_YAML = ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_gsdown_dwr_e250"
GEN_DIR = ROOT / "generated_models_gsdown_dwr_e250"
NAME = "y26n_gsdown_dwr_e250"
DWR_LAYERS = (6, 8)  # 深层 C3k2 -> DWRC3k2


def build_doc() -> dict:
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    bb = doc["backbone"]
    swapped = []
    for i in DWR_LAYERS:
        if bb[i][2] == "C3k2":
            bb[i][2] = "DWRC3k2"
            swapped.append(i)
    if not swapped:
        raise SystemExit(f"no C3k2 found at layers {DWR_LAYERS}; backbone={[(i,r[2]) for i,r in enumerate(bb)]}")
    print(f"SWAPPED C3k2->DWRC3k2 at backbone layers {swapped}")
    return doc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT.mkdir(parents=True, exist_ok=True)
    doc = build_doc()
    cfg_path = GEN_DIR / f"{NAME}.yaml"
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg_path}")
    if args.dry:
        return 0

    print(f"INIT_WEIGHTS {NAME} <- {OFFICIAL_WEIGHTS}")
    model = YOLO(str(cfg_path)).load(str(OFFICIAL_WEIGHTS))
    print(f"TRAIN_START {NAME} epochs={args.epochs} batch={args.batch} imgsz={args.imgsz}")
    model.train(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=8, seed=args.seed, device=0, project=str(PROJECT), name=NAME,
        exist_ok=True, patience=max(args.epochs, 250), cache=False, verbose=True,
    )
    print(f"TRAIN_DONE {NAME}")

    # 同进程 val/test（记录用，真值以独立进程 per_class.py 为准）
    weights = PROJECT / NAME / "weights" / "best.pt"
    best = YOLO(str(weights))
    v = best.val(data=str(DATA), split="val", imgsz=args.imgsz, batch=args.batch, device=0,
                 project=str(PROJECT), name=f"{NAME}_val", exist_ok=True, verbose=False)
    t = best.val(data=str(DATA), split="test", imgsz=args.imgsz, batch=args.batch, device=0,
                 project=str(PROJECT), name=f"{NAME}_test", exist_ok=True, verbose=False)
    out = {
        "name": NAME,
        "val_map50": float(v.box.map50), "val_map50_95": float(v.box.map),
        "test_map50": float(t.box.map50), "test_map50_95": float(t.box.map),
        "weights": str(weights), "cfg": str(cfg_path),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (PROJECT / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("GSDOWN_DWR_RESULT", json.dumps(out, ensure_ascii=False))
    print(f"DONE report={PROJECT / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
