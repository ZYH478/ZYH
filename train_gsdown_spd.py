#!/usr/bin/env python
"""iter22：在 vovgscsp_gsdown 上用 SPDConv 替换 backbone 下采样卷积（信息保真方向）。

背景与动机
----------
gsdown（轻量交付模型，1.936M/-18.5%，test mAP50 0.7338 / mAP50-95 0.4018）相对 base
掉点集中在强类（inclusion -3.5pp、patches -1.5pp），根因是 head 侧 GSConv 把 neck
通道压薄、强类丰富特征传不过去。gsdown 的 backbone 与 base 完全相同。

本 goal 唯一稳定有效的规律是"信息保真类"改进（SPDConv +2.56pp、DySample），
"重加权/增容类"（注意力/VFL/NWD/DWR叠gsdown）全部无效。SPDConv 与叠在 gsdown 上
失败的 DWR/PKI 机理不同：它不增加多尺度容量，而是把下采样时丢弃的空间细节无损
重排进通道（space-to-depth），恰好补"neck 信息不够"这个真痛点，且它是本 goal 产出
最大增益的模块（winner=SPD_P3+DySample），但从未在 gsdown 上单独试过。

做法
----
加载已验证正确的 gsdown YAML，把 backbone 的 stride-2 下采样 Conv 换成 SPDConv：
- gsdown_spd_p3   : 只换 P3 下采样（层 3，Conv[256,3,2] -> SPDConv[256,3,2]）——
                    winner 增益来自 SPD_P3，最保守、最可能守住轻量。
- gsdown_spd_p3p4 : 换 P3+P4 下采样（层 3 与 层 5）——力度更大，参数会涨一些。

其余一字不动，head 结构绝对正确。从 yolo26n.pt 迁移，保 end2end/reg_max=1。

对照：gsdown 独立真值 val mAP50-95 0.4077 / test 0.4018 / 1.936M。
判定：候选真值 > gsdown 且守住轻量优势（fused params 不明显超过 base 2.376M）
      → 在极致轻量前提下把强类精度拉回来。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py    # SPDConv 等实验模块
    python install_gsconv_modules.py         # GSConv/VoVGSCSP（gsdown head 需要）
    python -u train_gsdown_spd.py            # 两候选串行
    # 或单跑：python -u train_gsdown_spd.py --only gsdown_spd_p3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
GSDOWN_YAML = ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_gsdown_spd_e250"
GEN_DIR = ROOT / "generated_models_gsdown_spd_e250"

# gsdown backbone 下采样 Conv 所在层：层3=P3(256)、层5=P4(512)、层7=P5(1024)。
# 只换浅/中层，P5 保留（P5 换 SPD 参数暴涨且 winner 未从 P5 获益）。
CANDIDATES = {
    "gsdown_spd_p3": (3,),
    "gsdown_spd_p3p4": (3, 5),
}


def build_doc(spd_layers: tuple[int, ...]) -> dict:
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    bb = doc["backbone"]
    swapped = []
    for i in spd_layers:
        row = bb[i]
        # 期望是 [-1, 1, Conv, [c, 3, 2]] 的 stride-2 下采样
        if row[2] != "Conv":
            raise SystemExit(f"layer {i} is {row[2]}, not Conv; backbone={[(j,r[2]) for j,r in enumerate(bb)]}")
        cargs = row[3]
        if len(cargs) < 3 or cargs[2] != 2:
            raise SystemExit(f"layer {i} Conv args {cargs} not stride-2 downsample")
        bb[i][2] = "SPDConv"
        swapped.append((i, cargs))
    if not swapped:
        raise SystemExit(f"no downsample Conv swapped at layers {spd_layers}")
    print(f"SWAPPED Conv->SPDConv at backbone layers {swapped}")
    return doc


def train_one(tag: str, spd_layers: tuple[int, ...], args) -> dict:
    doc = build_doc(spd_layers)
    cfg_path = GEN_DIR / f"y26n_{tag}_e250.yaml"
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg_path}")
    if args.dry:
        # dry：仅 build 检查
        m = YOLO(str(cfg_path))
        info = m.model.info(verbose=False)
        print(f"DRY_BUILD_OK {tag} info={info}")
        return {"name": tag, "dry": True, "cfg": str(cfg_path)}

    name = f"y26n_{tag}_e250"
    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    model = YOLO(str(cfg_path)).load(str(OFFICIAL_WEIGHTS))
    print(f"TRAIN_START {name} epochs={args.epochs} batch={args.batch} imgsz={args.imgsz}")
    model.train(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=8, seed=args.seed, device=0, project=str(PROJECT), name=name,
        exist_ok=True, patience=max(args.epochs, 250), cache=False, verbose=True,
    )
    print(f"TRAIN_DONE {name}")

    weights = PROJECT / name / "weights" / "best.pt"
    best = YOLO(str(weights))
    v = best.val(data=str(DATA), split="val", imgsz=args.imgsz, batch=args.batch, device=0,
                 project=str(PROJECT), name=f"{name}_val", exist_ok=True, verbose=False)
    t = best.val(data=str(DATA), split="test", imgsz=args.imgsz, batch=args.batch, device=0,
                 project=str(PROJECT), name=f"{name}_test", exist_ok=True, verbose=False)
    out = {
        "name": name, "status": "done",
        "val_map50": float(v.box.map50), "val_map50_95": float(v.box.map),
        "test_map50": float(t.box.map50), "test_map50_95": float(t.box.map),
        "weights": str(weights), "cfg": str(cfg_path),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("GSDOWN_SPD_RESULT", json.dumps(out, ensure_ascii=False))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", type=str, default=None, help="只跑某个候选，如 gsdown_spd_p3")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT.mkdir(parents=True, exist_ok=True)

    todo = list(CANDIDATES.items())
    if args.only:
        todo = [(k, v) for k, v in todo if k == args.only]
        if not todo:
            raise SystemExit(f"--only {args.only} not in {list(CANDIDATES)}")

    report_path = PROJECT / "spd_report.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}

    for tag, layers in todo:
        print(f"==== CANDIDATE {tag} layers={layers} ====")
        try:
            out = train_one(tag, layers, args)
            report[out["name"]] = out
        except Exception:
            # 单候选崩溃隔离，不拖垮后续（iter18/19 教训）
            print(f"CANDIDATE_FAILED {tag}")
            traceback.print_exc()
            report[f"y26n_{tag}_e250"] = {"name": f"y26n_{tag}_e250", "status": "failed"}
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"REPORT_WRITTEN {report_path}")

    print("ALL_DONE", json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
