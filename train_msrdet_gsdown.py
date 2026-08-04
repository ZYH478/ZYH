#!/usr/bin/env python
"""iter23：把 MSR-Det 论文三个模块叠到 vovgscsp_gsdown 上，250e 串行对照训练。

owner 指定三方案（全部在 gsdown 基础上、从 yolo26n.pt 迁移、保 end2end/reg_max=1）：

1. gsdown_lkbm      : backbone 层6 C3k2->LLKBM(k=7)、层8 C3k2->LLKBM(k=9)（论文最优 {7,9}）
2. gsdown_agspp     : backbone 层9 SPPF->AGSPP（自适应门控 SPP）
3. gsdown_agspp_fem : 层9 SPPF->AGSPP，且 neck P3 输出(层16)后插 FEM，用 C5(层10)引导 P3

对照 = gsdown 现有独立真值 test mAP50 0.7338 / mAP50-95 0.4018 / 1.936M（轻量交付模型）。

关键结构事实（已核实 gsdown YAML）：
- backbone: 0-1 Conv, 2 C3k2, 3 Conv, 4 C3k2, 5 Conv, 6 C3k2[512,true],
            7 Conv, 8 C3k2[1024,true], 9 SPPF[1024,5,3,true], 10 C2PSA[1024]
- head: ... 16 VoVGSCSP[256]=P3, 19 VoVGSCSP[512]=P4, 22 VoVGSCSP[1024]=P5, 23 Detect[16,19,22]

FEM 方案需在 P3(层16) 后插一层 FEM，用 C5(层10) 引导，Detect 输入 P3 索引从 16->17，
其余 Concat 引用 P4/P5 的下采样链需要相应 +1 顺延。本脚本用「插入后重算索引」处理，
不手写死索引，避免错挂。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py      # SPDConv/DySample（gsdown 谱系依赖链）
    python install_gsconv_modules.py          # GSConv/VoVGSCSP（gsdown head 必需）
    python install_msrdet_modules.py          # LLKBM/AGSPP/FEM（本轮）
    python -u train_msrdet_gsdown.py

输出：
- runs_msrdet_gsdown_e250/<name>/weights/best.pt
- generated_models_msrdet_gsdown_e250/<name>.yaml
- runs_msrdet_gsdown_e250/report.json / status.json
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
PROJECT = ROOT / "runs_msrdet_gsdown_e250"
GEN_DIR = ROOT / "generated_models_msrdet_gsdown_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_gsdown() -> dict:
    return yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))


def build_lkbm() -> dict:
    """backbone 层6 C3k2->LLKBM(k=7)、层8 C3k2->LLKBM(k=9)。head 一字不动。"""
    doc = load_gsdown()
    bb = doc["backbone"]
    assert bb[6][2] == "C3k2", f"layer6 not C3k2: {bb[6]}"
    assert bb[8][2] == "C3k2", f"layer8 not C3k2: {bb[8]}"
    # C3k2 args=[c2, shortcut]; LLKBM args=[c2, k, shortcut]
    bb[6] = [bb[6][0], bb[6][1], "LLKBM", [512, 7, True]]
    bb[8] = [bb[8][0], bb[8][1], "LLKBM", [1024, 9, True]]
    print("LKBM: backbone[6]->LLKBM(k=7), backbone[8]->LLKBM(k=9)")
    return doc


def build_agspp() -> dict:
    """backbone 层9 SPPF->AGSPP。head 一字不动。"""
    doc = load_gsdown()
    bb = doc["backbone"]
    assert bb[9][2] == "SPPF", f"layer9 not SPPF: {bb[9]}"
    # SPPF args=[1024,5,3,true] -> AGSPP 同签名
    bb[9] = [bb[9][0], bb[9][1], "AGSPP", list(bb[9][3])]
    print("AGSPP: backbone[9] SPPF->AGSPP")
    return doc


def build_agspp_fem() -> dict:
    """层9 SPPF->AGSPP；在 P3(层16) 后插 FEM(用 C5=层10 引导)，重算 Detect/Concat 索引。

    插入策略：在原层 16(P3 VoVGSCSP[256]) 之后插入一层 FEM，from=[16, 10]。
    插入后，原 >=17 的层索引整体 +1，需要修正所有 head 里 >=17 的引用（含 Detect 的 [16,19,22]
    -> [17,20,23]，以及 P4/P5 分支 Concat 对 P4_lat 的引用）。用统一重映射函数处理。
    """
    doc = build_agspp()  # FEM 方案叠在 AGSPP 之上（owner 指定 AGSPP+FEM）
    bb_len = len(doc["backbone"])
    head = doc["head"]

    # head 层的全局索引 = bb_len + head_local_idx。P3 是 head 里输出 [256] 的那层，
    # 从 gsdown 结构确定其全局索引为 16（bb_len=11, head local 5）。
    P3_GLOBAL = 16
    C5_GLOBAL = 10  # C2PSA 输出（backbone 最后一层）
    insert_local = P3_GLOBAL - bb_len + 1  # 在 P3 之后插入

    def remap(ref):
        # 插入点在全局索引 (P3_GLOBAL+1)=17。>=17 的正索引整体 +1；负索引/其它不变。
        if isinstance(ref, int) and ref >= P3_GLOBAL + 1:
            return ref + 1
        return ref

    # 先重映射现有 head 所有 from 引用（负数 -1 相对引用不受影响）。
    for row in head:
        f = row[0]
        if isinstance(f, list):
            row[0] = [remap(x) for x in f]
        else:
            row[0] = remap(f)

    # 插入 FEM 层：from=[P3_GLOBAL, C5_GLOBAL], args=[256]（P3 通道）。
    fem_row = [[P3_GLOBAL, C5_GLOBAL], 1, "FEM", [256]]
    head.insert(insert_local, fem_row)

    # Detect 现在应引用增强后的 P3（新全局索引 P3_GLOBAL+1=17）而非原 16。
    det = head[-1]
    assert det[2] == "Detect", f"last head layer not Detect: {det}"
    new_det_from = []
    for x in det[0]:
        new_det_from.append(P3_GLOBAL + 1 if x == P3_GLOBAL else x)
    det[0] = new_det_from
    print(f"AGSPP+FEM: insert FEM(from=[{P3_GLOBAL},{C5_GLOBAL}]) after P3; Detect from={det[0]}")
    return doc


BUILDERS = {
    "y26n_gsdown_lkbm_e250": build_lkbm,
    "y26n_gsdown_agspp_e250": build_agspp,
    "y26n_gsdown_agspp_fem_e250": build_agspp_fem,
}


def save_status(status: str, current, pending) -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps({
        "status": status, "current": current, "pending": pending,
        "updated_at": now(), "report_json": str(REPORT_JSON),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def metric_dict(m) -> dict:
    infer = float(m.speed.get("inference", 0.0) or 0.0)
    out = {
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "infer_ms": infer, "fps_infer_only": (1000.0 / infer) if infer else None,
        "per_class": {},
    }
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[int(c)] if int(c) < len(NAMES) else str(int(c))
        out["per_class"][name] = {"map50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i])}
    return out


def run_one(name: str, epochs: int, batch: int, imgsz: int, seed: int) -> dict:
    doc = BUILDERS[name]()
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GEN_DIR / f"{name}.yaml"
    cfg.write_text("# generated by train_msrdet_gsdown.py\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")
    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    model = YOLO(str(cfg)).load(str(OFFICIAL_WEIGHTS))
    print(f"TRAIN_START {name} epochs={epochs} batch={batch} imgsz={imgsz}")
    model.train(
        data=str(DATA), epochs=epochs, imgsz=imgsz, batch=batch, workers=8,
        seed=seed, device=0, project=str(PROJECT), name=name, exist_ok=True,
        patience=max(epochs, 250), cache=False, verbose=True,
    )
    print(f"TRAIN_DONE {name}")

    weights = PROJECT / name / "weights" / "best.pt"
    best = YOLO(str(weights))
    params = int(sum(p.numel() for p in best.model.parameters()))
    v = best.val(data=str(DATA), split="val", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_val", exist_ok=True, verbose=False)
    t = best.val(data=str(DATA), split="test", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_test", exist_ok=True, verbose=False)
    return {
        "status": "done", "cfg": str(cfg), "weights": str(weights),
        "epochs": epochs, "batch": batch, "imgsz": imgsz, "seed": seed,
        "params_unfused": params,
        "val": metric_dict(v), "test": metric_dict(t),
        "finished_at": now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", default="", help="逗号分隔候选名，只跑指定")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    PROJECT.mkdir(parents=True, exist_ok=True)
    selected = [x.strip() for x in args.only.split(",") if x.strip()] or list(BUILDERS)
    unknown = [x for x in selected if x not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown candidates: {unknown}")

    report = {}
    if REPORT_JSON.exists():
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    report.setdefault("schema_version", 1)
    report.setdefault("experiments", {})
    completed = {k for k, v in report["experiments"].items() if v.get("status") == "done"}

    save_status("running", selected[0], selected)
    for pos, name in enumerate(selected, 1):
        if name in completed and not args.force:
            print(f"SKIP_DONE {name}")
            continue
        save_status("running", name, selected[pos:])
        try:
            result = run_one(name, args.epochs, args.batch, args.imgsz, args.seed)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            report["experiments"][name] = {"status": "failed", "error": repr(exc), "failed_at": now()}
            REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            save_status("failed", name, selected[pos:])
            print(f"CANDIDATE_FAILED {name} (isolated, continue)")
            continue  # 错误隔离：单候选崩溃不拖垮后续（iter19 教训）
        report["experiments"][name] = result
        REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("MSRDET_RESULT", name,
              f"val_mAP50={result['val']['map50']:.5f}",
              f"val_mAP50-95={result['val']['map50_95']:.5f}",
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"params_unfused={result['params_unfused']}")

    save_status("done", None, [])
    print(f"MSRDET_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
