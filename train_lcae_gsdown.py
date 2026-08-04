#!/usr/bin/env python
"""iter33 候选（backbone 浅层）：MSDGS neck 基座 + backbone P2/4 后插 LCAE。

核心假设（承接 iter31/32 混淆矩阵诊断，五连败后换赛道到浅层）：
crazing 病根 = 前景/背景不可分：test 上 52% GT crazing 判成背景（recall 0.379 全场最低）。
crazing 是低对比度弥散裂纹，前景相对局部背景灰度差异微弱但存在。此前所有攻法都放在深层
(backbone P3/P4、head、检测头、损失)，而微弱前景信号在深层已被下采样平滑掉。

LCAE = 局部对比度自适应增强(divisive normalization)，放在 backbone **最浅层 P2/4**
(第二个 Conv, layer 1 之后, stride=4, 信号尚未平滑)：
    mu=avgpool(x); var=avgpool(x^2)-mu^2; c=(x-mu)/sqrt(var+eps); out=x+gamma*c
σ 小的低对比度区(crazing 前景)被相对放大，σ 大的强边界(inclusion/scratches)几乎不动。

与两条死路的分野：
- vs iter28 HFDGS(x-avgpool 无差别高通，证伪)：LCAE 多除局部 σ = 自适应，只放大弱对比。
- vs iter29 backbone(DWR/SPD 深层，证伪)：LCAE 放最浅层，攻 crazing 唯一有信号的位置。

集成（层索引偏移，最易埋雷处，程序化处理）：
在 backbone index 1 后插入一层 LCAE → 其后所有层索引 +1 → head 所有绝对 from 引用
(Concat 6/4/13/10、Detect 16/19/22)统一 +1；-1 相对引用不变。

判定（停止规则，owner 定，主判据 = crazing recall + precision 双升）：
- crazing test recall(基线 0.379)与 precision(基线 0.539)**双双上升** → 治对病，多 seed 复核。
- recall 仍卡 8/21(0.379 同一整数) / 整机塌 → 标注天花板实锤，crazing 路彻底收口。
纪律(13hf 教训)：单 seed 不算数。

FBHead/LCAE 无关；head 仍标准 Detect，task 可自动推断，但保险起见显式传 task="detect"。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py && python install_gsconv_modules.py
    python install_msdgs_module.py && python install_lcae_module.py
    python -u train_lcae_gsdown.py

输出：
- runs_lcae_gsdown_e250/<name>/weights/best.pt
- generated_models_lcae_gsdown_e250/<name>.yaml
- runs_lcae_gsdown_e250/report.json / status.json
"""
from __future__ import annotations

import argparse
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
PROJECT = ROOT / "runs_lcae_gsdown_e250"
GEN_DIR = ROOT / "generated_models_lcae_gsdown_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# 插入点：backbone layer 1(第二个 Conv, P2/4)之后。LCAE 成为 index 2，其后全部 +1。
LCAE_INSERT_AFTER = 1
# LCAE YAML args：nominal 通道(P2/4 = 128)，parse_model 按 width 缩放得 c2=c1，通道不变。
LCAE_NOMINAL_CH = 128

GSDOWN_TRUTH = {"test_map50": 0.7338, "test_map50_95": 0.4018,
                "crazing_map50": 0.4431, "crazing_map50_95": 0.1536,
                "crazing_recall": 0.3794, "crazing_precision": 0.5395}
MSDGS135EQ_TRUTH = {"test_map50": 0.73241, "test_map50_95": 0.39884,
                    "crazing_map50": 0.4712, "crazing_map50_95": 0.1771}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _swap_neck_to_msdgs(doc: dict, dilations=(1, 3, 5), fracs=(1, 1, 1)) -> int:
    """把 gsdown head 里所有 VoVGSCSP 换成 MSDGS(135eq)。返回替换数。"""
    n_swapped = 0
    for block in doc.get("head", []):
        if len(block) >= 4 and block[2] == "VoVGSCSP":
            c2 = block[3][0]
            block[2] = "MSDGS"
            block[3] = [c2, True, 1, 0.5, list(dilations), list(fracs)]
            n_swapped += 1
    return n_swapped


def _shift_from(f, insert_pos: int):
    """把单个 from 引用做 +1 偏移：绝对索引 >= insert_pos 的 +1；相对(负)不变。"""
    if isinstance(f, list):
        return [_shift_from(x, insert_pos) for x in f]
    if isinstance(f, int) and f >= 0 and f >= insert_pos:
        return f + 1
    return f


def _insert_lcae(doc: dict) -> None:
    """在 backbone index=LCAE_INSERT_AFTER 之后插入 LCAE，并把其后所有绝对 from 引用 +1。

    LCAE 新层位置 = LCAE_INSERT_AFTER + 1。所有引用原 index >= (LCAE_INSERT_AFTER+1) 的 +1。
    等价于：所有绝对引用 >= insert_pos 的 +1，其中 insert_pos = LCAE_INSERT_AFTER + 1。
    """
    insert_pos = LCAE_INSERT_AFTER + 1  # 新层将占据的 index
    backbone = doc["backbone"]
    head = doc["head"]

    # 1) 先把 backbone + head 里所有绝对 from >= insert_pos 的引用 +1（在插入前按原索引判断）
    for block in backbone + head:
        block[0] = _shift_from(block[0], insert_pos)

    # 2) 插入 LCAE 层（from=-1，取上一层 P2/4 输出）
    lcae_layer = [-1, 1, "LCAE", [LCAE_NOMINAL_CH]]
    backbone.insert(insert_pos, lcae_layer)


def build_lcae_doc() -> dict:
    """MSDGS neck + backbone P2/4 后插 LCAE。"""
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    n_neck = _swap_neck_to_msdgs(doc)
    assert n_neck == 4, f"expected 4 VoVGSCSP in gsdown neck, got {n_neck}"
    _insert_lcae(doc)
    # 完整性自检：LCAE 恰好 1 层，且在 index insert_pos
    insert_pos = LCAE_INSERT_AFTER + 1
    lcae_idxs = [i for i, b in enumerate(doc["backbone"]) if len(b) >= 3 and b[2] == "LCAE"]
    assert lcae_idxs == [insert_pos], f"LCAE at {lcae_idxs}, expected [{insert_pos}]"
    print(f"lcae: neck 4×VoVGSCSP→MSDGS, backbone insert LCAE @index {insert_pos} (P2/4)")
    print(f"  head Detect from = {doc['head'][-1][0]} (expect [17,20,23])")
    return doc


BUILDERS = {"lcae": build_lcae_doc}


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
    p_arr = getattr(m.box, "p", None)
    r_arr = getattr(m.box, "r", None)
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[int(c)] if int(c) < len(NAMES) else str(int(c))
        entry = {"map50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i])}
        if p_arr is not None and i < len(p_arr):
            entry["precision"] = float(p_arr[i])
        if r_arr is not None and i < len(r_arr):
            entry["recall"] = float(r_arr[i])
        out["per_class"][name] = entry
    return out


def run_one(name: str, epochs: int, batch: int, imgsz: int, seed: int) -> dict:
    doc = BUILDERS[name]()
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GEN_DIR / f"{name}.yaml"
    cfg.write_text("# generated by train_lcae_gsdown.py\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")
    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    model = YOLO(str(cfg), task="detect").load(str(OFFICIAL_WEIGHTS))
    print(f"TRAIN_START {name} epochs={epochs} batch={batch} imgsz={imgsz} seed={seed}")
    model.train(
        data=str(DATA), epochs=epochs, imgsz=imgsz, batch=batch, workers=8,
        seed=seed, device=0, project=str(PROJECT), name=name, exist_ok=True,
        patience=max(epochs, 250), cache=False, verbose=True,
    )
    print(f"TRAIN_DONE {name}")

    weights = PROJECT / name / "weights" / "best.pt"
    best = YOLO(str(weights), task="detect")
    params = int(sum(p.numel() for p in best.model.parameters()))
    v = best.val(data=str(DATA), split="val", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_val", exist_ok=True)
    t = best.val(data=str(DATA), split="test", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_test", exist_ok=True)
    rec = {
        "status": "done", "cfg": str(cfg), "weights": str(weights),
        "epochs": epochs, "batch": batch, "imgsz": imgsz, "seed": seed,
        "lcae_insert_after": LCAE_INSERT_AFTER,
        "params_unfused": params,
        "val": metric_dict(v), "test": metric_dict(t),
        "finished_at": now(),
    }
    cz = rec["test"]["per_class"].get("crazing", {})
    print(f"LCAE_RESULT {name} test_mAP50={rec['test']['map50']:.5f} "
          f"test_mAP50-95={rec['test']['map50_95']:.5f} "
          f"crazing_R={cz.get('recall', 0):.4f}(base 0.3794) "
          f"crazing_P={cz.get('precision', 0):.4f}(base 0.5395) "
          f"crazing_map50={cz.get('map50', 0):.4f}(base 0.4431)")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lcae", choices=list(BUILDERS))
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PROJECT.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "experiments": {},
              "baseline": {"gsdown": GSDOWN_TRUTH, "msdgs135eq": MSDGS135EQ_TRUTH}}
    if REPORT_JSON.exists():
        try:
            report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
        report.setdefault("experiments", {})
        report.setdefault("baseline", {"gsdown": GSDOWN_TRUTH, "msdgs135eq": MSDGS135EQ_TRUTH})

    save_status("running", args.name, [])
    rec = run_one(args.name, args.epochs, args.batch, args.imgsz, args.seed)
    report["experiments"][args.name] = rec
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_status("done", None, [])
    print(f"LCAE_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
