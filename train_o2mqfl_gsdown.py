#!/usr/bin/env python
"""iter34 候选（监督侧，跳出 crazing 攻全局 mean AP）：MSDGS neck 基座 + one2many 辅助头
Quality Focal 难例强调（one2one 推理头保持纯 BCE）。

核心假设（gsdown 干净基线混淆矩阵诊断，首次坐实全局病根）：
类间混淆几乎为零（仅 1 个 inclusion→crazing），100% 误差预算 = 前景/背景：
180 缺陷漏检成 bg + 146 bg 误报成缺陷。可回收 mean AP 空间在 rolled-in_scale(0.602)/
inclusion(0.740)，非封顶 crazing。此前 6 次(iter29-33)全攻错位置。

O2M-QFL：只替换 self.one2many.bce = QualityFocal(|target-sigmoid(pred)|^beta)，
one2one 推理头保持纯 BCE，绕开 FocalCW 崩盘根因（污染推理头校准）。QFL 专为 TAL 软标签
设计，同时强调难正例(漏检)与难负例(误报)、压制易例，精准打在 fg/bg 误差预算。

判定（停止规则，owner 定）：
- 整机 test map50 / map50-95 上升（vs gsdown 0.7338/0.4018）→ 治对病，多 seed 复核。
- rolled-in_scale / inclusion 的 recall 上升是主要贡献来源（诊断预期）。
- crazing 不作硬指标（标注天花板），但不应崩。
- 整机塌 / 无增益 → 监督侧难例强调无效，收口。
纪律(13hf 教训)：单 seed 不算数，破线也须多 seed 配对检验。

隔离：O2M-QFL 双重隔离(env-gated + 只碰 o2m)。本脚本顶部先设 O2M_QFL_ENABLE=1，
再 import ultralytics（patch 在 loss.py import 时读该 env，故必须在 import 前设）。
head 仍标准 Detect（task 可自动推断），基座 = MSDGS(135eq) neck（与 FBCon/LCAE 对照同基座）。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py && python install_gsconv_modules.py
    python install_msdgs_module.py && python install_o2mqfl_module.py
    python -u train_o2mqfl_gsdown.py

输出：
- runs_o2mqfl_gsdown_e250/<name>/weights/best.pt
- generated_models_o2mqfl_gsdown_e250/<name>.yaml
- runs_o2mqfl_gsdown_e250/report.json / status.json
"""
from __future__ import annotations

# ！！！必须在 import ultralytics 之前设 env：patch 在 loss.py import 时读取该变量 ！！！
import os
os.environ.setdefault("O2M_QFL_ENABLE", "1")
os.environ.setdefault("O2M_QFL_BETA", "2.0")

import argparse
import json
from pathlib import Path
import time

import yaml
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
GSDOWN_YAML = ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_o2mqfl_gsdown_e250"
GEN_DIR = ROOT / "generated_models_o2mqfl_gsdown_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

GSDOWN_TRUTH = {"test_map50": 0.7338, "test_map50_95": 0.4018,
                "crazing_map50": 0.4432, "crazing_recall": 0.3794, "crazing_precision": 0.5395,
                "inclusion_map50": 0.7403, "inclusion_recall": 0.7016,
                "rolled_map50": 0.6023, "rolled_recall": 0.5930}
MSDGS135EQ_TRUTH = {"test_map50": 0.73241, "test_map50_95": 0.39884}


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


def build_o2mqfl_doc() -> dict:
    """MSDGS neck + 标准 Detect head（无结构改动，改动全在 loss 侧）。"""
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    n_neck = _swap_neck_to_msdgs(doc)
    assert n_neck == 4, f"expected 4 VoVGSCSP in gsdown neck, got {n_neck}"
    print("o2mqfl: neck 4×VoVGSCSP→MSDGS, head standard Detect (loss-side change only)")
    return doc


BUILDERS = {"o2mqfl": build_o2mqfl_doc}


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
    cfg.write_text("# generated by train_o2mqfl_gsdown.py\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")
    print(f"O2M_QFL_ENABLE={os.environ.get('O2M_QFL_ENABLE')} O2M_QFL_BETA={os.environ.get('O2M_QFL_BETA')}")
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
        "o2m_qfl_beta": float(os.environ.get("O2M_QFL_BETA", "2.0")),
        "params_unfused": params,
        "val": metric_dict(v), "test": metric_dict(t),
        "finished_at": now(),
    }
    ts = rec["test"]
    incl = ts["per_class"].get("inclusion", {})
    roll = ts["per_class"].get("rolled-in_scale", {})
    cz = ts["per_class"].get("crazing", {})
    print(f"O2MQFL_RESULT {name} test_mAP50={ts['map50']:.5f}(base 0.7338) "
          f"test_mAP50-95={ts['map50_95']:.5f}(base 0.4018)")
    print(f"  inclusion R={incl.get('recall', 0):.4f}(base 0.7016) map50={incl.get('map50', 0):.4f}(base 0.7403)")
    print(f"  rolled-in R={roll.get('recall', 0):.4f}(base 0.5930) map50={roll.get('map50', 0):.4f}(base 0.6023)")
    print(f"  crazing R={cz.get('recall', 0):.4f}(base 0.3794, 天花板不作硬指标)")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="o2mqfl", choices=list(BUILDERS))
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
    print(f"O2MQFL_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
