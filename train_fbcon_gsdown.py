#!/usr/bin/env python
"""iter32 候选A（结构侧）：MSDGS neck 基座 + FBHead（前景-背景对比增强分类头）。

核心假设（承接 iter31 混淆矩阵诊断，病根首次坐实）：
crazing 病根 = 前景/背景不可分：test 上 52.4% 的 GT crazing 判成背景（漏检 recall 0.379
全场最低），42.7% 的 crazing 预测来自背景（误报 precision 0.539 全场最低）。既不是回归、
也不是类间混淆。crazing 是低对比度弥散纹理，前景相对局部邻域背景差异微弱但存在。FBCon
用局部高通残差 x - avgpool(x) 放大微弱前景对比信号，**只作用于分类分支**（判别前景/背景），
直击「少检出 + 多误报」。

集成：MSDGS(135eq) neck 基座 + Detect→FBHead。每检测层分类分支前加 FBCon（depthwise 高通
残差 + 标量 gamma 初始 0=恒等起步）。box 回归不受影响。FBCon 在分类路径、推理保留（生效），
depthwise 极省参。build 验证已过：推理 fused 1.778M（+768 params，红线内）/ end2end=True
/ reg_max=1 / nl=3 / 输出 (1,300,6)。

与 iter28 HFDGS 的关键区别：同样 x - avgpool(x) 高通算子，iter28 放 neck 全局被证伪（放大
训练噪声、有害）；这里放**分类分支前**放大前景-背景对比——诊断指向的正确位置，是有依据的
重新定位，不是重走死路。

判定（停止规则，owner 定，主判据 = crazing recall + precision 双升）：
- crazing test recall（基线 0.379）与 precision（基线 0.539）**双双上升** → 治对病，
  继续多 seed 配对检验；且看整机 test map50/map50-95 不塌（vs gsdown 0.7338/0.4018）。
- 只升一个或都不升 / 整机塌 → 前景对比增强无法解决可分性，收口。
纪律（13hf 教训）：单 seed 不算数，破线也须再上 multi-seed 配对检验。

实验隔离（关键）：本候选须在**还原 loss.py（卸载 FocalCW patch）后**单独跑，否则会连带
吃 focal 损失、污染纯结构对照。串行：训 B（FocalCW）→还原 loss.py→装 FBCon→训 A。

注：FBHead 类名不含 "detect"，guess_model_task 推断失败，故加载 YAML 时显式传 task="detect"。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    # 先确认 loss.py 已还原（无 FocalCW patch）：
    cp ultralytics.../utils/loss.py.focalcw_bak ultralytics.../utils/loss.py   # 若装过 FocalCW
    python install_yolo26_exp_modules.py && python install_gsconv_modules.py
    python install_msdgs_module.py
    FBCON_K=5 python install_fbcon_module.py
    python -u train_fbcon_gsdown.py

输出：
- runs_fbcon_gsdown_e250/<name>/weights/best.pt
- generated_models_fbcon_gsdown_e250/<name>.yaml
- runs_fbcon_gsdown_e250/report.json / status.json
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
PROJECT = ROOT / "runs_fbcon_gsdown_e250"
GEN_DIR = ROOT / "generated_models_fbcon_gsdown_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# 对照真值（独立进程 fused 口径 + iter31 混淆矩阵诊断）
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


def _swap_detect_to_fbhead(doc: dict) -> int:
    """把 head 末尾的 Detect 换成 FBHead（保留 args=[nc]）。返回替换数。"""
    n_swapped = 0
    for block in doc.get("head", []):
        if len(block) >= 4 and block[2] == "Detect":
            block[2] = "FBHead"
            n_swapped += 1
    return n_swapped


def build_msdgs_fbcon_doc() -> dict:
    """MSDGS neck + Detect → FBHead。"""
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    n_neck = _swap_neck_to_msdgs(doc)
    assert n_neck == 4, f"expected 4 VoVGSCSP in gsdown neck, got {n_neck}"
    n_head = _swap_detect_to_fbhead(doc)
    assert n_head == 1, f"expected 1 Detect head, got {n_head}"
    print("msdgs_fbcon: neck 4×VoVGSCSP→MSDGS, head Detect→FBHead")
    return doc


BUILDERS = {
    "msdgs_fbcon": build_msdgs_fbcon_doc,
}


def save_status(status: str, current, pending) -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps({
        "status": status, "current": current, "pending": pending,
        "updated_at": now(), "report_json": str(REPORT_JSON),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def metric_dict(m) -> dict:
    """抓整机 + 逐类 map50/map50-95 + 逐类 precision/recall（停止判据要 crazing P/R）。"""
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
    cfg.write_text("# generated by train_fbcon_gsdown.py\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")
    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    # FBHead 类名不含 "detect" -> 显式传 task="detect"。
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
                 project=str(PROJECT), name=f"{name}_val", exist_ok=True, verbose=False)
    t = best.val(data=str(DATA), split="test", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_test", exist_ok=True, verbose=False)
    return {
        "status": "done", "cfg": str(cfg), "weights": str(weights),
        "epochs": epochs, "batch": batch, "imgsz": imgsz, "seed": seed,
        "fbcon_k": int(os.environ.get("FBCON_K", "5")),
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
    report["baseline"] = {"gsdown": GSDOWN_TRUTH, "msdgs135eq": MSDGS135EQ_TRUTH}
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
            continue
        report["experiments"][name] = result
        REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        cz = result["test"]["per_class"].get("crazing", {})
        print("FBCON_RESULT", name,
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"crazing_R={cz.get('recall', float('nan')):.4f}(base 0.3794)",
              f"crazing_P={cz.get('precision', float('nan')):.4f}(base 0.5395)",
              f"crazing_map50={cz.get('map50', float('nan')):.4f}(base 0.4431)",
              f"params_unfused={result['params_unfused']}")

    save_status("done", None, [])
    print(f"FBCON_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
