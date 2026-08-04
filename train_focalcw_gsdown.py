#!/usr/bin/env python
"""iter32 候选B（损失侧）：MSDGS neck 基座 + FocalCW 分类损失（Focal + crazing 定向类加权）。

核心假设（承接 iter31 混淆矩阵诊断，病根首次坐实）：
crazing 病根 = 前景/背景不可分：test 上 52.4% 的 GT crazing 判成背景（漏检 recall 0.379
全场最低），42.7% 的 crazing 预测来自背景（误报 precision 0.539 全场最低）。既不是回归、
也不是类间混淆（混淆矩阵 crazing 列除 1 个判成 inclusion 外无跨类混淆）。这是典型的
低对比度前景 + 类难度失衡。Focal 聚焦难样本（压低易分背景负样本、逼模型不漏低置信真
crazing）+ crazing 定向类加权（放大 crazing 通道损失贡献），直击「少检出 + 多误报」。

集成：MSDGS(135eq) neck 基座 + 标准 Detect 头（结构不变）。改动只在分类损失：
BCE → FocalBCE（逐元素 focal 调制，走已内置 class_weights 钩子）。推理零影响、结构/
输出/fused 参数与 MSDGS 基线完全一致（1.777M）。crazing 定向权重经 model.model.class_weights。

判定（停止规则，owner 定，主判据 = crazing recall + precision 双升）：
- crazing test recall（基线 0.379）与 precision（基线 0.539）**双双上升** → 治对病，
  继续多 seed 配对检验；且看整机 test map50/map50-95 不塌（vs gsdown 0.7338/0.4018）。
- 只升一个或都不升 / 整机塌 → focal+类加权无法解决前景/背景可分性，收口。
纪律（13hf 教训）：单 seed 不算数，破线也须再上 multi-seed 配对检验。

实验隔离（关键）：FocalCW 全局 patch v8DetectionLoss.__init__，与 FBCon（候选A）不能同装
同跑，否则 A 会连带吃 focal 污染对照。串行：装 FocalCW→训 B→还原 loss.py→装 FBCon→训 A。

对照真值（独立进程 fused 口径 + iter31 诊断混淆矩阵）：
- gsdown: test map50 0.7338 / map50-95 0.4018；crazing recall 0.3794 / precision 0.5395
  / map50 0.4431 / map50-95 0.1536
- MSDGS135eq: test map50 0.73241 / map50-95 0.39884；crazing map50 0.4712 / map50-95 0.1771

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py      # SPDConv/DySample
    python install_gsconv_modules.py          # GSConv/VoVGSCSP（gsdown head 依赖链）
    python install_msdgs_module.py            # MSDGS
    FOCALCW_GAMMA=1.5 python install_focalcw_module.py   # Focal 分类损失
    CRAZING_W=2.0 python -u train_focalcw_gsdown.py

输出：
- runs_focalcw_gsdown_e250/<name>/weights/best.pt
- generated_models_focalcw_gsdown_e250/<name>.yaml
- runs_focalcw_gsdown_e250/report.json / status.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import yaml
import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO26_EXP_ROOT", "/root/autodl-tmp/neu-det-yolo26"))
DATA = ROOT / "dataset" / "neu-det.yaml"
GSDOWN_YAML = ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_focalcw_gsdown_e250"
GEN_DIR = ROOT / "generated_models_focalcw_gsdown_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CRAZING_IDX = 0
# crazing 定向类权重（其余类=1.0）。CRAZING_W 环境变量控制，默认 2.0。
CRAZING_W = float(os.environ.get("CRAZING_W", "2.0"))

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


def build_msdgs_focalcw_doc() -> dict:
    """MSDGS neck + 标准 Detect 头（结构不变，改动只在损失）。"""
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    n_neck = _swap_neck_to_msdgs(doc)
    assert n_neck == 4, f"expected 4 VoVGSCSP in gsdown neck, got {n_neck}"
    print(f"msdgs_focalcw: neck 4×VoVGSCSP→MSDGS, head=标准Detect, 损失=Focal+crazing_w={CRAZING_W}")
    return doc


BUILDERS = {
    "msdgs_focalcw": build_msdgs_focalcw_doc,
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
    # m.box.p / m.box.r 是逐类 precision/recall（按 ap_class_index 顺序）。
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
    cfg.write_text("# generated by train_focalcw_gsdown.py\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")
    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    model = YOLO(str(cfg)).load(str(OFFICIAL_WEIGHTS))
    # crazing 定向类加权：走 loss.py 已内置的 class_weights 钩子（reshape 成 (1,1,nc)）。
    cw = torch.ones(len(NAMES))
    cw[CRAZING_IDX] = CRAZING_W
    model.model.class_weights = cw
    print(f"CLASS_WEIGHTS {cw.tolist()} (crazing idx={CRAZING_IDX}, w={CRAZING_W})")
    print(f"TRAIN_START {name} epochs={epochs} batch={batch} imgsz={imgsz} seed={seed}")
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
        "crazing_w": CRAZING_W, "focalcw_gamma": float(os.environ.get("FOCALCW_GAMMA", "1.5")),
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
        print("FOCALCW_RESULT", name,
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"crazing_R={cz.get('recall', float('nan')):.4f}(base 0.3794)",
              f"crazing_P={cz.get('precision', float('nan')):.4f}(base 0.5395)",
              f"crazing_map50={cz.get('map50', float('nan')):.4f}(base 0.4431)")

    save_status("done", None, [])
    print(f"FOCALCW_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
