#!/usr/bin/env python
"""iter31：MSDGS neck 基座 + UBHead（不确定性感知框头）。

核心假设（承接 iter30 收口）：
crazing 的 map50-95/map50 比值 0.347 全场最低 → 「能找到、框不准」。YOLOv26n 为提速把
reg_max=1（DFL→Identity），恰好砍掉了建模边界模糊的分布/不确定性部件。4 个特征端赛道
（backbone×3 + neck）都撬不动 crazing 定位精度，因为问题不在提特征，在「提出特征后单点
回归表达不了模糊边界」。UBHead 旁挂极简单层 conv σ 头，把 reg_max=1 的 L1 回归项升级为
σ 加权对数似然（RLE 残差形式）：边界清晰边学小 σ 得强约束，边界模糊边学大 σ 自动降权。

集成：MSDGS(135eq) neck 基座 + Detect→UBHead。σ 头只在训练期前向，fuse 后删除，
推理零成本、输出 [1,300,6]、end2end 不破。build 验证已过：推理 fused 1.777M（=MSDGS 基线）
/ end2end=True / reg_max=1 / nl=3 / log_sigma 与 boxes 形状对齐 (1,4,8400)。

对照 = gsdown 独立真值 test mAP50 0.7338 / mAP50-95 0.4018；MSDGS135eq 单独 test mAP50 0.73241 /
       mAP50-95 0.39884 / crazing map50 0.4712 / crazing map50-95 0.1771（历史最好，本轮起点）。

判定（停止规则，owner 定）：只跑 seed0，盯 crazing 的 test map50-95 动没动（UBHead 靶心是
定位精度 map50-95，不是 map50）。
- crazing map50-95 从 0.15-0.18 级明显上抬（+0.03，到 0.18+）且整机 map50-95 不低于基线
  → 机理坐实，继续多 seed 配对检验 + 原创化；
- 不动 → 坐实 crazing 瓶颈在标注层（矩形框标注弥散裂纹网，GT 本身歧义），任何回归头都救不动，
  收口回 MSDGS 效率线。后者也是有价值的负结果。
纪律（13hf 教训）：单 seed 不算数，破线也须再上 multi-seed 配对检验。

注：UBHead 类名不含 "detect"，guess_model_task 推断失败，故加载 YAML 时显式传 task="detect"。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py      # SPDConv/DySample
    python install_gsconv_modules.py          # GSConv/VoVGSCSP（gsdown head 依赖链）
    python install_msdgs_module.py            # MSDGS
    python install_ubhead_module.py           # UBHead + UBE2ELoss
    python -u train_ubhead_gsdown.py

输出：
- runs_ubhead_gsdown_e250/<name>/weights/best.pt
- generated_models_ubhead_gsdown_e250/<name>.yaml
- runs_ubhead_gsdown_e250/report.json / status.json
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
PROJECT = ROOT / "runs_ubhead_gsdown_e250"
GEN_DIR = ROOT / "generated_models_ubhead_gsdown_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# 对照真值（独立进程 fused 口径）
GSDOWN_TRUTH = {"test_map50": 0.7338, "test_map50_95": 0.4018,
                "crazing_map50": 0.4431, "crazing_map50_95": 0.1536}
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


def _swap_detect_to_ubhead(doc: dict) -> int:
    """把 head 末尾的 Detect 换成 UBHead（保留 args=[nc]）。返回替换数。"""
    n_swapped = 0
    for block in doc.get("head", []):
        if len(block) >= 4 and block[2] == "Detect":
            block[2] = "UBHead"
            n_swapped += 1
    return n_swapped


def build_msdgs_ubhead_doc() -> dict:
    """MSDGS neck + Detect → UBHead。"""
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    n_neck = _swap_neck_to_msdgs(doc)
    assert n_neck == 4, f"expected 4 VoVGSCSP in gsdown neck, got {n_neck}"
    n_head = _swap_detect_to_ubhead(doc)
    assert n_head == 1, f"expected 1 Detect head, got {n_head}"
    print("msdgs_ubhead: neck 4×VoVGSCSP→MSDGS, head Detect→UBHead")
    return doc


BUILDERS = {
    "msdgs_ubhead": build_msdgs_ubhead_doc,
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
    cfg.write_text("# generated by train_ubhead_gsdown.py\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")
    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    # UBHead 类名不含 "detect" -> 显式传 task="detect"，否则任务推断为 None。
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
        d_gsdown = result["test"]["map50_95"] - GSDOWN_TRUTH["test_map50_95"]
        crazing = result["test"]["per_class"].get("crazing", {})
        print("UBHEAD_RESULT", name,
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"vs_gsdown={d_gsdown:+.5f}",
              f"crazing_map50-95={crazing.get('map50_95', float('nan')):.5f}",
              f"(gsdown 0.1536 / msdgs 0.1771)",
              f"crazing_map50={crazing.get('map50', float('nan')):.5f}",
              f"params_unfused={result['params_unfused']}")

    save_status("done", None, [])
    print(f"UBHEAD_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
