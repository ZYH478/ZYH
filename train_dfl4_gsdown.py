#!/usr/bin/env python
"""iter36：在 MSDGS135eq 效率基座上恢复 DFL 定位分布建模，250e 串行对照训练。

病根回顾（混淆矩阵，gsdown 基线 test）：整机误差几乎全是前景/背景（漏检+误报），
类间混淆≈0。A/B 两方案改的是 box 回归侧，能力上限=把「已检出目标」的框收更紧
（map50-95 抬升、map50 基本不动），碰不到漏检。这是明确的先验，不是悲观估计。

reg_max 机制（从远程 tasks.py / loss.py 真实实现确认）：
- reg_max 是 YAML 顶层键，parse_model 注入 Detect。
- box head 宽度 c2 = max(16, ch[0]//4, reg_max*4)。ch[0]=64 => 16；reg_max=4 => 16。
  所以 reg_max=1->4 不改 box head 宽度，只把 box 输出 4->16 通道并激活 DFL。
- loss 侧全自动：use_dfl = reg_max>1；BboxLoss(reg_max) 启用 DFLoss；proj=arange(reg_max)。
  改 reg_max 后 DFL 分布回归损失自动开启，无需碰 loss.py。

候选（均以 MSDGS135eq YAML 为基座，从 yolo26n.pt 迁移，保 end2end）：
- msdgs_dfl4      : 方案 A。纯 YAML 单改 reg_max 1->4。box head 宽度不变(16)，只恢复 DFL。
- msdgs_bh32_dfl4 : 方案 B。reg_max=4 + Detect->BoxHead32Detect（box 分支宽度强制 32）。
                    需先 install_boxhead32_module.py。A 是 B 的嵌套参照，能拆出「加宽」边际贡献。

对照锚：
- gsdown 独立真值 test mAP50 0.7338 / mAP50-95 0.4018 / fused 1.936M。
- MSDGS135eq(reg_max=1) 独立真值 test 0.732411 / 0.398837 / fused 1.777M（-8.2%）。
判定：test mAP50-95 真值稳定 > 0.4018（reg_max=4 会略增参/降速，需报参数）→ 进 n=4 配对复核。
     单 seed 破线只算候选（13hf/UBHead 教训：单 seed 不算结论，head 侧历史 0/3 泛化难）。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py          # GSConv/VoVGSCSP（gsdown head 必需）
    python install_msdgs_module.py            # MSDGS（基座 neck）
    python install_boxhead32_module.py        # BoxHead32Detect（方案 B 需要）
    python -u train_dfl4_gsdown.py

输出：
- runs_dfl4_gsdown_e250/<name>/weights/best.pt
- generated_models_dfl4_gsdown_e250/<name>.yaml
- runs_dfl4_gsdown_e250/report.json / status.json
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
# MSDGS135eq 基座 YAML（reg_max=1，neck 全 MSDGS）
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_dfl4_gsdown_e250"
GEN_DIR = ROOT / "generated_models_dfl4_gsdown_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

GSDOWN_TRUTH = {
    "test_map50": 0.7338, "test_map50_95": 0.4018, "fused_params": 1935814,
    "crazing_map50": 0.4432, "inclusion_map50": 0.7403, "inclusion_recall": 0.7016,
    "rolled_map50": 0.6023, "rolled_recall": 0.593,
    "scratches_map50": 0.9240, "pitted_map50": 0.7852,
}
MSDGS_TRUTH = {"test_map50": 0.732411, "test_map50_95": 0.398837, "fused_params": 1777318}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_base() -> dict:
    return yaml.safe_load(MSDGS_YAML.read_text(encoding="utf-8"))


def build_dfl4() -> dict:
    """方案 A：纯改 reg_max 1->4，其余一字不动。"""
    doc = _load_base()
    old = doc.get("reg_max", 1)
    doc["reg_max"] = 4
    print(f"reg_max {old} -> 4 (方案A: 纯 DFL 恢复，box head 宽度不变)")
    return doc


def build_bh32_dfl4() -> dict:
    """方案 B：reg_max=4 + head 里的 Detect 换成 BoxHead32Detect。"""
    doc = _load_base()
    doc["reg_max"] = 4
    n_swapped = 0
    for block in doc.get("head", []):
        if len(block) >= 3 and block[2] == "Detect":
            block[2] = "BoxHead32Detect"
            n_swapped += 1
    assert n_swapped == 1, f"expected 1 Detect head, got {n_swapped}"
    print(f"reg_max -> 4 + Detect -> BoxHead32Detect x{n_swapped} (方案B: DFL4 + 加宽 box head=32)")
    return doc


BUILDERS = {
    "msdgs_dfl4": build_dfl4,
    "msdgs_bh32_dfl4": build_bh32_dfl4,
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
        out["per_class"][name] = {
            "map50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i]),
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
        }
    return out


def fused_param_count(weights: Path) -> int:
    """独立进程口径：reload -> fuse -> 数参数。"""
    fm = YOLO(str(weights))
    try:
        fm.model.fuse()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN fuse failed: {exc!r}")
    return int(sum(p.numel() for p in fm.model.parameters()))


def run_one(name: str, epochs: int, batch: int, imgsz: int, seed: int) -> dict:
    doc = BUILDERS[name]()
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    cfg = GEN_DIR / f"{name}.yaml"
    cfg.write_text("# generated by train_dfl4_gsdown.py\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")
    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    model = YOLO(str(cfg), task="detect").load(str(OFFICIAL_WEIGHTS))
    params_unfused = int(sum(p.numel() for p in model.model.parameters()))
    print(f"PARAMS_UNFUSED {name} {params_unfused}")
    print(f"TRAIN_START {name} epochs={epochs} batch={batch} imgsz={imgsz} seed={seed}")
    model.train(
        data=str(DATA), epochs=epochs, imgsz=imgsz, batch=batch, workers=8,
        seed=seed, device=0, project=str(PROJECT), name=name, exist_ok=True,
        patience=max(epochs, 250), cache=False, verbose=True,
    )
    print(f"TRAIN_DONE {name}")

    weights = PROJECT / name / "weights" / "best.pt"
    best = YOLO(str(weights))
    v = best.val(data=str(DATA), split="val", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_val", exist_ok=True, verbose=False)
    t = best.val(data=str(DATA), split="test", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_test", exist_ok=True, verbose=False)
    fused = fused_param_count(weights)
    return {
        "status": "done", "cfg": str(cfg), "weights": str(weights),
        "epochs": epochs, "batch": batch, "imgsz": imgsz, "seed": seed,
        "params_unfused": params_unfused, "params_fused": fused,
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
    report["baseline"] = {"gsdown": GSDOWN_TRUTH, "msdgs135eq": MSDGS_TRUTH}
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
            continue  # 错误隔离：单候选崩溃不拖垮后续
        report["experiments"][name] = result
        REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("DFL4_RESULT", name,
              f"val_mAP50-95={result['val']['map50_95']:.5f}",
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"params_fused={result['params_fused']}")

    save_status("done", None, [])
    print(f"DFL4_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
