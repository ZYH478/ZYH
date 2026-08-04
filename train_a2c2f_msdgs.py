#!/usr/bin/env python
"""iter38：MSDGS135eq 基座 + backbone 层6/8 C3k2 -> A2C2f（区域注意力，加容量），250e seed0 单验。

背景（如实）：backbone 侧「换归纳偏置不加容量」的轻量替换已 14 次失败（iter29 判小数据
backbone 多尺度增益≈0）。但 A2C2f 与那些本质不同——它是「实打实加注意力容量」，且在另一条
赛道（150e / 纯 yolo26n 主干 + 原生 neck）上曾 +0.46pp（0.41262 -> 0.41717）。owner 要求
把它移植到 MSDGS 主线上实测，用数字说话，不靠推理下结论。

⚠️ 两个已知前提（跑前记录，防止误读结果）：
1. +0.46pp 是在「150e / 原生 neck」赛道测的，对照 baseline 0.41262 竟高于本线 gsdown 0.4018
   ——两条赛道口径不同，A2C2f 的正增益不保证移植到 250e/MSDGS 线仍为正。
2. A2C2f 比 C3k2 重，会顶穿参数红线（其原线 fused 2.417M）。组合后大概率 > 1.936M，
   与 MSDGS「轻量 -8.2%」叙事冲突。本轮以「精度能否突破」为目标，参数作为代价如实记录。

reg_max 保持 1（iter36 已证 reg_max>1 在本数据崩，绝不再动）。end2end 保持。

候选（MSDGS135eq YAML 为基座，从 yolo26n.pt 迁移）：
- msdgs_a2c2f_bb : backbone 层6 C3k2[512,true] -> A2C2f[512,True,4]，
                   层8 C3k2[1024,true] -> A2C2f[1024,True,1]（对齐 iter5 口径）。

对照锚：
- gsdown 独立真值 test mAP50 0.7338 / mAP50-95 0.4018 / fused 1.936M。
- MSDGS135eq 独立真值 test 0.732411 / 0.398837 / fused 1.777M（-8.2%）。
判定：seed0 test mAP50-95 稳定 > 0.4018 才算候选，过线再上 seed0/1/2/3 配对复核；
     未过线→收口归档 iter38。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py          # GSConv/VoVGSCSP（gsdown head 必需）
    python install_msdgs_module.py            # MSDGS（基座 neck）
    # A2C2f 是 ultralytics 原生模块，无需安装
    python -u train_a2c2f_msdgs.py

输出：
- runs_a2c2f_msdgs_e250/<name>/weights/best.pt
- generated_models_a2c2f_msdgs_e250/<name>.yaml
- runs_a2c2f_msdgs_e250/report.json / status.json
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
MSDGS_YAML = ROOT / "generated_models_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_a2c2f_msdgs_e250"
GEN_DIR = ROOT / "generated_models_a2c2f_msdgs_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

GSDOWN_TRUTH = {
    "test_map50": 0.7338, "test_map50_95": 0.4018, "fused_params": 1935814,
    "crazing_map50": 0.4432, "inclusion_map50": 0.7403, "inclusion_recall": 0.7016,
    "rolled_map50": 0.6023, "rolled_recall": 0.593, "scratches_map50": 0.9240,
}
MSDGS_TRUTH = {"test_map50": 0.732411, "test_map50_95": 0.398837, "fused_params": 1777318}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_base() -> dict:
    return yaml.safe_load(MSDGS_YAML.read_text(encoding="utf-8"))


def build_a2c2f_bb() -> dict:
    """把 backbone 层6/8 的 C3k2 换成 A2C2f（区域注意力）。

    MSDGS YAML backbone 标称通道（未乘 width scale）：
    - index6 = C3k2[512, true]（repeats=2）  -> A2C2f[512, True, 4]（repeats 保持 2）
    - index8 = C3k2[1024, true]（repeats=2） -> A2C2f[1024, True, 1]（repeats 保持 2）
    识别键：module=='C3k2' 且 args[1] is True（层6/8 是深层 true；层2/4 是 false）。
    A2C2f YAML args = [c2, a2, area]，parse_model 把 repeats(block[1]) 作为 n 插入：
    A2C2f(c1, c2, n=repeats, a2=True, area=area)。所以第3位是 **area(区域数)** 不是 n。
    对齐 iter5：层6 area=4、层8 area=1，repeats 两处都保持基座原值 2（不改 block[1]）。
    """
    doc = _load_base()
    bb = doc.get("backbone", [])
    swapped = []
    for i, block in enumerate(bb):
        if len(block) >= 4 and block[2] == "C3k2" and block[3] \
                and len(block[3]) >= 2 and block[3][1] is True:
            c2 = block[3][0]
            area = 4 if c2 == 512 else 1  # 层6(512,P4)->area=4，层8(1024,P5)->area=1，对齐 iter5
            block[2] = "A2C2f"
            block[3] = [c2, True, area]  # [c2, a2=True, area]；repeats(block[1]) 保持不动
            swapped.append((i, c2, block[1], area))
    assert len(swapped) == 2, f"expected 2 deep C3k2[*,true], swapped {swapped}"
    print(f"swapped backbone deep C3k2 -> A2C2f at (idx,c2,repeats,area)={swapped}")
    return doc


BUILDERS = {
    "msdgs_a2c2f_bb": build_a2c2f_bb,
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
    cfg.write_text("# generated by train_a2c2f_msdgs.py\n"
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
            continue
        report["experiments"][name] = result
        REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("A2C2F_RESULT", name,
              f"val_mAP50-95={result['val']['map50_95']:.5f}",
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"params_fused={result['params_fused']}")

    save_status("done", None, [])
    print(f"A2C2F_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
