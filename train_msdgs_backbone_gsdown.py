#!/usr/bin/env python
"""iter029：MSDGS neck 之上再做 backbone 增强，重试历史上被"瘦 neck"掐死的 backbone 改进。

核心假设（本轮唯一动机，必须写清）：
iter19-24 累计 13 次 backbone/模块尝试全部失败，失败机理完全同款——不是模块本身不行，而是
gsdown 那个被 GSConv 压薄的 VoVGSCSP neck 太瘦，接不住 backbone 增强出来的丰富特征：
- DWR 叠 base backbone 有效（iter19 +1.69pp，crazing 弱类冲到全场新高 0.2024）；
  叠 gsdown 失效（iter20 crazing 从 0.20 掉回 0.14）。归因原话：DWR 多尺度空洞特征与已被
  GSConv 削薄的 neck 表达能力不匹配。
- SPDConv 叠 base 是精度赢家 winner 的核心功臣（+3.5pp）；叠 gsdown 掉点（iter22 -1.24pp）。
  归因：SPD 让 backbone 通道翻 4 倍，瘦 neck 消化不了。

现在 neck 已换成 MSDGS（多尺度解耦、物理隔离，表达力强于原 VoVGSCSP）。当年掐死这些 backbone
改进的"瘦 neck"瓶颈已被移除——这是前提条件被实打实改变的重试，不是空想。

两个候选（都在 MSDGS(135eq) neck 基础上叠 backbone 增强，从 yolo26n.pt 迁移，保 end2end/reg_max=1）：
- msdgs_dwr   : MSDGS neck + backbone 层6/8 C3k2 → DWRC3k2（首选：DWR 是唯一在 base 上真涨精度、
               还把最弱 crazing 拉到历史新高的 backbone 模块；空洞 depthwise 比 C3k2 更省参）
- msdgs_spd_p3: MSDGS neck + backbone 层3 下采样 Conv → SPDConv（次选：winner 核心功臣，信息保真类，
               唯一稳定有效的模块族；会小幅增参）

对照 = gsdown 独立真值 test mAP50 0.7338 / mAP50-95 0.4018 / 1.936M；
       MSDGS135eq 单独 test mAP50-95 0.3988（本轮要看 backbone 增强能否在 MSDGS 上把它推过 0.4018）。

纪律（13hf 教训）：单 seed 破线不算数。本脚本只跑 seed0 看谁破线、破多少；破线者再单独上 multi-seed
配对检验确认稳健，才固化结论。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py      # SPDConv/DySample
    python install_gsconv_modules.py          # GSConv/VoVGSCSP（gsdown head 依赖链）
    python install_backbone_modules.py        # DCNv2Conv/PKIC3k2/DWRC3k2
    python install_msdgs_module.py            # MSDGS
    python -u train_msdgs_backbone_gsdown.py

输出：
- runs_msdgs_backbone_gsdown_e250/<name>/weights/best.pt
- generated_models_msdgs_backbone_gsdown_e250/<name>.yaml
- runs_msdgs_backbone_gsdown_e250/report.json / status.json
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
PROJECT = ROOT / "runs_msdgs_backbone_gsdown_e250"
GEN_DIR = ROOT / "generated_models_msdgs_backbone_gsdown_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# 对照真值（独立进程 fused 口径，口径铁律锁死）
GSDOWN_TRUTH = {"test_map50": 0.7338, "test_map50_95": 0.4018}
MSDGS135EQ_TRUTH = {"test_map50": 0.73241, "test_map50_95": 0.39884}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _swap_neck_to_msdgs(doc: dict, dilations=(1, 3, 5), fracs=(1, 1, 1)) -> int:
    """把 gsdown head 里所有 VoVGSCSP 换成 MSDGS(dilations, fracs)。返回替换数。"""
    n_swapped = 0
    for block in doc.get("head", []):
        if len(block) >= 4 and block[2] == "VoVGSCSP":
            c2 = block[3][0]
            # VoVGSCSP args=[c2] -> MSDGS args=[c2, shortcut, g, e, dilations, fracs]
            block[2] = "MSDGS"
            block[3] = [c2, True, 1, 0.5, list(dilations), list(fracs)]
            n_swapped += 1
    return n_swapped


def build_msdgs_dwr_doc() -> dict:
    """MSDGS neck + backbone 层6/8 C3k2 → DWRC3k2。"""
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    n_neck = _swap_neck_to_msdgs(doc)
    assert n_neck == 4, f"expected 4 VoVGSCSP in gsdown neck, got {n_neck}"
    # backbone 层6=C3k2[512,true]、层8=C3k2[1024,true] → DWRC3k2
    bb = doc["backbone"]
    n_bb = 0
    for i in (6, 8):
        assert bb[i][2] == "C3k2", f"backbone[{i}] expected C3k2, got {bb[i][2]}"
        bb[i][2] = "DWRC3k2"
        n_bb += 1
    assert n_bb == 2, f"expected 2 backbone C3k2 swapped, got {n_bb}"
    print(f"msdgs_dwr: neck 4×VoVGSCSP→MSDGS, backbone 层6/8 C3k2→DWRC3k2")
    return doc


def build_msdgs_spd_p3_doc() -> dict:
    """MSDGS neck + backbone 层3 下采样 Conv → SPDConv。"""
    doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
    n_neck = _swap_neck_to_msdgs(doc)
    assert n_neck == 4, f"expected 4 VoVGSCSP in gsdown neck, got {n_neck}"
    # backbone 层3 = Conv[256,3,2] 下采样 → SPDConv[256]
    bb = doc["backbone"]
    assert bb[3][2] == "Conv" and bb[3][3][1:] == [3, 2], f"backbone[3] expected Conv[.,3,2], got {bb[3]}"
    c_out = bb[3][3][0]
    bb[3][2] = "SPDConv"
    bb[3][3] = [c_out]  # SPDConv args=[c2]，space-to-depth 无损下采样
    print(f"msdgs_spd_p3: neck 4×VoVGSCSP→MSDGS, backbone 层3 Conv→SPDConv[{c_out}]")
    return doc


BUILDERS = {
    "msdgs_dwr": build_msdgs_dwr_doc,
    "msdgs_spd_p3": build_msdgs_spd_p3_doc,
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
    cfg.write_text("# generated by train_msdgs_backbone_gsdown.py\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"CFG {cfg}")
    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    model = YOLO(str(cfg)).load(str(OFFICIAL_WEIGHTS))
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
        print("MSDGS_BB_RESULT", name,
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"vs_gsdown={d_gsdown:+.5f}",
              f"params_unfused={result['params_unfused']}")

    save_status("done", None, [])
    print(f"MSDGS_BB_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
