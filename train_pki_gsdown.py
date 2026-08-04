#!/usr/bin/env python
"""iter37 方案D：MSDGS135eq 基座 + backbone 层4 C3k2 -> PKIC3k2（轻量多核），250e seed0 单验。

立场（如实）：backbone 侧增强本 goal 已 13 次失败，iter29 判「小数据 backbone 多尺度增益≈0」。
D 方案属这一类，owner 明确要求试。按纪律：只替换层4、不动深层/下采样、单独 seed0 定性、
不和 head 改动混跑。一枪实验，不是主线。

reg_max 保持 1（iter36 已证 reg_max>1 在本数据崩，绝不再动）。

候选（MSDGS135eq YAML 为基座，从 yolo26n.pt 迁移，保 end2end/reg_max=1）：
- msdgs_pki_l4 : backbone 层4 C3k2[128,false,0.25] -> PKIC3k2（并联 DW 3/5/7）。

对照锚：
- gsdown 独立真值 test mAP50 0.7338 / mAP50-95 0.4018 / fused 1.936M。
- MSDGS135eq 独立真值 test 0.732411 / 0.398837 / fused 1.777M（-8.2%）。
判定：seed0 test mAP50-95 稳定 > 0.4018 才算候选，过线再上 seed0/1/2/3 配对复核；
     未过线→收口归档 iter37（预期结果，backbone 方向到顶）。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py          # GSConv/VoVGSCSP（gsdown head 必需）
    python install_msdgs_module.py            # MSDGS（基座 neck）
    python install_pki_module.py              # PKIC3k2（本轮）
    python -u train_pki_gsdown.py

输出：
- runs_pki_gsdown_e250/<name>/weights/best.pt
- generated_models_pki_gsdown_e250/<name>.yaml
- runs_pki_gsdown_e250/report.json / status.json
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
PROJECT = ROOT / "runs_pki_gsdown_e250"
GEN_DIR = ROOT / "generated_models_pki_gsdown_e250"
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


def build_pki_l4() -> dict:
    """把 backbone 层4 的 C3k2 换成 PKIC3k2。层4 = 第5个 backbone block（index 4）。

    识别：backbone 里 module=='C3k2' 且 args[0]==128（层4 是 [128,false,0.25]）。
    层2 是 [64,...]（c2=64）、层6/8 是 [128/... true]，为避免误伤只改 args==[128,False,0.25] 的第一处。
    """
    doc = _load_base()
    bb = doc.get("backbone", [])
    swapped_at = []
    for i, block in enumerate(bb):
        # block: [from, n, module, args]。YAML 标称通道未乘 width scale：
        # index2=[256,false,0.25]、index4=[512,false,0.25]、index6=[512,true]、index8=[1024,true]。
        # [512, false] 唯一定位层4（512/true 是层6）。运行时经 width=0.25 才变实际 128。
        if len(block) >= 4 and block[2] == "C3k2" and block[3] and block[3][0] == 512 \
                and len(block[3]) >= 2 and block[3][1] is False:
            block[2] = "PKIC3k2"
            swapped_at.append(i)
            break  # 只改层4（第一处 [512, false, ...]）
    assert len(swapped_at) == 1, f"expected 1 layer4 C3k2[512,false], swapped at {swapped_at}"
    print(f"swapped backbone block index {swapped_at[0]} C3k2 -> PKIC3k2 (layer4, 80x80, 并联 DW 3/5/7)")
    return doc


BUILDERS = {
    "msdgs_pki_l4": build_pki_l4,
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
    cfg.write_text("# generated by train_pki_gsdown.py\n"
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
        print("PKI_RESULT", name,
              f"val_mAP50-95={result['val']['map50_95']:.5f}",
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"params_fused={result['params_fused']}")

    save_status("done", None, [])
    print(f"PKI_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
