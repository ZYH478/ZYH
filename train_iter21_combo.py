#!/usr/bin/env python
"""iter21：把 iter19 有效的 backbone 改进(PKI/DWR)叠到两个赢家基础上，对照择优。

owner 要求：iter19 的 backbone 候选结合到 vovgscsp_gsdown 和赢家 SPD_P3+DySample
基础上训练，观察结果是否变好。

已跑过不重复：
- gsdown + DWR = iter20 fail(真值 0.3895，-1.82pp)。

本轮 4 个组合(避开高风险 dcnv2，dwr×gsdown 已跑)：
- gsdown_pki      : gsdown backbone 浅中层 C3k2(2,4) -> PKIC3k2
- gsdown_pki_dwr  : gsdown backbone 浅PKI(2,4)+深DWR(6,8)
- winner_dwr      : winner backbone 深层 C3k2(6,8) -> DWRC3k2
- winner_pki_dwr  : winner backbone 浅PKI(2,4)+深DWR(6,8)

关键结构事实(已核实)：
- gsdown backbone(层0-10)== base，改动全在 head(VoVGSCSP+GSConv)。
- winner backbone 层3=SPDConv，head 用 DySample 上采样；浅中层2/4、深层6/8 仍是标准 C3k2。
- PKI(2,4)/DWR(6,8) 改动点与 gsdown 的 head 改动、winner 的 SPD(层3)+DySample(head)
  都不重叠 → 可干净正交叠加。

做法：直接加载已验证正确的赢家 YAML，只替换 backbone 指定层的 C3k2，其余一字不动。
从 yolo26n.pt 迁移，保 end2end/reg_max=1。单候选崩溃 continue 不拖垮整批(iter18 教训)。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py
    python install_backbone_modules.py
    python -u train_iter21_combo.py
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
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_iter21_combo_e250"
GEN_DIR = ROOT / "generated_models_iter21_combo_e250"
STATUS_JSON = PROJECT / "status.json"
REPORT_JSON = PROJECT / "combo_report.json"

GSDOWN_YAML = ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"
WINNER_YAML = ROOT / "generated_models_module_combo2_e250" / "y26n_s2_spd_p3_dysample_e250.yaml"

# 候选：name -> (base_yaml, {layer_idx: new_module})
SPECS = {
    "y26n_i21_gsdown_pki_e250": (GSDOWN_YAML, {2: "PKIC3k2", 4: "PKIC3k2"}),
    "y26n_i21_gsdown_pki_dwr_e250": (GSDOWN_YAML, {2: "PKIC3k2", 4: "PKIC3k2", 6: "DWRC3k2", 8: "DWRC3k2"}),
    "y26n_i21_winner_dwr_e250": (WINNER_YAML, {6: "DWRC3k2", 8: "DWRC3k2"}),
    "y26n_i21_winner_pki_dwr_e250": (WINNER_YAML, {2: "PKIC3k2", 4: "PKIC3k2", 6: "DWRC3k2", 8: "DWRC3k2"}),
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def build_doc(base_yaml: Path, swaps: dict[int, str]) -> dict:
    doc = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    bb = doc["backbone"]
    done = []
    for i, mod in swaps.items():
        if bb[i][2] == "C3k2":
            bb[i][2] = mod
            done.append((i, mod))
        else:
            raise SystemExit(f"layer {i} is {bb[i][2]}, not C3k2 (base={base_yaml.name})")
    print(f"SWAPPED {base_yaml.name}: {done}")
    return doc


def save_status(status: str, current, pending) -> None:
    STATUS_JSON.write_text(json.dumps({
        "status": status, "current": current, "pending": pending,
        "updated_at": now(), "stage": "iter21_combo",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def run_one(name: str, base_yaml: Path, swaps: dict[int, str], epochs: int, batch: int, imgsz: int, seed: int) -> dict:
    doc = build_doc(base_yaml, swaps)
    cfg_path = GEN_DIR / f"{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"INIT_WEIGHTS {name} <- {OFFICIAL_WEIGHTS}")
    model = YOLO(str(cfg_path)).load(str(OFFICIAL_WEIGHTS))
    print(f"TRAIN_START {name} epochs={epochs} batch={batch} imgsz={imgsz}")
    model.train(
        data=str(DATA), epochs=epochs, imgsz=imgsz, batch=batch,
        workers=8, seed=seed, device=0, project=str(PROJECT), name=name,
        exist_ok=True, patience=max(epochs, 250), cache=False, verbose=True,
    )
    print(f"TRAIN_DONE {name}")

    weights = PROJECT / name / "weights" / "best.pt"
    best = YOLO(str(weights))
    v = best.val(data=str(DATA), split="val", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_val", exist_ok=True, verbose=False)
    t = best.val(data=str(DATA), split="test", imgsz=imgsz, batch=batch, device=0,
                 project=str(PROJECT), name=f"{name}_test", exist_ok=True, verbose=False)
    best.model.fuse()
    params = sum(p.numel() for p in best.model.parameters())
    return {
        "status": "done",
        "val_map50": float(v.box.map50), "val_map50_95": float(v.box.map),
        "test_map50": float(t.box.map50), "test_map50_95": float(t.box.map),
        "fused_params": int(params),
        "weights": str(weights), "cfg": str(cfg_path),
        "finished_at": now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT.mkdir(parents=True, exist_ok=True)

    selected = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else list(SPECS)
    unknown = [x for x in selected if x not in SPECS]
    if unknown:
        raise SystemExit(f"Unknown: {unknown}")

    if args.dry:
        for name in selected:
            base_yaml, swaps = SPECS[name]
            build_doc(base_yaml, swaps)
        return 0

    report = {}
    if REPORT_JSON.exists():
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    report.setdefault("experiments", {})

    for pos, name in enumerate(selected, 1):
        if report["experiments"].get(name, {}).get("status") == "done":
            print(f"SKIP_DONE {name}")
            continue
        base_yaml, swaps = SPECS[name]
        save_status("running", name, selected[pos:])
        try:
            result = run_one(name, base_yaml, swaps, args.epochs, args.batch, args.imgsz, args.seed)
        except Exception as exc:  # noqa: BLE001
            # 单候选崩溃只记录并跳过，不拖垮整批(iter18 教训)。
            report["experiments"][name] = {"status": "failed", "error": repr(exc), "failed_at": now()}
            REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"CANDIDATE_FAILED {name}: {exc!r} -- continue to next")
            continue
        report["experiments"][name] = result
        REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("I21_RESULT", name,
              f"val_mAP50={result['val_map50']:.5f}",
              f"val_mAP50-95={result['val_map50_95']:.5f}",
              f"test_mAP50-95={result['test_map50_95']:.5f}",
              f"fused_params={result['fused_params']}")

    save_status("done", None, [])
    print(f"I21_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
