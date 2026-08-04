#!/usr/bin/env python
"""在 UWWT-Dataset-1500（5类超声波焊缝缺陷）上用与 NEU-DET goal 相同的官方配方
（250e / batch32 / imgsz640 / seed0，从 yolo26n.pt 迁移）训练 6 个架构做对比：

- base                : 官方 yolo26n
- gsdown              : VoVGSCSP + GSConv 下采样（轻量轴）
- winner              : SPD_P3 + DySample（NEU 精度轴）
- dwr_deep            : base backbone 深层 C3k2(6/8) -> DWRC3k2
- winner_dwr          : winner + backbone 深层 DWR(6/8)
- winner_pki_dwr      : winner + 浅层 PKI(2/4) + 深层 DWR(6/8)

自定义模块（SPD/DySample/DWR/PKI）依赖 install_yolo26_exp_modules.py +
install_backbone_modules.py 先注入。单候选崩溃 continue 隔离（iter18 教训）。
训练后独立 val + test 评估，落 report.json（含 per-class）。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py
    python install_backbone_modules.py
    python -u train_uwwt.py
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
DS_DIR = ROOT / "UWWT-Dataset-1500"
DATA_ABS = ROOT / "uwwt_data_abs.yaml"
OFFICIAL_WEIGHTS = Path(os.environ.get("YOLO26_EXP_WEIGHTS", ROOT / "yolo26n.pt"))
PROJECT = ROOT / "runs_uwwt_e250"
STATUS_JSON = PROJECT / "status.json"
REPORT_JSON = PROJECT / "uwwt_report.json"

# id -> 架构 YAML（base 用内置 yolo26n.yaml 字符串）
SPECS = {
    "uwwt_base": "yolo26n.yaml",
    "uwwt_gsdown": str(ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"),
    "uwwt_winner": str(ROOT / "generated_models_module_combo2_e250" / "y26n_s2_spd_p3_dysample_e250.yaml"),
    "uwwt_dwr_deep": str(ROOT / "generated_models_iter19_backbone_e250" / "y26n_i19_dwr_deep_e250.yaml"),
    "uwwt_winner_dwr": str(ROOT / "generated_models_iter21_combo_e250" / "y26n_i21_winner_dwr_e250.yaml"),
    "uwwt_winner_pki_dwr": str(ROOT / "generated_models_iter21_combo_e250" / "y26n_i21_winner_pki_dwr_e250.yaml"),
}

EPOCHS = 250
BATCH = 32
IMGSZ = 640
SEED = 0
DEVICE = 0


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_data_abs() -> Path:
    """生成绝对路径 data yaml（原 data.yaml 的 path 是 '.'，ultralytics 会误解析）。"""
    src = yaml.safe_load((DS_DIR / "data.yaml").read_text(encoding="utf-8"))
    src["path"] = str(DS_DIR)
    DATA_ABS.write_text(yaml.safe_dump(src, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return DATA_ABS


def read_json(p: Path, default):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def extract(res) -> dict:
    box = res.box
    ap50, ap5095 = {}, {}
    for i, ci in enumerate(box.ap_class_index):
        name = str(res.names[int(ci)])
        _, _, a50, a95 = box.class_result(i)
        ap50[name] = float(a50)
        ap5095[name] = float(a95)
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "class_ap50": ap50,
        "class_ap50_95": ap5095,
    }


def fused_params(weights: Path) -> int:
    ym = YOLO(str(weights))
    m = ym.model
    if hasattr(m, "fuse"):
        m = m.fuse()
    return int(sum(p.numel() for p in m.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    data = make_data_abs()
    print(f"DATA_ABS {data}")
    PROJECT.mkdir(parents=True, exist_ok=True)
    report = read_json(REPORT_JSON, {"dataset": "UWWT-Dataset-1500", "recipe": f"official {args.epochs}e/batch{BATCH}/seed{SEED} from yolo26n.pt", "experiments": {}})

    for name, model_cfg in SPECS.items():
        if args.only and name != args.only:
            continue
        if report["experiments"].get(name, {}).get("status") == "done":
            print(f"SKIP {name} (already done)")
            continue
        write_json(STATUS_JSON, {"current": name, "at": now(), "stage": "uwwt"})
        try:
            print(f"TRAIN_START {name} model={model_cfg} epochs={args.epochs} batch={BATCH}")
            model = YOLO(model_cfg)
            model.train(
                data=str(data), epochs=args.epochs, batch=BATCH, imgsz=IMGSZ,
                device=DEVICE, seed=SEED, pretrained=str(OFFICIAL_WEIGHTS),
                project=str(PROJECT), name=name, exist_ok=True, plots=False,
                save=True, verbose=False, cache="ram",
            )
            run_dir = Path(str(model.trainer.save_dir)).resolve()
            best = run_dir / "weights" / "best.pt"
            del model

            vm = YOLO(str(best)).val(data=str(data), split="val", imgsz=IMGSZ, batch=BATCH, device=DEVICE, plots=False, verbose=False)
            val_metrics = extract(vm)
            tm = YOLO(str(best)).val(data=str(data), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE, plots=False, verbose=False)
            test_metrics = extract(tm)
            params = fused_params(best)

            report["experiments"][name] = {
                "status": "done",
                "model": model_cfg,
                "weights": str(best),
                "fused_params": params,
                "val": val_metrics,
                "test": test_metrics,
                "finished_at": now(),
            }
            write_json(REPORT_JSON, report)
            print(f"UWWT_RESULT {name} val_mAP50={val_metrics['map50']:.4f} val_mAP50-95={val_metrics['map50_95']:.4f} "
                  f"test_mAP50={test_metrics['map50']:.4f} test_mAP50-95={test_metrics['map50_95']:.4f} params={params}")
        except Exception as exc:
            report["experiments"][name] = {"status": "failed", "model": model_cfg, "error": repr(exc), "at": now()}
            write_json(REPORT_JSON, report)
            print(f"CANDIDATE_FAILED {name}: {exc!r} -- continue to next")
            continue

    write_json(STATUS_JSON, {"current": None, "status": "done", "at": now(), "stage": "uwwt"})
    print(f"UWWT_DONE report={REPORT_JSON}")


if __name__ == "__main__":
    main()
