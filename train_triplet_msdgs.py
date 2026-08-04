#!/usr/bin/env python
"""iter39-B：MSDGS135eq 基座 + 检测头三分支前各插 1 个 TripletAttention，250e seed0 单验。

来源：MSAF-YOLO (Measurement 2026) 引用的 Triplet Attention (Misra et al., WACV 2021)。
在 MSAF 消融里它是唯一「涨精度又降参」的模块（+1.3 mAP 且 params 2.46M→1.9M）。

落点（MSDGS 的 neck 全是 MSDGS 模块、无 C3k2，故不能像 MSAF 插进 C3k2）：
- 在三个检测分支输入前（head 的 layer 16/19/22 = 三个尺度的 MSDGS 输出）后各追加一个
  TripletAttention，Detect 从新的索引取。channel-preserving，YAML args=[]。
- 追加 3 个 block 会使其后所有层索引 +1/+2/+3，故需重写所有绝对 from 引用与 Detect 索引。

reg_max 保持 1，end2end 保持。

候选（MSDGS135eq YAML 为基座，从 yolo26n.pt 迁移）：
- msdgs_triplet : head 三检测尺度输出后各插 1 个 TripletAttention。

对照锚：
- gsdown 独立真值 test mAP50 0.7338 / mAP50-95 0.4018 / fused 1.936M。
- MSDGS135eq 独立真值 test 0.732411 / 0.398837 / fused 1.777M（-8.2%）。
判定：seed0 test mAP50-95 稳定 > 0.4018 才算候选，过线再上 seed0/1/2/3 配对复核；
     未过线→收口归档。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_gsconv_modules.py
    python install_msdgs_module.py
    python install_triplet_module.py
    python -u train_triplet_msdgs.py

输出：
- runs_triplet_msdgs_e250/<name>/weights/best.pt
- generated_models_triplet_msdgs_e250/<name>.yaml
- runs_triplet_msdgs_e250/report.json / status.json
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
PROJECT = ROOT / "runs_triplet_msdgs_e250"
GEN_DIR = ROOT / "generated_models_triplet_msdgs_e250"
REPORT_JSON = PROJECT / "report.json"
STATUS_JSON = PROJECT / "status.json"

NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

GSDOWN_TRUTH = {
    "test_map50": 0.7338, "test_map50_95": 0.4018, "fused_params": 1935814,
}
MSDGS_TRUTH = {"test_map50": 0.732411, "test_map50_95": 0.398837, "fused_params": 1777318}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_base() -> dict:
    return yaml.safe_load(MSDGS_YAML.read_text(encoding="utf-8"))


def build_triplet() -> dict:
    """在 head 的三个检测尺度输出（MSDGS 层 16/19/22）后各插一个 TripletAttention。

    MSDGS YAML 结构（0-indexed）：backbone 0..10（11 个 block），head 11..22（12 个 block），
    Detect 在 index 23，from=[16,19,22]。head 里三处 MSDGS 输出分别是 index 16/19/22。

    做法：在 16/19/22 之后各插一个 TripletAttention（from=-1，channel-preserving，args=[]）。
    插入后索引偏移：原 16 后插 -> 新增 index 17；原 19（现 20）后插 -> 新增 21；
    原 22（现 24）后插 -> 新增 25。Detect 从三个新 TripletAttention 取。

    为避免手工重算所有 from 引用出错，这里用「基于原始索引的稳健重建」：
    先把原 blocks 复制，逐个在目标位置后插入，同时维护 old->new 索引映射，
    最后把所有绝对整数 from 引用（含 Concat 的多源）按映射改写，Detect 指向 3 个 TA。
    """
    doc = _load_base()
    bb = doc.get("backbone", [])
    head = doc.get("head", [])
    n_bb = len(bb)  # backbone block 数（绝对索引 0..n_bb-1）
    all_blocks = bb + head
    n_all = len(all_blocks)

    # 找 Detect，确认三个检测源（应为 head 里三处 MSDGS 输出的绝对索引）
    detect_idx = None
    for i, blk in enumerate(all_blocks):
        if blk[2] == "Detect":
            detect_idx = i
            break
    assert detect_idx is not None, "no Detect found"
    detect_from = list(all_blocks[detect_idx][3]) if isinstance(all_blocks[detect_idx][3], list) else None
    # Detect 的 from 在 args? 不——from 是 block[0]。重取。
    detect_sources = all_blocks[detect_idx][0]
    assert isinstance(detect_sources, list) and len(detect_sources) == 3, \
        f"Detect from expected 3 sources, got {detect_sources}"
    targets = sorted(int(s) if s >= 0 else n_all + s for s in detect_sources)
    print(f"detect_sources(abs)={detect_sources} -> targets={targets}, n_bb={n_bb}, n_all={n_all}")

    # 逐块重建，在每个 target 之后插入 TripletAttention；维护 old->new 映射
    new_blocks = []
    old2new = {}  # 原绝对索引 -> 新绝对索引
    ta_new_idx = {}  # 原 target 索引 -> 其后 TA 的新绝对索引
    for old_i, blk in enumerate(all_blocks):
        new_blocks.append([x for x in blk])  # 浅拷贝一层
        old2new[old_i] = len(new_blocks) - 1
        if old_i in targets:
            # 插入 TripletAttention，from=-1（接在刚加的 block 之后），n=1，args=[]
            new_blocks.append([-1, 1, "TripletAttention", []])
            ta_new_idx[old_i] = len(new_blocks) - 1

    # 改写所有绝对 from 引用（>=0 的整数）按 old2new 映射；-1 之类相对引用不动
    def remap_from(frm):
        if isinstance(frm, list):
            return [remap_from(f) for f in frm]
        if isinstance(frm, int) and frm >= 0:
            return old2new[frm]
        return frm

    for nb in new_blocks:
        nb[0] = remap_from(nb[0])

    # 找到重建后的 Detect（module=='Detect'），把它的三个源改成三个 TA 的新索引
    for nb in new_blocks:
        if nb[2] == "Detect":
            nb[0] = [ta_new_idx[t] for t in targets]
            break

    # 拆回 backbone/head：插入都发生在 head 段（targets 都在 head 里），
    # 但为稳健起见，用「新的 backbone 长度 = 原 backbone 段在 new_blocks 中的块数」重新切分。
    # backbone 段没有插入（targets 全 >= n_bb），故前 n_bb 块仍是 backbone。
    new_n_bb = old2new[n_bb - 1] + 1
    doc["backbone"] = new_blocks[:new_n_bb]
    doc["head"] = new_blocks[new_n_bb:]
    n_ta = sum(1 for nb in new_blocks if nb[2] == "TripletAttention")
    print(f"inserted {n_ta} TripletAttention; new total blocks={len(new_blocks)}; "
          f"Detect from={[nb[0] for nb in new_blocks if nb[2]=='Detect']}")
    assert n_ta == 3, f"expected 3 TripletAttention, got {n_ta}"
    return doc


BUILDERS = {
    "msdgs_triplet": build_triplet,
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
    cfg.write_text("# generated by train_triplet_msdgs.py\n"
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
        print("TRIPLET_RESULT", name,
              f"val_mAP50-95={result['val']['map50_95']:.5f}",
              f"test_mAP50={result['test']['map50']:.5f}",
              f"test_mAP50-95={result['test']['map50_95']:.5f}",
              f"params_fused={result['params_fused']}")

    save_status("done", None, [])
    print(f"TRIPLET_DONE report={REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
