#!/usr/bin/env python
"""NEU-DET YOLO26n vs MSDGS neck-only 四 Seed 公平对照编排器。

- yolo26n: seed0 复用既有 best.pt；补跑 seed1/2/3。训练配方与 seed0 的 args.yaml 逐字一致：
  YOLO(yolo26n.pt) 直接加载 + pretrained=True + cls_remap=True + patience=250。
- msdgs:   四个 Seed 已训练完成（runs_multiseed_msdgs_vs_gsdown + runs_msdgs_gsdown_e250），仅独立评测。
- 全部 8 个 best.pt 由独立新进程 eval_generalization_aluminum_pcb.py 在 test split 重测总体/逐类指标
  与统一 forward-only FPS（warmup 50 / iter 200 / batch 32 / RTX 4090）。
- 汇总每模型四 Seed 统计（均值/总体标准差/最差/极差）+ 配对 MSDGS-YOLO26n 对照 + 逐类对照。

真值口径：只认独立评测报告，绝不使用训练同进程的验证指标。
口径限定：同一 NEU-DET 数据集重新训练后的多 Seed 稳定性/鲁棒性比较，非 zero-shot 泛化。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
PROJECT = ROOT / "runs_neudet_4seed_e250"
DATA = ROOT / "dataset" / "neu-det.yaml"
OFFICIAL = ROOT / "yolo26n.pt"
EVAL = ROOT / "eval_generalization_aluminum_pcb.py"
PY = sys.executable

EPOCHS, BATCH, IMGSZ, PATIENCE, WORKERS = 250, 32, 640, 250, 8
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SPLIT_COUNTS = {"train": 1200, "valid": 300, "test": 299}
ALL_SEEDS = [0, 1, 2, 3]
TRAIN_SEEDS = [1, 2, 3]

YOLO_SEED0 = ROOT / "runs_module_sweep_e250" / "y26n_base_e250" / "weights" / "best.pt"
MSDGS_SEED0 = ROOT / "runs_msdgs_gsdown_e250" / "y26n_gsdown_msdgs_135eq_e250" / "weights" / "best.pt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def yolo_weights(seed: int) -> Path:
    return YOLO_SEED0 if seed == 0 else PROJECT / f"y26n_seed{seed}" / "weights" / "best.pt"


def msdgs_weights(seed: int) -> Path:
    if seed == 0:
        return MSDGS_SEED0
    return ROOT / "runs_multiseed_msdgs_vs_gsdown" / f"msdgs135eq_seed{seed}" / "weights" / "best.pt"


def weights_for(model: str, seed: int) -> Path:
    return yolo_weights(seed) if model == "yolo26n" else msdgs_weights(seed)


def report_path(model: str, seed: int) -> Path:
    return PROJECT / f"independent_{model}_seed{seed}_report.json"


def status_write(**kw) -> None:
    atomic_json(PROJECT / "status.json", {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **kw})


# ----------------------------- build check -----------------------------
def count_images(split_dir: str) -> int:
    d = ROOT / "dataset" / split_dir / "images"
    if not d.is_dir():
        return -1
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sum(1 for p in d.iterdir() if p.suffix.lower() in exts)


def build_check() -> None:
    import yaml
    log("BUILD_CHECK start")
    doc = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    assert int(doc["nc"]) == 6, f"nc != 6: {doc['nc']}"
    raw = doc["names"]
    names = [str(raw[i]) for i in sorted(raw)] if isinstance(raw, dict) else [str(x) for x in raw]
    assert names == CLASSES, f"class mismatch: {names}"
    for split, want in (("train", 1200), ("valid", 300), ("test", 299)):
        got = count_images(split)
        assert got == want, f"{split} images {got} != {want}"
    for s in ALL_SEEDS:
        w = msdgs_weights(s)
        assert w.is_file(), f"missing MSDGS seed{s} weights {w}"
    assert YOLO_SEED0.is_file(), f"missing yolo26n seed0 weights {YOLO_SEED0}"
    assert OFFICIAL.is_file(), f"missing yolo26n.pt {OFFICIAL}"
    assert EVAL.is_file(), f"missing eval script {EVAL}"
    log("BUILD_CHECK ok: nc=6, classes match, splits 1200/300/299, all pre-trained weights + eval present")


# ----------------------------- train worker -----------------------------
def worker_train(seed: int, epochs: int = EPOCHS, name: str | None = None) -> None:
    from ultralytics import YOLO
    name = name or f"y26n_seed{seed}"
    log(f"TRAIN worker seed={seed} epochs={epochs} name={name} start")
    model = YOLO(str(OFFICIAL))
    model.train(
        data=str(DATA), epochs=epochs, patience=PATIENCE, batch=BATCH, imgsz=IMGSZ,
        device=0, workers=WORKERS, pretrained=True, cls_remap=True,
        optimizer="auto", cos_lr=False, close_mosaic=10, amp=True, rect=False, cache=False,
        seed=seed, deterministic=True,
        project=str(PROJECT), name=name, exist_ok=True,
        plots=False, verbose=False, val=True, save=True,
    )
    log(f"TRAIN worker seed={seed} done")


def csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    return max(0, len(rows) - 1)  # minus header


def seed_trained(seed: int) -> bool:
    return yolo_weights(seed).is_file() and csv_rows(PROJECT / f"y26n_seed{seed}" / "results.csv") >= EPOCHS


def train_all() -> None:
    for s in TRAIN_SEEDS:
        if seed_trained(s):
            log(f"seed{s} already complete ({csv_rows(PROJECT / f'y26n_seed{s}' / 'results.csv')} epochs), skip")
            continue
        status_write(phase="train", current=f"y26n_seed{s}")
        log(f"launch training subprocess seed={s}")
        rc = subprocess.call([PY, str(Path(__file__).resolve()), "--worker-train", "--seed", str(s)])
        if rc != 0:
            raise RuntimeError(f"train seed{s} subprocess rc={rc}")
        if not seed_trained(s):
            raise RuntimeError(f"seed{s} training incomplete after subprocess "
                               f"(rows={csv_rows(PROJECT / f'y26n_seed{s}' / 'results.csv')})")
        log(f"seed{s} training verified complete")


# ----------------------------- eval -----------------------------
def eval_one(model: str, seed: int) -> None:
    rep = report_path(model, seed)
    if rep.is_file():
        try:
            if load_json(rep).get("status") == "done":
                log(f"eval {model} seed{seed} report exists, skip")
                return
        except Exception:
            pass
    w = weights_for(model, seed)
    if not w.is_file():
        raise FileNotFoundError(f"weights missing {w}")
    status_write(phase="eval", current=f"{model}_seed{seed}")
    log(f"eval {model} seed{seed} -> {rep.name}")
    cmd = [PY, str(EVAL), "--dataset", "neudet", "--model", model, "--weights", str(w),
           "--data", str(DATA), "--output", str(rep), "--imgsz", str(IMGSZ),
           "--batch", str(BATCH), "--warmup", "50", "--iterations", "200"]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"eval {model} seed{seed} subprocess rc={rc}")
    r = load_json(rep)
    if r.get("status") != "done":
        raise RuntimeError(f"eval {model} seed{seed} status != done")
    log(f"eval {model} seed{seed} done map50={r['test']['map50']:.6f} "
        f"map50_95={r['test']['map50_95']:.6f} fps={r['benchmark']['fps']:.2f} "
        f"fused_params={r['structure']['fused_params']}")


def eval_all() -> None:
    for model in ("yolo26n", "msdgs"):
        for s in ALL_SEEDS:
            eval_one(model, s)


# ----------------------------- summarize -----------------------------
def stat_block(values):
    vals = [float(v) for v in values]
    return {
        "values": vals,
        "mean": statistics.fmean(vals),
        "std_pop": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "range": max(vals) - min(vals),
    }


def summarize() -> dict:
    log("SUMMARIZE start")
    reports = {m: {s: load_json(report_path(m, s)) for s in ALL_SEEDS} for m in ("yolo26n", "msdgs")}
    models_out = {}
    for m in ("yolo26n", "msdgs"):
        fused = {reports[m][s]["structure"]["fused_params"] for s in ALL_SEEDS}
        assert len(fused) == 1, f"{m} fused_params vary across seeds: {fused}"
        models_out[m] = {
            "fused_params": fused.pop(),
            "map50": stat_block([reports[m][s]["test"]["map50"] for s in ALL_SEEDS]),
            "map50_95": stat_block([reports[m][s]["test"]["map50_95"] for s in ALL_SEEDS]),
            "fps": stat_block([reports[m][s]["benchmark"]["fps"] for s in ALL_SEEDS]),
            "class_ap50": {c: stat_block([reports[m][s]["test"]["class_ap50"][c] for s in ALL_SEEDS]) for c in CLASSES},
            "class_ap50_95": {c: stat_block([reports[m][s]["test"]["class_ap50_95"][c] for s in ALL_SEEDS]) for c in CLASSES},
            "per_seed": {s: {
                "map50": reports[m][s]["test"]["map50"],
                "map50_95": reports[m][s]["test"]["map50_95"],
                "fps": reports[m][s]["benchmark"]["fps"],
            } for s in ALL_SEEDS},
        }

    def paired(field):
        deltas, wins = {}, 0
        for s in ALL_SEEDS:
            d = reports["msdgs"][s]["test"][field] - reports["yolo26n"][s]["test"][field]
            deltas[s] = d
            wins += 1 if d > 0 else 0
        return {"per_seed_delta": deltas, "mean_delta": statistics.fmean(list(deltas.values())), "msdgs_wins": wins}

    yp = models_out["yolo26n"]["fused_params"]
    gp = models_out["msdgs"]["fused_params"]
    yf = models_out["yolo26n"]["fps"]["mean"]
    gf = models_out["msdgs"]["fps"]["mean"]
    paired_out = {
        "map50": paired("map50"),
        "map50_95": paired("map50_95"),
        "params_reduction_abs": yp - gp,
        "params_reduction_pct": (yp - gp) / yp * 100.0,
        "fps_gain_pct": (gf - yf) / yf * 100.0,
    }

    def class_paired(field):
        out = {}
        for c in CLASSES:
            deltas, wins = [], 0
            for s in ALL_SEEDS:
                d = reports["msdgs"][s]["test"][field][c] - reports["yolo26n"][s]["test"][field][c]
                deltas.append(d)
                wins += 1 if d > 0 else 0
            out[c] = {"mean_delta": statistics.fmean(deltas), "msdgs_wins": wins}
        return out

    per_class_paired = {"class_ap50": class_paired("class_ap50"), "class_ap50_95": class_paired("class_ap50_95")}

    report = {
        "status": "done",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "neudet",
        "seeds": ALL_SEEDS,
        "protocol": {
            "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "patience": PATIENCE,
            "deterministic": True, "cache": False, "pretrained": "yolo26n.pt", "cls_remap": True,
            "truth": "independent fresh-process test eval; unified forward-only FPS warmup50 iter200 batch32 RTX4090",
            "scope": "same-dataset retrained multi-seed stability/robustness comparison; NOT zero-shot generalization",
        },
        "models": models_out,
        "paired": paired_out,
        "per_class_paired": per_class_paired,
    }
    atomic_json(PROJECT / "report.json", report)
    write_comparison_csv(reports, models_out, paired_out, per_class_paired)
    write_report_md(models_out, paired_out, per_class_paired)
    log("SUMMARIZE done -> report.json, comparison.csv, report.md")
    return report


def write_comparison_csv(reports, models_out, paired_out, per_class_paired) -> None:
    p = PROJECT / "comparison.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "yolo26n", "msdgs", "delta_msdgs_minus_yolo26n"])
        for s in ALL_SEEDS:
            y = reports["yolo26n"][s]["test"]
            g = reports["msdgs"][s]["test"]
            w.writerow([f"seed{s}", "map50", f"{y['map50']:.6f}", f"{g['map50']:.6f}", f"{g['map50'] - y['map50']:+.6f}"])
            w.writerow([f"seed{s}", "map50_95", f"{y['map50_95']:.6f}", f"{g['map50_95']:.6f}", f"{g['map50_95'] - y['map50_95']:+.6f}"])
        for metric in ("map50", "map50_95", "fps"):
            ym = models_out["yolo26n"][metric]
            gm = models_out["msdgs"][metric]
            for stat in ("mean", "std_pop", "min", "max", "range"):
                w.writerow([f"stat_{metric}", stat, f"{ym[stat]:.6f}", f"{gm[stat]:.6f}", f"{gm[stat] - ym[stat]:+.6f}"])
        w.writerow(["efficiency", "fused_params", models_out["yolo26n"]["fused_params"],
                    models_out["msdgs"]["fused_params"], -paired_out["params_reduction_abs"]])
        w.writerow(["efficiency", "params_reduction_pct", "", f"{paired_out['params_reduction_pct']:.2f}", ""])
        w.writerow(["efficiency", "fps_gain_pct", "", f"{paired_out['fps_gain_pct']:.2f}", ""])
        for c in CLASSES:
            ym = models_out["yolo26n"]["class_ap50"][c]
            gm = models_out["msdgs"]["class_ap50"][c]
            w.writerow(["class_ap50_mean", c, f"{ym['mean']:.6f}", f"{gm['mean']:.6f}",
                        f"{per_class_paired['class_ap50'][c]['mean_delta']:+.6f}"])
        for c in CLASSES:
            ym = models_out["yolo26n"]["class_ap50_95"][c]
            gm = models_out["msdgs"]["class_ap50_95"][c]
            w.writerow(["class_ap50_95_mean", c, f"{ym['mean']:.6f}", f"{gm['mean']:.6f}",
                        f"{per_class_paired['class_ap50_95'][c]['mean_delta']:+.6f}"])


def _f(x, n=6):
    return f"{x:.{n}f}"


def write_report_md(models_out, paired_out, per_class_paired) -> None:
    y = models_out["yolo26n"]
    g = models_out["msdgs"]
    L = []
    A = L.append
    A("# NEU-DET YOLO26n 与 MSDGS neck-only 四 Seed 公平对照\n")
    A(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}（远端机器时钟）\n")
    A("真值口径：独立新 Python 进程从磁盘重载每个 Seed 的 `best.pt`，在 NEU-DET test split 重测总体与逐类指标；"
      "统一 forward-only FPS（warmup 50 / iter 200 / batch 32 / RTX 4090）。"
      "训练协议 epochs=250、imgsz=640、batch=32、patience=250、deterministic=True、cache=False、pretrained=yolo26n.pt、cls_remap=True。\n")
    A("> 口径限定：同一 NEU-DET 数据集重新训练后的多 Seed 稳定性/鲁棒性比较，不表述为 zero-shot 泛化。\n")

    A("## 四 Seed 总体结果\n")
    A("| Seed | YOLO26n mAP50 | MSDGS mAP50 | Δ | YOLO26n mAP50-95 | MSDGS mAP50-95 | Δ |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    for s in ALL_SEEDS:
        ym = y["per_seed"][s]
        gm = g["per_seed"][s]
        A(f"| {s} | {_f(ym['map50'])} | {_f(gm['map50'])} | {gm['map50'] - ym['map50']:+.6f} | "
          f"{_f(ym['map50_95'])} | {_f(gm['map50_95'])} | {gm['map50_95'] - ym['map50_95']:+.6f} |")
    A("")
    A(f"MSDGS 在 mAP50 上赢得 **{paired_out['map50']['msdgs_wins']}/4** 配对 Seed，"
      f"在 mAP50-95 上赢得 **{paired_out['map50_95']['msdgs_wins']}/4**。\n")

    A("## 四 Seed 稳定性统计\n")
    A("| 模型 | mAP50 均值 | 总体标准差 | 最差 | 极差 | mAP50-95 均值 | 总体标准差 | 最差 | 极差 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in (("YOLO26n", y), ("MSDGS", g)):
        a, b = m["map50"], m["map50_95"]
        A(f"| {name} | {_f(a['mean'])} | {_f(a['std_pop'])} | {_f(a['min'])} | {_f(a['range'])} | "
          f"{_f(b['mean'])} | {_f(b['std_pop'])} | {_f(b['min'])} | {_f(b['range'])} |")
    A(f"| MSDGS-YOLO26n | {g['map50']['mean'] - y['map50']['mean']:+.6f} | — | — | — | "
      f"{g['map50_95']['mean'] - y['map50_95']['mean']:+.6f} | — | — | — |")
    A("")

    A("## 参数量与 FPS\n")
    A("| 模型 | fused params | 四 Seed FPS 均值 | FPS 标准差 | FPS 极差 |")
    A("|---|---:|---:|---:|---:|")
    A(f"| YOLO26n | {y['fused_params']:,} | {_f(y['fps']['mean'], 2)} | {_f(y['fps']['std_pop'], 2)} | {_f(y['fps']['range'], 2)} |")
    A(f"| MSDGS | {g['fused_params']:,} | {_f(g['fps']['mean'], 2)} | {_f(g['fps']['std_pop'], 2)} | {_f(g['fps']['range'], 2)} |")
    A("")
    A(f"MSDGS 相对 YOLO26n：参数减少 `{paired_out['params_reduction_abs']:,}`，即 **-{paired_out['params_reduction_pct']:.2f}%**；"
      f"同一 RTX 4090 forward-only 脚本下 FPS 均值提升 **+{paired_out['fps_gain_pct']:.2f}%**。\n")

    A("## 六类 AP50 四 Seed 对照\n")
    A("| 类别 | YOLO26n 均值±std | MSDGS 均值±std | 均值差 | MSDGS 胜出 Seed |")
    A("|---|---:|---:|---:|---:|")
    for c in CLASSES:
        ym, gm, pc = y["class_ap50"][c], g["class_ap50"][c], per_class_paired["class_ap50"][c]
        A(f"| {c} | {_f(ym['mean'])} ± {_f(ym['std_pop'])} | {_f(gm['mean'])} ± {_f(gm['std_pop'])} | "
          f"{pc['mean_delta']:+.6f} | {pc['msdgs_wins']}/4 |")
    A("")

    A("## 六类 AP50-95 四 Seed 对照\n")
    A("| 类别 | YOLO26n 均值±std | MSDGS 均值±std | 均值差 | MSDGS 胜出 Seed |")
    A("|---|---:|---:|---:|---:|")
    for c in CLASSES:
        ym, gm, pc = y["class_ap50_95"][c], g["class_ap50_95"][c], per_class_paired["class_ap50_95"][c]
        A(f"| {c} | {_f(ym['mean'])} ± {_f(ym['std_pop'])} | {_f(gm['mean'])} ± {_f(gm['std_pop'])} | "
          f"{pc['mean_delta']:+.6f} | {pc['msdgs_wins']}/4 |")
    A("")
    (PROJECT / "report.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------- main -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-train", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--build-check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    PROJECT.mkdir(parents=True, exist_ok=True)

    if args.worker_train:
        worker_train(args.seed)
        return 0
    if args.build_check:
        build_check()
        return 0
    if args.smoke:
        build_check()
        smoke_dir = PROJECT / "_smoke_y26n"
        worker_train(99, epochs=2, name="_smoke_y26n")
        ok = (smoke_dir / "weights" / "best.pt").is_file()
        shutil.rmtree(smoke_dir, ignore_errors=True)
        log(f"SMOKE {'ok' if ok else 'FAILED'}")
        return 0 if ok else 1
    if args.summarize_only:
        summarize()
        return 0
    if args.eval_only:
        build_check()
        eval_all()
        rep = summarize()
        log(f"EVAL_ONLY DONE map50_delta={rep['paired']['map50']['mean_delta']:+.6f} "
            f"map50_95_delta={rep['paired']['map50_95']['mean_delta']:+.6f}")
        return 0

    # full run
    status_write(phase="build_check", status="running")
    build_check()
    status_write(phase="train", status="running")
    train_all()
    status_write(phase="eval", status="running")
    eval_all()
    status_write(phase="summarize", status="running")
    rep = summarize()
    status_write(phase="done", status="done",
                 map50_mean_yolo26n=rep["models"]["yolo26n"]["map50"]["mean"],
                 map50_mean_msdgs=rep["models"]["msdgs"]["map50"]["mean"],
                 map50_mean_delta=rep["paired"]["map50"]["mean_delta"],
                 map50_95_mean_delta=rep["paired"]["map50_95"]["mean_delta"],
                 params_reduction_pct=rep["paired"]["params_reduction_pct"],
                 fps_gain_pct=rep["paired"]["fps_gain_pct"])
    log("ALL DONE")
    log(f"RESULT map50 yolo26n={rep['models']['yolo26n']['map50']['mean']:.6f} "
        f"msdgs={rep['models']['msdgs']['map50']['mean']:.6f} "
        f"delta={rep['paired']['map50']['mean_delta']:+.6f} | "
        f"params -{rep['paired']['params_reduction_pct']:.2f}% | "
        f"fps +{rep['paired']['fps_gain_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
