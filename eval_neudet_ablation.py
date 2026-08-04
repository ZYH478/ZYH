#!/usr/bin/env python
"""NEU-DET seed0 消融：统一协议独立复评（component progression + MSDGS 设计选择）。

真值口径与主实验一致：独立新进程重载 best.pt -> test split 评测 + 统一 forward-only FPS。
全部 NEU-DET seed0，可与主实验 seed0（baseline / msdgs_135eq）直接并列。
复用主实验已产出的 baseline 与 135eq seed0 独立报告，不重复评测。
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
ABL = ROOT / "runs_neudet_ablation"
MAIN = ROOT / "runs_neudet_4seed_e250"
EVAL = ROOT / "eval_generalization_aluminum_pcb.py"
DATA = ROOT / "dataset" / "neu-det.yaml"
PY = sys.executable

# label -> (weights_relpath, dilations, split, group, pretty)
VARIANTS = [
    ("gsdown",         "runs_module_stage3_e250/y26n_s3_vovgscsp_gsdown_e250/weights/best.pt", "-",       "-",     "component", "GSConv-down + VoVGSCSP (Slim-neck)"),
    ("db_d2",          "runs_dualbranch_gsdown_e250/y26n_gsdown_db_d2_e250/weights/best.pt",   "{1,2}",   "equal", "dilation",  "DualBranchGS dil {1,2}"),
    ("db_d3",          "runs_dualbranch_gsdown_e250/y26n_gsdown_db_d3_e250/weights/best.pt",   "{1,3}",   "equal", "dilation",  "DualBranchGS dil {1,3}"),
    ("db_d5",          "runs_dualbranch_gsdown_e250/y26n_gsdown_db_d5_e250/weights/best.pt",   "{1,5}",   "equal", "dilation",  "DualBranchGS dil {1,5}"),
    ("msdgs_13local",  "runs_msdgs_gsdown_e250/y26n_gsdown_msdgs_13local_e250/weights/best.pt","{1,3}",   "local", "design",    "MSDGS {1,3} local-split"),
    ("msdgs_135local", "runs_msdgs_gsdown_e250/y26n_gsdown_msdgs_135local_e250/weights/best.pt","{1,3,5}","local", "design",    "MSDGS {1,3,5} local-split"),
]
# references reused from main run (seed0), do not re-eval:
REF = {
    "baseline":     (MAIN / "independent_yolo26n_seed0_report.json", "-",       "-",     "component", "YOLO26n baseline (C3k2 neck + Conv-down)"),
    "msdgs_135eq":  (MAIN / "independent_msdgs_seed0_report.json",   "{1,3,5}", "equal", "design",    "MSDGS {1,3,5} equal (OURS)"),
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def report_path(label): return ABL / f"report_{label}.json"


def eval_one(label, rel):
    out = report_path(label)
    if out.is_file():
        try:
            if load(out).get("status") == "done":
                log(f"{label} exists, skip"); return
        except Exception:
            pass
    w = ROOT / rel
    if not w.is_file():
        raise FileNotFoundError(f"{label}: missing {w}")
    log(f"eval {label} -> {out.name}")
    cmd = [PY, str(EVAL), "--dataset", "neudet_abl", "--model", "msdgs", "--weights", str(w),
           "--data", str(DATA), "--output", str(out), "--imgsz", "640", "--batch", "32",
           "--warmup", "50", "--iterations", "200"]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"{label} eval rc={rc}")
    r = load(out)
    log(f"{label} done params={r['structure']['fused_params']} map50={r['test']['map50']:.6f} "
        f"map50_95={r['test']['map50_95']:.6f} fps={r['benchmark']['fps']:.1f}")


def summarize():
    rows = []
    # references first
    for label, (p, dil, split, group, pretty) in REF.items():
        r = load(p)
        rows.append((label, group, pretty, dil, split, r["structure"]["fused_params"],
                     r["test"]["map50"], r["test"]["map50_95"], r["benchmark"]["fps"]))
    for label, rel, dil, split, group, pretty in VARIANTS:
        r = load(report_path(label))
        rows.append((label, group, pretty, dil, split, r["structure"]["fused_params"],
                     r["test"]["map50"], r["test"]["map50_95"], r["benchmark"]["fps"]))
    base = next(x for x in rows if x[0] == "baseline")
    base_p = base[5]
    obj = {"status": "done", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "dataset": "neudet", "seed": 0, "protocol": "independent test eval + unified forward-only FPS",
           "rows": [dict(label=r[0], group=r[1], config=r[2], dilations=r[3], split=r[4],
                         fused_params=r[5], map50=r[6], map50_95=r[7], fps=r[8],
                         params_delta_pct_vs_yolo26n=(r[5]-base_p)/base_p*100.0) for r in rows]}
    (ABL).mkdir(parents=True, exist_ok=True)
    (ABL / "ablation.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# NEU-DET seed0 消融（统一独立复评协议）\n",
         f"生成 {time.strftime('%Y-%m-%d %H:%M:%S')}；真值=独立进程 test split 复评 + forward-only FPS。全部 seed0。\n",
         "| 配置 | 组 | dilations | split | fused params | Δparams vs YOLO26n | mAP50 | mAP50-95 | FPS |",
         "|---|---|---|---|---:|---:|---:|---:|---:|"]
    order = {"component": 0, "dilation": 1, "design": 2}
    for r in sorted(rows, key=lambda x: (order.get(x[1], 9),)):
        L.append(f"| {r[2]} | {r[1]} | {r[3]} | {r[4]} | {r[5]:,} | "
                 f"{(r[5]-base_p)/base_p*100.0:+.2f}% | {r[6]:.6f} | {r[7]:.6f} | {r[8]:.1f} |")
    (ABL / "ablation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("SUMMARY -> ablation.json, ablation.md")
    for r in rows:
        log(f"  {r[2]:44s} p={r[5]:>9,} ({(r[5]-base_p)/base_p*100:+6.2f}%) mAP50={r[6]:.4f} mAP50-95={r[7]:.4f} fps={r[8]:.0f}")


def main():
    ABL.mkdir(parents=True, exist_ok=True)
    for label, rel, *_ in VARIANTS:
        eval_one(label, rel)
    summarize()
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
