#!/usr/bin/env python
"""修正 runs_a2c2f_msdgs_e250/report.json 里被训练混乱期污染的 a2c2f test/val 数字。

背景：那次训练经历多实例抢卡→被杀→重启，report.json 记的 test map50=0.7454/
map50-95=0.3966 是某个被污染实例写入的残留，非最终 best.pt 成绩。经统一 bench +
原口径复验（verify_a2c2f_best.py），当前磁盘 best.pt（08:29:39 干净单实例产物）真实
成绩 test map50=0.70626 / map50-95=0.39035。

做法：不删原始记录（保留作证据），把污染的 test 块整体移到 _polluted_original 下，
写入真实 test 数字（来自 bench_three_models.json，同环境背靠背），并加 note。
"""
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
REPORT = ROOT / "runs_a2c2f_msdgs_e250" / "report.json"
BENCH = ROOT / "bench_three_models.json"

report = json.loads(REPORT.read_text(encoding="utf-8"))
bench = json.loads(BENCH.read_text(encoding="utf-8"))

exp = report["experiments"]["msdgs_a2c2f_bb"]
a2 = bench["models"]["a2c2f_msdgs"]["metric"]

# 保留污染原值作证据
exp["_polluted_original"] = {
    "note": "训练混乱期(多实例抢卡→被杀→重启)某污染实例写入的残留 test/val，非最终 best.pt 成绩",
    "test": exp.get("test"),
    "val": exp.get("val"),
}

# 用统一 bench(同环境背靠背, 与原口径复验一致)的真实 test 覆盖
exp["test"] = {
    "map50": a2["map50"],
    "map50_95": a2["map50_95"],
    "precision": a2["precision"],
    "recall": a2["recall"],
    "per_class": a2["per_class"],
    "source": "bench_three_models.json 2026-07-29 同环境背靠背; 已由 verify_a2c2f_best.py 原口径(batch=32)复验一致",
}
exp["corrected_note"] = ("test 已修正为干净 best.pt 真实成绩 map50=0.70626/map50-95=0.39035; "
                         "原污染值(0.74544/0.39662)见 _polluted_original")

REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("FIXED", REPORT)
print("test.map50", exp["test"]["map50"], "test.map50_95", exp["test"]["map50_95"])
