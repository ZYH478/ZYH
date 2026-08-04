#!/usr/bin/env python
"""在远程打印 stage2 组合搜索的计划候选、已完成结果与当前进度。

只读脚本，不改动任何训练状态。远程用法：

    cd /root/autodl-tmp/neu-det-yolo26
    python inspect_stage2.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
COMBO = ROOT / "runs_module_combo_e250"
REPORT = COMBO / "combo_report.json"
STATUS = COMBO / "status.json"


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    status = load(STATUS)
    report = load(REPORT)

    print("=== stage2 status ===")
    print("status   :", status.get("status"))
    print("current  :", status.get("current"))
    print("pending  :", status.get("pending"))
    print("updated  :", status.get("updated_at"))

    plan = report.get("stage2_plan", {})
    base = plan.get("baseline", {})
    print("\n=== baseline ===")
    print(f"{base.get('name')}  val50={base.get('val_map50')}  "
          f"val95={base.get('val_map50_95')}  "
          f"test50={base.get('test_map50')}  test95={base.get('test_map50_95')}")

    print("\n=== stage1 winners (better than baseline) ===")
    for w in plan.get("stage1_winners", []):
        print(f"  {w['name']:<40} dval95={w['delta_val_map50_95']:+.4f} "
              f"dval50={w['delta_val_map50']:+.4f} score={w['score']:+.4f}")

    print("\n=== selected stage2 candidates ===")
    for i, c in enumerate(plan.get("selected_candidates", []), 1):
        print(f"  {i:>2}. {c['name']}")
        print(f"      note: {c['note']}")
        print(f"      comp: {c.get('components')}")

    exps = report.get("experiments", {})
    done = {k: v for k, v in exps.items() if v.get("status") == "done"}
    print(f"\n=== stage2 completed: {len(done)} ===")
    for name, v in done.items():
        val = v.get("val", {})
        test = v.get("test", {})
        print(f"  {name:<44} val50={val.get('map50'):.4f} "
              f"val95={val.get('map50_95'):.4f} "
              f"test50={test.get('map50'):.4f} test95={test.get('map50_95'):.4f} "
              f"params={v.get('params')}")

    best = report.get("best_summary", {})
    top = best.get("best_by_val_map50_95")
    if top:
        print("\n=== current best (stage1+stage2, by val mAP50-95) ===")
        print(f"  {top.get('name')} [{top.get('source')}] "
              f"val95={top.get('val_map50_95'):.4f} val50={top.get('val_map50'):.4f}")


if __name__ == "__main__":
    main()
