#!/usr/bin/env python
"""iter32 诊断（不训练）：拆解 crazing 的 map50 病根——漏检 / 误报 / 类间混淆。

四轮结构改进（backbone×3 / neck / DySnake / UBHead）全部证伪，且 UBHead 暴露
map50 与 map50-95 反向（map50-95 微涨、map50 退步）。说明我们可能一直在治错的病：
一直在攻定位精度（map50-95），但 map50 的病根从未诊断过。

map50 低只有三种可能：漏检（recall 低）、误报（precision 低）、类间混淆（判成别的类）。
本脚本用已有 best.pt 在 test split 跑一次 val，导出：
1. 混淆矩阵（原始计数）——看 crazing 的 GT 被判成什么、预测为 crazing 的来自什么
2. 每类 P/R——区分漏检 vs 误报
3. crazing 专项拆解：召回缺口去哪了（背景漏检 or 判成其它类）

用法（远程）：
    python diag_confusion.py <weights.pt> <tag> [split]
默认 split=test。gsdown 系需先注入 GSConv/VoVGSCSP 模块链。
"""
from __future__ import annotations

import json
import sys

import numpy as np
from ultralytics import YOLO

ROOT = "/root/autodl-tmp/neu-det-yolo26"
DATA = f"{ROOT}/dataset/neu-det.yaml"
NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def main() -> int:
    weights = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "diag"
    split = sys.argv[3] if len(sys.argv) > 3 else "test"

    model = YOLO(weights)
    m = model.val(data=DATA, split=split, imgsz=640, batch=32, device=0,
                  project=f"{ROOT}/runs", name=f"diagcm_{tag}", exist_ok=True, verbose=False)

    # 混淆矩阵：ultralytics ConfusionMatrix.matrix 形状 (nc+1, nc+1)
    # matrix[i][j] = GT 类别 j 被预测成类别 i 的数量；最后一 index 是 background。
    cm = m.confusion_matrix.matrix  # np.ndarray (7,7)
    labels = NAMES + ["background"]

    print("CONFUSION_MATRIX_START")
    print("rows=predicted, cols=ground_truth (last=background)")
    header = "pred\\gt".ljust(16) + "".join(f"{c[:8]:>10}" for c in labels)
    print(header)
    for i, row_name in enumerate(labels):
        row = "".join(f"{int(cm[i][j]):>10}" for j in range(len(labels)))
        print(f"{row_name:<16}{row}")
    print("CONFUSION_MATRIX_END")

    # 每类 P/R（来自 val 结果）
    print("PER_CLASS_PR_START")
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        name = NAMES[int(c)] if int(c) < len(NAMES) else str(int(c))
        p = float(m.box.p[i]); r = float(m.box.r[i])
        ap50 = float(m.box.ap50[i]); ap = float(m.box.ap[i])
        per_class[name] = {"precision": p, "recall": r, "map50": ap50, "map50_95": ap}
        print(f"  {name:<16} P={p:.4f} R={r:.4f} map50={ap50:.4f} map50-95={ap:.4f}")
    print("PER_CLASS_PR_END")

    # crazing 专项拆解：GT=crazing 那一列，去向分布
    ci = NAMES.index("crazing")
    gt_crazing_col = cm[:, ci]  # 所有 GT=crazing 的样本被预测成各类的数量
    total_gt_crazing = gt_crazing_col.sum()
    print("CRAZING_BREAKDOWN_START")
    print(f"  total GT crazing detections accounted: {int(total_gt_crazing)}")
    if total_gt_crazing > 0:
        for i, row_name in enumerate(labels):
            cnt = int(gt_crazing_col[i])
            if cnt > 0:
                frac = cnt / total_gt_crazing
                tag2 = "  <-- correct" if i == ci else ("  <-- MISSED (bg)" if row_name == "background" else "  <-- confused")
                print(f"    predicted as {row_name:<16}: {cnt:>4} ({frac:.1%}){tag2}")
    # 预测为 crazing 的来自哪里（误报来源）
    pred_crazing_row = cm[ci, :]
    total_pred_crazing = pred_crazing_row.sum()
    print(f"  total predicted-as-crazing: {int(total_pred_crazing)}")
    if total_pred_crazing > 0:
        for j, col_name in enumerate(labels):
            cnt = int(pred_crazing_row[j])
            if cnt > 0:
                frac = cnt / total_pred_crazing
                tag2 = "  <-- correct" if j == ci else ("  <-- FALSE POS (from bg)" if col_name == "background" else "  <-- confused-in")
                print(f"    from GT {col_name:<16}: {cnt:>4} ({frac:.1%}){tag2}")
    print("CRAZING_BREAKDOWN_END")

    out = {"tag": tag, "split": split, "per_class": per_class,
           "confusion_matrix": cm.tolist(), "labels": labels}
    print("DIAG_JSON_START")
    print(json.dumps(out, ensure_ascii=False))
    print("DIAG_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
