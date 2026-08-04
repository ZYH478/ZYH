# MSDGS-line SADGS（各向异性条带解耦 neck）seed0 报告

SADGS = MSDGS 单变量消融：neck 4×MSDGS(d1,d3,d5) → SADGS(d1,d3,strip)。
固定协议：250 epochs / imgsz=640 / batch=32 / seed0 / cache=False / yolo26n.pt。
最终真值由独立新进程重新加载磁盘 `best.pt`，分别评测 val/test。

| test mAP50 | test mAP50-95 | crazing mAP50-95 | fused 参数 | GFLOPs | Owner gate |
|---:|---:|---:|---:|---:|---|
| 0.727359 | 0.385251 | 0.142256 | 1777798 | None | FAIL |

## 相对 MSDGS

```json
{
  "owner_gate_pass": false,
  "owner_checks": {
    "test_map50_gt_msdgs": false,
    "test_map50_95_ge_msdgs": false,
    "test_crazing_map50_95_gt_msdgs": false
  },
  "ideal_gate_pass": false,
  "ideal_checks": {
    "test_map50_ge_0.738": false,
    "test_map50_95_ge_0.402": false,
    "test_crazing_map50_95_ge_0.187": false
  },
  "deltas_vs_msdgs": {
    "map50": -0.005052118944654738,
    "map50_95": -0.013586227644143856,
    "crazing_map50_95": -0.034814014729739745
  }
}
```
