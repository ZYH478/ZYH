# MSDGS Backbone P4 Stable-DCNv2 seed0/seed3 报告

固定协议：250 epochs / imgsz=640 / batch=32 / cache=False / resume=False / yolo26n.pt。
稳定化：epoch 0-20 strength=0，20-80 线性升至1；offset范围[-1.5,1.5]；mask范围[0.5,1.5]；offset/mask梯度0.25×。
所有最终指标由独立新进程从磁盘重新加载各 seed 的 best.pt 后评测。crazing 完整报告，但不作为本轮硬否决项。

| seed | test mAP50 | test mAP50-95 | crazing mAP50-95 | 核心双指标 | seed3诊断 |
|---:|---:|---:|---:|---|---|
| 0 | 0.717540 | 0.386830 | 0.128405 | FAIL | - |
| 3 | 0.713667 | 0.390467 | 0.113929 | FAIL | FAIL |

## 汇总

```json
{
  "n": 2,
  "seeds": [
    0,
    3
  ],
  "test_map50": {
    "n": 2,
    "mean": 0.715603195164298,
    "std": 0.002738369321618096,
    "min": 0.7136668756475887,
    "max": 0.7175395146810074,
    "range": 0.0038726390334187233
  },
  "test_map50_95": {
    "n": 2,
    "mean": 0.38864879596984087,
    "std": 0.0025720233676439393,
    "min": 0.3868301008052096,
    "max": 0.39046749113447216,
    "range": 0.0036373903292625798
  },
  "test_crazing_map50_95": {
    "n": 2,
    "mean": 0.12116713480546736,
    "std": 0.010236203934249533,
    "min": 0.1139290455899511,
    "max": 0.12840522402098362,
    "range": 0.014476178431032521
  },
  "core_pass_count": 0,
  "seed0_core_pass": false,
  "seed3_diagnostic_pass": false,
  "continue_to_seed1_seed2": false
}
```
