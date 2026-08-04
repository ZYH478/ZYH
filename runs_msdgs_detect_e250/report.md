# MSDGS P5 定向 Detect 实验报告

更新时间：2026-07-29 15:24:55

固定协议：250 epochs / imgsz 640 / batch 32 / cache=False / yolo26n.pt / NEU-DET；最终指标均由独立新进程重载 `best.pt`。

## 总体结果

| 候选 | Seed | Split | Precision | Recall | mAP50 | mAP50-95 | fused 参数 | fused GFLOPs | infer ms | FPS | Seed0 晋级 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| msdgs_p4p5_gate | 0 | val | 0.727509 | 0.653889 | 0.699687 | 0.390489 | 1811623 | 4.109568 | 1.993301 | 501.68 | FAIL |
| msdgs_p4p5_gate | 0 | test | 0.725490 | 0.676969 | 0.710660 | 0.382435 | 1811623 | 4.109568 | 1.901362 | 525.94 | FAIL |
| msdgs_p5box32 | 0 | val | 0.722142 | 0.675021 | 0.726451 | 0.397163 | 1821190 | 4.117504 | 2.328341 | 429.49 | FAIL |
| msdgs_p5box32 | 0 | test | 0.731800 | 0.713924 | 0.726601 | 0.392852 | 1821190 | 4.117504 | 1.527562 | 654.64 | FAIL |
| msdgs_p5cafm_gate | 0 | val | 0.771536 | 0.656922 | 0.730360 | 0.406074 | 1954215 | 4.223949 | 2.312394 | 432.45 | FAIL |
| msdgs_p5cafm_gate | 0 | test | 0.747715 | 0.666995 | 0.705787 | 0.381818 | 1954215 | 4.223949 | 1.797427 | 556.35 | FAIL |

## Seed0 六类 AP

| 候选 | Split | 类别 | AP50 | AP50-95 |
|---|---|---|---:|---:|
| msdgs_p4p5_gate | val | crazing | 0.430472 | 0.162808 |
| msdgs_p4p5_gate | val | inclusion | 0.794190 | 0.422731 |
| msdgs_p4p5_gate | val | patches | 0.867295 | 0.557492 |
| msdgs_p4p5_gate | val | pitted_surface | 0.712741 | 0.421219 |
| msdgs_p4p5_gate | val | rolled-in_scale | 0.515609 | 0.244570 |
| msdgs_p4p5_gate | val | scratches | 0.877811 | 0.534113 |
| msdgs_p4p5_gate | test | crazing | 0.404416 | 0.129216 |
| msdgs_p4p5_gate | test | inclusion | 0.744232 | 0.393705 |
| msdgs_p4p5_gate | test | patches | 0.896354 | 0.581224 |
| msdgs_p4p5_gate | test | pitted_surface | 0.775417 | 0.436878 |
| msdgs_p4p5_gate | test | rolled-in_scale | 0.567364 | 0.264542 |
| msdgs_p4p5_gate | test | scratches | 0.876175 | 0.489045 |
| msdgs_p5box32 | val | crazing | 0.422302 | 0.140943 |
| msdgs_p5box32 | val | inclusion | 0.823126 | 0.428113 |
| msdgs_p5box32 | val | patches | 0.898340 | 0.576986 |
| msdgs_p5box32 | val | pitted_surface | 0.734597 | 0.451185 |
| msdgs_p5box32 | val | rolled-in_scale | 0.610237 | 0.297823 |
| msdgs_p5box32 | val | scratches | 0.870101 | 0.487928 |
| msdgs_p5box32 | test | crazing | 0.418471 | 0.148430 |
| msdgs_p5box32 | test | inclusion | 0.799402 | 0.415612 |
| msdgs_p5box32 | test | patches | 0.934361 | 0.600625 |
| msdgs_p5box32 | test | pitted_surface | 0.770156 | 0.465425 |
| msdgs_p5box32 | test | rolled-in_scale | 0.584600 | 0.286597 |
| msdgs_p5box32 | test | scratches | 0.852618 | 0.440421 |
| msdgs_p5cafm_gate | val | crazing | 0.432272 | 0.154839 |
| msdgs_p5cafm_gate | val | inclusion | 0.806520 | 0.427108 |
| msdgs_p5cafm_gate | val | patches | 0.895636 | 0.590523 |
| msdgs_p5cafm_gate | val | pitted_surface | 0.733071 | 0.440555 |
| msdgs_p5cafm_gate | val | rolled-in_scale | 0.622690 | 0.288070 |
| msdgs_p5cafm_gate | val | scratches | 0.891971 | 0.535348 |
| msdgs_p5cafm_gate | test | crazing | 0.364967 | 0.121932 |
| msdgs_p5cafm_gate | test | inclusion | 0.739312 | 0.367308 |
| msdgs_p5cafm_gate | test | patches | 0.894250 | 0.584433 |
| msdgs_p5cafm_gate | test | pitted_surface | 0.755943 | 0.453458 |
| msdgs_p5cafm_gate | test | rolled-in_scale | 0.566849 | 0.263127 |
| msdgs_p5cafm_gate | test | scratches | 0.913399 | 0.500647 |

## Seed0 条件决策

| 候选 | mAP50-95 门槛 | mAP50 下限 | 目标类增益 | crazing 保护 | 总判定 | 理想值 |
|---|---|---|---|---|---|---|
| msdgs_p4p5_gate_seed0 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| msdgs_p5box32_seed0 | FAIL | PASS | PASS | FAIL | FAIL | FAIL |
| msdgs_p5cafm_gate_seed0 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

```json
{
  "seed0_gates": {
    "msdgs_p4p5_gate_seed0": {
      "pass": false,
      "ideal": false,
      "checks": {
        "map50_95_gt_0.401794": false,
        "map50_ge_0.722411": false,
        "target_class_gain_ge_0.01": false,
        "crazing_drop_le_0.02": false
      },
      "deltas_vs_msdgs": {
        "map50": -0.02175146448351517,
        "map50_95": -0.016402058097657157,
        "pitted_surface_map50_95": -0.031816875416595836,
        "rolled-in_scale_map50_95": 0.006112586960331234,
        "crazing_map50_95": -0.047853610902186317
      }
    },
    "msdgs_p5box32_seed0": {
      "pass": false,
      "ideal": false,
      "checks": {
        "map50_95_gt_0.401794": false,
        "map50_ge_0.722411": true,
        "target_class_gain_ge_0.01": true,
        "crazing_drop_le_0.02": false
      },
      "deltas_vs_msdgs": {
        "map50": -0.005809868278620844,
        "map50_95": -0.005985588047993429,
        "pitted_surface_map50_95": -0.0032707559773266226,
        "rolled-in_scale_map50_95": 0.028167183301132548,
        "crazing_map50_95": -0.02863923334950977
      }
    },
    "msdgs_p5cafm_gate_seed0": {
      "pass": false,
      "ideal": false,
      "checks": {
        "map50_95_gt_0.401794": false,
        "map50_ge_0.722411": false,
        "target_class_gain_ge_0.01": false,
        "crazing_drop_le_0.02": false
      },
      "deltas_vs_msdgs": {
        "map50": -0.026624743873636558,
        "map50_95": -0.01701962023053638,
        "pitted_surface_map50_95": -0.01523691296121632,
        "rolled-in_scale_map50_95": 0.00469745652325787,
        "crazing_map50_95": -0.05513791761914433
      }
    }
  },
  "combo": "skipped because neither A nor B exceeded the full seed0 gate",
  "p5_cafm": "trained because preceding candidates did not fully reach ideal/target-class condition",
  "multiseed_candidates": []
}
```

## 多 Seed 汇总

| 候选 | mean mAP50-95 | std | mean mAP50 | 配对平均增益 | 正增益 seeds | mean FPS | 最终门槛 |
|---|---:|---:|---:|---:|---:|---:|---|
| - | - | - | - | - | - | - | - |

## 最终排名

| 排名 | 候选 | mean mAP50-95 | mean mAP50 | 参数 | GFLOPs | FPS | 验收 |
|---:|---|---:|---:|---:|---:|---:|---|
| - | 尚未形成多 seed 排名 | - | - | - | - | - | - |
