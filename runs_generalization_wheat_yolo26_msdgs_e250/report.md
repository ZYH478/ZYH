# YOLO26n 与 MSDGS Wheat 迁移性/鲁棒性对比

> 本报告是两个结构在各目标数据集上按同一协议重新训练后的迁移性与鲁棒性验证，不是 zero-shot 泛化。

## 统一口径

- 训练：250 epochs，imgsz=640，batch=32，seed=0，deterministic=True，cache=False，resume=False，patience=0（禁用 EarlyStopping），均从 yolo26n.pt 初始化。
- 真值：每组训练后由独立新 Python 进程从磁盘重载 best.pt，在 test split 评测。
- FPS：同一 RTX 4090，模型 fuse 后，batch=32，固定 warmup/iterations，仅计 forward，排除预处理与后处理。

## wheat

### 总体性能

| 模型 | test mAP50 | test mAP50-95 | FPS | fused params |
|---|---:|---:|---:|---:|
| yolo26n | 0.329283 | 0.180043 | 2163.4 | 2,377,176 |
| msdgs | 0.282712 | 0.154177 | 2405.0 | 1,778,488 |

### 各缺陷 test mAP50

| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |
|---|---:|---:|---:|
| CrownAndRootRot | 0.174160 | 0.275740 | +0.101580 |
| HealthyWheat | 0.190408 | 0.157042 | -0.033365 |
| LeafRust | 0.779167 | 0.703333 | -0.075833 |
| PowderyMildew | 0.462169 | 0.481062 | +0.018893 |
| WheatLooseSmut | 0.725117 | 0.517830 | -0.207287 |
| WheatAphids | 0.378375 | 0.216340 | -0.162035 |
| WheatCystNematode | 0.096456 | 0.071511 | -0.024945 |
| WheatRedSpider | 0.243266 | 0.138677 | -0.104589 |
| WheatScab | 0.631109 | 0.591776 | -0.039334 |
| WheatSharpEyespot | 0.003654 | 0.000000 | -0.003654 |
| WheatStalkRot | 0.145769 | 0.122775 | -0.022994 |
| WheatTake-all | 0.121751 | 0.116456 | -0.005294 |

### 各缺陷 test mAP50-95

| 缺陷类别 | YOLO26n | MSDGS | MSDGS-YOLO26n |
|---|---:|---:|---:|
| CrownAndRootRot | 0.058155 | 0.108951 | +0.050796 |
| HealthyWheat | 0.067127 | 0.072429 | +0.005302 |
| LeafRust | 0.601350 | 0.483770 | -0.117580 |
| PowderyMildew | 0.312827 | 0.275640 | -0.037187 |
| WheatLooseSmut | 0.366748 | 0.301709 | -0.065039 |
| WheatAphids | 0.243944 | 0.136372 | -0.107572 |
| WheatCystNematode | 0.047998 | 0.028873 | -0.019125 |
| WheatRedSpider | 0.089393 | 0.051378 | -0.038015 |
| WheatScab | 0.243974 | 0.239823 | -0.004151 |
| WheatSharpEyespot | 0.002192 | 0.000000 | -0.002192 |
| WheatStalkRot | 0.049834 | 0.068673 | +0.018840 |
| WheatTake-all | 0.076971 | 0.082506 | +0.005535 |

