#!/usr/bin/env python
"""NWD 标签分配 + Varifocal 分类损失的运行时 monkey-patch（零参数精度改进）。

设计对齐 iou_patch.py：不写 loss.py，纯运行时替换类属性，每个候选训练前
先 reset_all() 恢复官方实现，再按需 patch，训练脚本单进程内循环即可干净隔离。

背景：vovgscsp_gsdown（1.936M/-18.5%、val mAP50-95 0.4077）是最激进纯轻量模型。
本模块在其结构不变（零新增参数/FLOPs）前提下，只改两处逻辑：

1. NWD 标签分配（patch TaskAlignedAssigner.iou_calculation）：
   overlaps = (1-ratio)*CIoU + ratio*NWD。把两框建模为 2D 高斯，NWD =
   exp(-sqrt(W2)/const)，W2 为 2 阶 Wasserstein 距离平方（坐标先除 norm 归一化）。
   直击小目标（crazing/scratches）正样本稀缺：IoU 对小框微小偏移过敏，NWD 平滑。
   官方 iou_calculation 就是给子类 override 的点（RotatedTAL 同款），注入安全。

2. Varifocal 分类损失（patch v8DetectionLoss.__init__ 的 self.bce）：
   weight = alpha*sigmoid(pred)^gamma*(1-label) + gt_score*label，label=(gt_score>0)。
   IoU-aware 软标签，聚焦高质量正样本、抑制大量简单负样本。E2E 模型 o2m/o2o 两个
   v8DetectionLoss 实例都经此路径，patch __init__ 一次全覆盖。

用法（训练脚本内）：
    import nwd_vfl_patch as nv
    nv.reset_all()                       # 每候选前恢复官方
    nv.patch_nwd(ratio=0.5, const=0.1)   # 需要时打 NWD
    nv.patch_vfl(gamma=2.0, alpha=0.75)  # 需要时打 VFL
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


_ORIG_IOU_CALC = None
_ORIG_V8_INIT = None


def _capture_originals() -> None:
    """首次调用时保存官方原版实现，供 reset 恢复。"""
    global _ORIG_IOU_CALC, _ORIG_V8_INIT
    if _ORIG_IOU_CALC is None:
        from ultralytics.utils.tal import TaskAlignedAssigner
        _ORIG_IOU_CALC = TaskAlignedAssigner.iou_calculation
    if _ORIG_V8_INIT is None:
        from ultralytics.utils.loss import v8DetectionLoss
        _ORIG_V8_INIT = v8DetectionLoss.__init__


def reset_all() -> None:
    """恢复 TaskAlignedAssigner.iou_calculation 与 v8DetectionLoss.__init__ 到官方原版。"""
    _capture_originals()
    from ultralytics.utils.tal import TaskAlignedAssigner
    from ultralytics.utils.loss import v8DetectionLoss
    TaskAlignedAssigner.iou_calculation = _ORIG_IOU_CALC
    v8DetectionLoss.__init__ = _ORIG_V8_INIT
    if hasattr(v8DetectionLoss, "_vfl_patched"):
        delattr(v8DetectionLoss, "_vfl_patched")


def _nwd_similarity(gt_xyxy, pd_xyxy, const, norm):
    """把 xyxy 框建模为 2D 高斯，返回归一化 Wasserstein 相似度 (N,)。

    坐标先除 norm(=imgsz) 归一化到 [0,1]，再算：
      W2 = ||mu_g - mu_p||^2 + ||Sigma_g^0.5 - Sigma_p^0.5||_F^2
         = dcx^2 + dcy^2 + ((wg-wp)/2)^2 + ((hg-hp)/2)^2
      NWD = exp(-sqrt(W2)/const)
    gt_xyxy / pd_xyxy 形状 (N,4)，assigner 内为 imgsz 像素尺度。
    """
    gx1, gy1, gx2, gy2 = (gt_xyxy / norm).chunk(4, -1)
    px1, py1, px2, py2 = (pd_xyxy / norm).chunk(4, -1)
    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    gw, gh = (gx2 - gx1).clamp(min=0), (gy2 - gy1).clamp(min=0)
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    pw, ph = (px2 - px1).clamp(min=0), (py2 - py1).clamp(min=0)
    center = (gcx - pcx) ** 2 + (gcy - pcy) ** 2
    wh = ((gw - pw) / 2) ** 2 + ((gh - ph) / 2) ** 2
    w2 = center + wh
    nwd = torch.exp(-torch.sqrt(w2 + 1e-7) / const)
    return nwd.squeeze(-1)


def patch_nwd(ratio: float = 0.5, const: float = 0.1, norm: float = 640.0) -> tuple:
    """把 iou_calculation 换成 overlaps = (1-ratio)*CIoU + ratio*NWD。返回 (ratio,const,norm)。"""
    _capture_originals()
    from ultralytics.utils.tal import TaskAlignedAssigner
    from ultralytics.utils.metrics import bbox_iou

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        ciou = bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
        nwd = _nwd_similarity(gt_bboxes, pd_bboxes, const, norm)
        return (1.0 - ratio) * ciou + ratio * nwd

    TaskAlignedAssigner.iou_calculation = iou_calculation
    return ratio, const, norm


class _VFLBce:
    """替换 self.bce(pred,target)->(bs,na,nc) 的逐元素 VFL；外部 .sum()/tss 后即标准 VFL。

    必须是模块级类（非闭包），否则 ultralytics 多进程 dataloader pickle loss 对象时
    报 Can't pickle local object。gamma/alpha 存为实例属性，可 pickle。
    """

    def __init__(self, gamma: float, alpha: float):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, pred_score, gt_score):
        label = (gt_score > 0).to(gt_score.dtype)
        weight = self.alpha * pred_score.detach().sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        bce = F.binary_cross_entropy_with_logits(pred_score, gt_score, reduction="none")
        return bce * weight


def _make_vfl_bce(gamma: float, alpha: float):
    """返回可 pickle 的逐元素 VFL callable。"""
    return _VFLBce(gamma, alpha)


def patch_vfl(gamma: float = 2.0, alpha: float = 0.75) -> tuple:
    """把 v8DetectionLoss.__init__ 的 self.bce 换成 VFL 包装器。返回 (gamma,alpha)。"""
    _capture_originals()
    from ultralytics.utils.loss import v8DetectionLoss
    orig_init = _ORIG_V8_INIT

    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.bce = _make_vfl_bce(gamma, alpha)

    v8DetectionLoss.__init__ = new_init
    v8DetectionLoss._vfl_patched = True
    return gamma, alpha


def patch_from_spec(spec: dict) -> dict:
    """按 spec 里的开关打补丁；训练脚本调用。spec 支持 nwd/vfl 布尔与参数。返回实际生效配置。"""
    active = {}
    if spec.get("nwd"):
        r, c, n = patch_nwd(
            ratio=float(spec.get("nwd_ratio", 0.5)),
            const=float(spec.get("nwd_const", 0.1)),
            norm=float(spec.get("nwd_norm", 640.0)),
        )
        active["nwd"] = {"ratio": r, "const": c, "norm": n}
    if spec.get("vfl"):
        g, a = patch_vfl(
            gamma=float(spec.get("vfl_gamma", 2.0)),
            alpha=float(spec.get("vfl_alpha", 0.75)),
        )
        active["vfl"] = {"gamma": g, "alpha": a}
    return active
