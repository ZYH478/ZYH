#!/usr/bin/env python
"""给 ultralytics 8.4.93 注入可选 box 回归损失（Focaler-CIoU / WIoU v3）。

依据探查报告：唯一 IoU 注入点 = BboxLoss.forward（loss.py 硬编码 CIoU=True）。
本脚本 monkey-patch BboxLoss.forward，只替换 IoU 项，DFL/L1 分支逐字节照抄官方 8.4.93
（含 reg_max=1 的 stride/imgsz 归一化）。覆盖 o2m/o2o/aux 全部 E2E 路径（都经 BboxLoss.forward），
不碰 metrics.bbox_iou（验证指标不受污染）。

损失类型由环境变量 YOLO26_IOU_TYPE 选择：ciou(默认对照) / focaler_ciou / wiou。
- focaler_ciou: Focaler-IoU 线性重映射包裹 CIoU，聚焦难样本区间 [d,u]。
  忠实实现 = IoU_focaler - CIoU_penalty；d/u 由 YOLO26_FOCALER_D/U 控制（默认 0.0/0.95）。
- wiou: WIoU v3 动态非单调聚焦。R_wiou=exp(rho2/c2*)（c2* detach），
  β=L_IoU/L̄_IoU 离群度（L̄ 动量均值），r=β/(δ·α^(β-δ))，δ=3/α=1.9（论文默认）。
两者均尺度无关，在 decoded-box 坐标系下安全。

用法（远程）：先 python install_iou_losses.py（写模块+patch loss.py，幂等），
再 YOLO26_IOU_TYPE=focaler_ciou python -u train_xx.py。
"""
from __future__ import annotations

import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated optional box regression losses for YOLO26.
# Injected by install_iou_losses.py. Do not edit by hand.
import math
import os

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.loss import bbox2dist


def _iou_loss_term(pb, tb, weight, target_scores_sum, iou_type, state):
    """按 iou_type 计算 (1-IoU)*weight 归一后的 loss_iou 标量。pb/tb 形状 (N,4)。"""
    if iou_type == "focaler_ciou":
        d = float(os.environ.get("YOLO26_FOCALER_D", "0.0"))
        u = float(os.environ.get("YOLO26_FOCALER_U", "0.95"))
        ciou = bbox_iou(pb, tb, xywh=False, CIoU=True)   # (N,1) = IoU - penalty
        iou = bbox_iou(pb, tb, xywh=False)               # (N,1) plain IoU
        penalty = iou - ciou                             # rho2/c2 + v*alpha
        iou_foc = ((iou - d) / (u - d)).clamp(0, 1)      # 难样本区间线性重映射
        focaler_ciou = iou_foc - penalty
        return ((1.0 - focaler_ciou) * weight).sum() / target_scores_sum

    if iou_type == "wiou":
        iou = bbox_iou(pb, tb, xywh=False)               # (N,1) plain IoU
        x1p, y1p, x2p, y2p = pb.chunk(4, -1)
        x1g, y1g, x2g, y2g = tb.chunk(4, -1)
        cw = x2p.maximum(x2g) - x1p.minimum(x1g)
        ch = y2p.maximum(y2g) - y1p.minimum(y1g)
        rho2 = ((x1g + x2g - x1p - x2p) ** 2 + (y1g + y2g - y1p - y2p) ** 2) / 4  # (N,1) 中心距²
        c2 = (cw ** 2 + ch ** 2 + 1e-7).detach()         # 外接框对角线²，detach 防放大外接框
        r_wiou = torch.exp(rho2 / c2)                    # WIoU v1 距离聚焦（rho2 保梯度）
        base = r_wiou * (1.0 - iou)                      # L_WIoUv1
        # v3 非单调聚焦（全程 detach，只调梯度增益不参与反传）
        l_iou = (1.0 - iou).detach()
        prev = state.get("l_iou_mean", None)
        mean = prev if prev is not None else l_iou.mean().clamp(min=1e-7)
        beta = (l_iou / (mean + 1e-7)).clamp(0, 10)
        delta, alpha = 3.0, 1.9
        r = beta / (delta * torch.pow(alpha, beta - delta) + 1e-7)
        loss = (base * r.detach() * weight).sum() / target_scores_sum
        with torch.no_grad():
            m = l_iou.mean()
            state["l_iou_mean"] = (0.9 * mean + 0.1 * m) if prev is not None else m
        return loss

    # ciou（默认对照，与官方一致）
    iou = bbox_iou(pb, tb, xywh=False, CIoU=True)
    return ((1.0 - iou) * weight).sum() / target_scores_sum


def make_bbox_forward(iou_type):
    """生成替换版 BboxLoss.forward：只改 IoU 项，DFL/L1 分支照抄官方 8.4.93。"""
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if not hasattr(self, "_iou_state"):
            self._iou_state = {}
        loss_iou = _iou_loss_term(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], weight,
            target_scores_sum, iou_type, self._iou_state,
        )

        # ==== DFL loss（逐字节照抄官方 8.4.93 BboxLoss.forward）====
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl
    return forward


def install(iou_type=None):
    """按环境变量或参数 patch BboxLoss.forward。返回实际生效的 iou_type。"""
    from ultralytics.utils.loss import BboxLoss

    it = iou_type or os.environ.get("YOLO26_IOU_TYPE", "ciou")
    BboxLoss.forward = make_bbox_forward(it)
    return it
'''


PATCH_INIT = '''

# ===== IOU loss injection (install_iou_losses.py) =====
import os as _os
if _os.environ.get("YOLO26_IOU_TYPE", "ciou") != "ciou":
    try:
        from ultralytics.utils.yolo26_iou_losses import install as _install_iou
        _t = _install_iou()
        print("IOU_LOSS_INSTALLED", _t)
    except Exception as _e:
        print("IOU_LOSS_INSTALL_FAIL", repr(_e))
# ===== end IOU loss injection =====
'''


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "utils" / "yolo26_iou_losses.py"
    loss_file = pkg / "utils" / "loss.py"

    print("ultralytics_version", ultralytics.__version__)
    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    lt = loss_file.read_text(encoding="utf-8")
    if "yolo26_iou_losses" not in lt:
        backup = loss_file.with_name("loss.py.iou_bak")
        if not backup.exists():
            shutil.copy(loss_file, backup)
            print("backup_created", backup)
        lt = lt + PATCH_INIT
        loss_file.write_text(lt, encoding="utf-8")
        print("patched loss.py")
    else:
        print("loss.py already patched")

    print("INSTALL_IOU_LOSSES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
