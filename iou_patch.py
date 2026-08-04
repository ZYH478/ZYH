# -*- coding: utf-8 -*-
"""Runtime monkey-patch for BboxLoss.forward to swap the IoU regression term.

Train scripts import this module and call patch_bbox_loss(iou_type) BEFORE
building the trainer. All comments are ASCII to avoid encoding corruption.

Supported iou_type:
  ciou          - identity, official BboxLoss untouched (correctness anchor)
  focaler_ciou  - Focaler-IoU linear remap on IoU main term, keep CIoU penalty
  wiou          - WIoU v3 dynamic non-monotonic focusing (running IoU mean)
"""
import os

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.loss import bbox2dist


def _iou_term(pb, tb, iou_type, state):
    """Return per-sample (1 - iou_variant), shape [n,1]."""
    if iou_type == "wiou":
        iou = bbox_iou(pb, tb, xywh=False, CIoU=False)
        x1p, y1p, x2p, y2p = pb.chunk(4, -1)
        x1g, y1g, x2g, y2g = tb.chunk(4, -1)
        cw = x2p.maximum(x2g) - x1p.minimum(x1g)
        ch = y2p.maximum(y2g) - y1p.minimum(y1g)
        rho2 = ((x1g + x2g - x1p - x2p) ** 2 + (y1g + y2g - y1p - y2p) ** 2) / 4
        r_wiou = torch.exp(rho2 / (cw ** 2 + ch ** 2 + 1e-7))
        base = r_wiou * (1.0 - iou)
        iou_mean = state.get("iou_mean")
        if iou_mean is None:
            iou_mean = iou.detach().mean().item()
        beta = (iou.detach() / (iou_mean + 1e-7)).clamp(0, 10)
        delta, alpha = 3.0, 1.9
        r = beta / (delta * torch.pow(torch.tensor(alpha), beta - delta) + 1e-7)
        with torch.no_grad():
            m = iou.mean().item()
            state["iou_mean"] = 0.9 * iou_mean + 0.1 * m
        return base * r

    iou = bbox_iou(pb, tb, xywh=False, CIoU=True)
    if iou_type == "focaler_ciou":
        d = float(os.environ.get("YOLO26_FOCALER_D", "0.0"))
        u = float(os.environ.get("YOLO26_FOCALER_U", "0.95"))
        iou_plain = bbox_iou(pb, tb, xywh=False, CIoU=False)
        penalty = iou_plain - iou
        iou_foc = ((iou_plain - d) / (u - d)).clamp(0, 1)
        return 1.0 - (iou_foc - penalty)
    return 1.0 - iou


def make_bbox_forward(iou_type):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if not hasattr(self, "_iou_state"):
            self._iou_state = {}
        iou_term = _iou_term(pred_bboxes[fg_mask], target_bboxes[fg_mask], iou_type, self._iou_state)
        loss_iou = (iou_term * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
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


def patch_bbox_loss(iou_type=None):
    """Patch BboxLoss.forward in place. Returns the active iou_type."""
    from ultralytics.utils.loss import BboxLoss
    it = iou_type or os.environ.get("YOLO26_IOU_TYPE", "ciou")
    if it and it != "ciou":
        BboxLoss.forward = make_bbox_forward(it)
    return it
