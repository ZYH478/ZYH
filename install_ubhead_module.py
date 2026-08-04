#!/usr/bin/env python
"""iter31：把不确定性感知框头 UBHead + UBE2ELoss 装进当前 ultralytics 8.4.93。

动机（承接 iter30 诊断收口）：
- crazing 是本 goal 万年最弱类，其 map50-95/map50 比值 0.347 全场最低 → 「能找到、框不准」。
- YOLOv26n 为提速把 reg_max=1（DFL→Identity），恰好砍掉了建模边界模糊的分布/不确定性部件。
- 4 个特征端赛道（backbone×3 + neck）都撬不动 crazing 定位精度，因为问题不在提特征，
  在「提出特征后单点回归表达不了模糊边界」。UBHead 直击这个 v26 官方默认掩盖的结构空白。

设计（极简单层 conv σ 头，最保守，先验证机理）：
- UBHead 继承官方 Detect：训练时每个检测层额外挂一路极简单层 conv（sigma_cv[i]，
  输入该层特征、输出 4 通道 = 每条边 l/t/r/b 的 log_sigma），forward 训练期在
  one2many / one2one 两个 dict 里各加 "log_sigma" key（两条都挂，避免 o2m/o2o 梯度
  不一致——VFL 崩盘教训）。
- 推理 / fuse 时丢弃 sigma_cv（连同 one2many 主分支），推理零额外成本，
  推理 fused 参数 / FLOPs / 输出形状 [1,300,6] 与 MSDGS 基线完全一致，end2end 契约不破。
- UBE2ELoss 继承官方 E2ELoss：把 reg_max=1 的 L1 回归项升级为 σ 加权的对数似然
  （RLE 残差形式：loss = |error|/sigma + log(sigma)，边界清晰边学小 σ 正常约束、
  边界模糊边学大 σ 自动降权），CIoU 项保留不动。total = o2m*L_o2m + o2o*L_o2o。

接线点（远程 8.4.93 源码探明，与 AuxDetect 同款范式）：
- Detect ∈ head；E2ELoss / v8DetectionLoss / BboxLoss ∈ utils.loss。
- tasks.py parse_model frozenset：end2end 检测头在此拿到 args.extend([reg_max, end2end, ch])
  ——UBHead 必须进这个集合；`m.legacy = legacy` 集合 UBHead 也要进。
- DetectionModel.init_criterion 已被 auxdetect patch 成 _aux_init_criterion；本脚本再包一层
  幂等分发：头是 UBHead 且 end2end → UBE2ELoss，否则回落原有逻辑（不破坏 aux/官方）。
- 其余 isinstance(m, Detect) 判定（stride/bias_init/end2end）UBHead 作为子类自动命中。

安全：改 tasks.py 前备份为 tasks.py.ubhead_bak（存在则不覆盖）。幂等。
用法（远程）：python install_ubhead_module.py  成功打印 INSTALL_UBHEAD_OK。
"""
from __future__ import annotations

import shutil
from pathlib import Path


MODULE_SRC = '''# Auto-generated UBHead (uncertainty-aware bbox head) + UBE2ELoss for YOLO26 (NEU-DET iter31).
# Injected by install_ubhead_module.py. Do not edit by hand.
import copy

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss, BboxLoss

__all__ = ["UBHead", "UBE2ELoss"]


class UBHead(Detect):
    """Detect + 不确定性感知框头（uncertainty-aware bbox head）。

    训练时每个检测层额外挂一路极简单层 conv（sigma_cv[i]: Conv2d(ch_i, 4, 1)），
    输出每条边 l/t/r/b 的 log_sigma。forward 训练期在 one2many / one2one 两个 dict 里
    各加 "log_sigma" key。推理 / fuse 时丢弃 sigma_cv，零额外成本、输出形状不变。

    log_sigma 分支使用未 detach 的特征（σ 的梯度参与训练回传，与主回归损失同源）。
    """

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        # 极简单层 conv：每层特征 -> 4 通道 log_sigma（每条边一个不确定性标量）。
        self.sigma_cv = nn.ModuleList(nn.Conv2d(x, 4, 1) for x in ch)
        # bias 初始化为 0：初始 sigma=exp(0)=1，等价于普通 L1，训练稳定起点。
        for conv in self.sigma_cv:
            nn.init.constant_(conv.weight, 0.0)
            nn.init.constant_(conv.bias, 0.0)

    def forward_sigma(self, x):
        """产出拼接后的 log_sigma，形状 (bs, 4, sum(anchors))，与 forward_head 的 boxes 对齐。"""
        bs = x[0].shape[0]
        return torch.cat([self.sigma_cv[i](x[i]).view(bs, 4, -1) for i in range(self.nl)], dim=-1)

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            if self.training and getattr(self, "sigma_cv", None) is not None:
                # 两条损失都挂 log_sigma：one2many 用原特征，one2one 用 detach 特征（与其 boxes 同源）。
                preds["log_sigma"] = self.forward_sigma(x)
                one2one["log_sigma"] = self.forward_sigma(x_detach)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def fuse(self):
        # 推理优化：删掉 one2many 主分支与 sigma 分支，只留 one2one。
        self.cv2 = self.cv3 = None
        self.sigma_cv = None


class _UBBboxLoss(BboxLoss):
    """BboxLoss（reg_max=1 分支）+ σ 加权对数似然回归。

    reg_max=1 时官方回归 = CIoU + L1(归一化 ltrb)。本类保留 CIoU 不动，
    把 L1 项换成 RLE 残差形式的对数似然：per-edge loss = |error|/sigma + log(sigma)。
    sigma = exp(log_sigma)，log_sigma 由 UBHead 的 sigma_cv 产出、按 fg_mask 取前景。
    边界清晰边（小 error）学小 sigma 得强约束；边界模糊边（大 error 且无法对齐）
    学大 sigma 自动降权，不再被矩形 GT 硬拽偏。
    """

    def __init__(self, reg_max=1):
        super().__init__(reg_max)
        self.log_sigma_fg = None  # 由 loss 侧在调用前注入（前景对齐后的 log_sigma，形状 (n_fg, 4)）

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # CIoU 项：与官方完全一致，不动。
        from ultralytics.utils.metrics import bbox_iou
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 回归项：reg_max=1 -> 走归一化 ltrb 的对数似然（替换官方 L1）。
        from ultralytics.utils.tal import bbox2dist
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        error = torch.abs(pred_dist[fg_mask] - target_ltrb[fg_mask])  # (n_fg, 4)

        if self.log_sigma_fg is not None:
            log_sigma = self.log_sigma_fg  # (n_fg, 4)
            # 数值稳定：clamp log_sigma 到合理区间，避免 sigma 爆炸/塌缩。
            log_sigma = torch.clamp(log_sigma, min=-3.0, max=3.0)
            sigma = torch.exp(log_sigma)
            # RLE 残差对数似然：|error|/sigma + log(sigma)，逐边后按边平均。
            nll = (error / sigma + log_sigma).mean(-1, keepdim=True)
            loss_reg = (nll * weight).sum() / target_scores_sum
        else:
            # 回退官方 L1（安全兜底）。
            loss_reg = (error.mean(-1, keepdim=True) * weight).sum() / target_scores_sum

        return loss_iou, loss_reg


class _UBDetLoss(v8DetectionLoss):
    """v8DetectionLoss，替换 bbox_loss 为 _UBBboxLoss，并在算 box loss 前对齐注入 log_sigma。"""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bbox_loss = _UBBboxLoss(self.reg_max).to(self.device)
        self._pending_log_sigma = None  # (bs, 4, total_anchors)，由 loss() 从 preds 取出暂存

    def loss(self, preds, batch):
        # 从 preds dict 取 log_sigma 暂存，供 get_assigned_targets_and_loss 内部对齐使用。
        self._pending_log_sigma = preds.get("log_sigma") if isinstance(preds, dict) else None
        return super().loss(preds, batch)

    def get_assigned_targets_and_loss(self, preds, batch):
        # 复刻官方流程，但在调用 bbox_loss 前，把 log_sigma 按 fg_mask 对齐后注入 bbox_loss。
        # 这里直接调用父类实现，靠 bbox_loss.log_sigma_fg 的注入点完成对齐——
        # 但父类内部先算 anchor/target 再调 bbox_loss，需要 fg_mask，故重写关键段。
        import torch as _torch
        from ultralytics.utils.tal import make_anchors

        loss = _torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = _torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = _torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            # 对齐 log_sigma：(bs,4,A) -> (bs,A,4)，按 fg_mask 取前景 -> (n_fg,4)。
            if self._pending_log_sigma is not None:
                ls = self._pending_log_sigma.permute(0, 2, 1).contiguous()  # (bs, A, 4)
                self.bbox_loss.log_sigma_fg = ls[fg_mask]
            else:
                self.bbox_loss.log_sigma_fg = None
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor,
            )
            self.bbox_loss.log_sigma_fg = None  # 用完清空，防跨 batch 泄漏

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()


class UBE2ELoss(E2ELoss):
    """E2ELoss，用 _UBDetLoss 作 one2many / one2one 两条损失（各自消费自己的 log_sigma）。"""

    def __init__(self, model, loss_fn=_UBDetLoss):
        super().__init__(model, loss_fn)
'''


PATCH_SRC = '''

# ===== UBHead loss wiring (injected by install_ubhead_module.py) =====
def _ub_init_criterion(self):
    """检测头是 UBHead 且 end2end 时用 UBE2ELoss；否则回落既有逻辑（aux / 官方）。"""
    from ultralytics.nn.modules.yolo26_ubhead import UBHead, UBE2ELoss

    m = self.model[-1]
    if isinstance(m, UBHead) and getattr(self, "end2end", False):
        return UBE2ELoss(self)
    return _ub_prev_init_criterion(self)


# 保存链上前一个 init_criterion（可能是 auxdetect 的 _aux_init_criterion 或官方），幂等分发。
if not hasattr(DetectionModel, "_ub_prev_init_criterion"):
    _ub_prev_init_criterion = DetectionModel.init_criterion
    DetectionModel._ub_prev_init_criterion = staticmethod(_ub_prev_init_criterion)
    DetectionModel.init_criterion = _ub_init_criterion
# ===== end UBHead loss wiring =====
'''


def inject_init_export(text: str) -> tuple[str, bool]:
    if "yolo26_ubhead" in text:
        return text, False
    return text + (
        "\n# UBHead uncertainty-aware bbox head\n"
        "from .yolo26_ubhead import UBHead  # noqa: E402,F401\n"
    ), True


def inject_tasks_import(text: str) -> tuple[str, bool]:
    if "yolo26_ubhead" in text:
        return text, False
    line = "from ultralytics.nn.modules.yolo26_ubhead import UBHead  # noqa: E402,F401\n"
    loss_anchor = "from ultralytics.utils.loss import ("
    if loss_anchor in text:
        return text.replace(loss_anchor, line + loss_anchor, 1), True
    return text + "\n" + line, True


def inject_frozenset(text: str) -> tuple[str, bool]:
    """把 UBHead 加进 parse_model 的 end2end 检测头 frozenset。"""
    if "                UBHead,\n" in text:
        return text, False
    marker = "                Detect,\n"  # 16-space indent, 仅出现在多行 frozenset 内
    if marker not in text:
        return text, False
    return text.replace(marker, marker + "                UBHead,\n", 1), True


def inject_legacy_set(text: str) -> tuple[str, bool]:
    """把 UBHead 加进 `if m in {..., Detect, ...}` 的 legacy 集合。"""
    if "if m in {UBHead, " in text or "UBHead, Detect," in text:
        return text, False
    # auxdetect 可能已把开头改成 {AuxDetect, Detect,；兼容两种。
    for marker in ("if m in {AuxDetect, Detect,", "if m in {Detect,"):
        if marker in text:
            repl = marker.replace("{", "{UBHead, ", 1)
            return text.replace(marker, repl, 1), True
    return text, False


def inject_once(text: str, marker: str, addition: str) -> tuple[str, bool]:
    if marker in text:
        return text, False
    return text + addition, True


def main() -> int:
    import ultralytics

    pkg = Path(ultralytics.__file__).resolve().parent
    mod_file = pkg / "nn" / "modules" / "yolo26_ubhead.py"
    init_file = pkg / "nn" / "modules" / "__init__.py"
    tasks_file = pkg / "nn" / "tasks.py"

    print("ultralytics_version", ultralytics.__version__)
    print("pkg", pkg)

    mod_file.write_text(MODULE_SRC, encoding="utf-8")
    print("wrote", mod_file)

    it = init_file.read_text(encoding="utf-8")
    it, it_changed = inject_init_export(it)
    if it_changed:
        init_file.write_text(it, encoding="utf-8")
    print("init_export_changed", it_changed)

    backup = tasks_file.with_name(tasks_file.name + ".ubhead_bak")
    if not backup.exists():
        shutil.copy(tasks_file, backup)
        print("backup_created", backup)
    else:
        print("backup_exists", backup)

    tk = tasks_file.read_text(encoding="utf-8")
    tk, imp_changed = inject_tasks_import(tk)
    tk, frozen_changed = inject_frozenset(tk)
    tk, legacy_changed = inject_legacy_set(tk)
    tk, patch_changed = inject_once(tk, "_ub_init_criterion", PATCH_SRC)
    if imp_changed or frozen_changed or legacy_changed or patch_changed:
        tasks_file.write_text(tk, encoding="utf-8")
    print("tasks_import_changed", imp_changed)
    print("tasks_frozenset_changed", frozen_changed)
    print("tasks_legacy_set_changed", legacy_changed)
    print("tasks_init_criterion_patched", patch_changed)

    print("INSTALL_UBHEAD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
