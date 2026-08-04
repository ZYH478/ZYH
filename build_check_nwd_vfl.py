#!/usr/bin/env python
"""build + patch 验证（不训练）：确认 gsdown 可实例化、NWD/VFL 补丁真正生效、
一次 forward+loss+backward 不报错、端到端输出格式正确。

远程用法：
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate yolo26
    cd /root/autodl-tmp/neu-det-yolo26
    python install_yolo26_exp_modules.py && python install_gsconv_modules.py
    python -u build_check_nwd_vfl.py
"""
from __future__ import annotations

import torch

import train_yolo26_module_sweep as sweep
import train_yolo26_stage3_gsconv as stage3
import nwd_vfl_patch as nv


def gsdown_doc():
    return sweep.base_doc(sweep.BASE_BACKBONE, stage3.make_gsconv_head(dysample=False, gsconv_down=True))


def build_model():
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    import yaml
    from pathlib import Path
    p = Path("/tmp/_gsdown_check.yaml")
    p.write_text(yaml.safe_dump(gsdown_doc(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    m = YOLO(str(p))
    # 训练时 model.args 是 IterableSimpleNamespace（有 .box/.cls/.dfl）；此处手动注入以测 loss。
    m.model.args = get_cfg(DEFAULT_CFG)
    return m


def check_nwd_similarity():
    """数值自检：完全重合的框 NWD≈1，远离的框 NWD→0，尺度合理。"""
    a = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    same = nv._nwd_similarity(a, a.clone(), const=0.1, norm=640.0)
    far = nv._nwd_similarity(a, torch.tensor([[400.0, 400.0, 500.0, 500.0]]), const=0.1, norm=640.0)
    tiny_shift = nv._nwd_similarity(a, torch.tensor([[105.0, 105.0, 205.0, 205.0]]), const=0.1, norm=640.0)
    print(f"NWD self={float(same):.4f} (expect ~1)  far={float(far):.4f} (expect small)  "
          f"tiny_shift={float(tiny_shift):.4f} (expect high)")
    assert float(same) > 0.99, "self-NWD should be ~1"
    assert float(far) < float(tiny_shift), "far should score lower than tiny shift"


def run_one_loss(model, tag):
    """跑一次 forward + loss + backward，返回 loss 标量确认可训练。"""
    model.model.train()
    imgs = torch.rand(2, 3, 640, 640)
    batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0, 0, 1]),
        "cls": torch.tensor([[0.0], [3.0], [5.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.15], [0.6, 0.6, 0.25, 0.1]]),
    }
    if torch.cuda.is_available():
        model.model = model.model.cuda()
        imgs = imgs.cuda()
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["img"] = imgs
    loss, items = model.model.loss(batch)
    loss.sum().backward()
    print(f"{tag}: loss={float(loss.sum()):.4f} items={[round(float(x),4) for x in items]} backward_ok")
    return float(loss.sum())


def main():
    check_nwd_similarity()

    m = build_model()
    det = m.model.model[-1]
    print(f"BUILD_OK params={sum(p.numel() for p in m.model.parameters())} "
          f"end2end={getattr(det,'end2end',None)} reg_max={getattr(det,'reg_max',None)} nl={det.nl}")

    # end2end 前向输出格式
    m.model.eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 640, 640)
        if torch.cuda.is_available():
            m.model = m.model.cuda(); x = x.cuda()
        out = m.model(x)
    y = out[0] if isinstance(out, (list, tuple)) else out
    print(f"FORWARD_OUT type={type(y).__name__} shape={tuple(y.shape) if torch.is_tensor(y) else 'n/a'}")

    # 官方对照 loss
    nv.reset_all()
    l0 = run_one_loss(build_model(), "ctrl(CIoU+BCE)")

    # NWD
    nv.reset_all(); nv.patch_nwd(ratio=0.5, const=0.1)
    l1 = run_one_loss(build_model(), "nwd")

    # VFL
    nv.reset_all(); nv.patch_vfl(gamma=2.0, alpha=0.75)
    l2 = run_one_loss(build_model(), "vfl")

    # NWD+VFL
    nv.reset_all(); nv.patch_nwd(ratio=0.5, const=0.1); nv.patch_vfl(gamma=2.0, alpha=0.75)
    l3 = run_one_loss(build_model(), "nwd+vfl")

    # reset 后应恢复官方（与 l0 同数量级）
    nv.reset_all()
    l4 = run_one_loss(build_model(), "after_reset(should≈ctrl)")

    print(f"ALL_BUILD_CHECK_OK  losses ctrl={l0:.3f} nwd={l1:.3f} vfl={l2:.3f} nwd_vfl={l3:.3f} reset={l4:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
