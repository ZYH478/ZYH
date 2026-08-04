#!/usr/bin/env python
"""Build check for iter34 O2M-QFL (no training).
Verifies the double isolation: one2many.bce -> QualityFocal, one2one.bce -> pure BCE.
"""
import os
os.environ.setdefault("O2M_QFL_ENABLE", "1")
os.environ.setdefault("O2M_QFL_BETA", "2.0")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from pathlib import Path  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.loss import E2EDetectLoss  # noqa: E402

ROOT = Path("/root/autodl-tmp/neu-det-yolo26")
GSDOWN_YAML = ROOT / "generated_models_module_stage3_e250" / "y26n_s3_vovgscsp_gsdown_e250.yaml"
GEN = ROOT / "generated_models_o2mqfl_gsdown_e250"
GEN.mkdir(parents=True, exist_ok=True)


def swap_neck(doc, dilations=(1, 3, 5), fracs=(1, 1, 1)):
    n = 0
    for b in doc.get("head", []):
        if len(b) >= 4 and b[2] == "VoVGSCSP":
            c2 = b[3][0]
            b[2] = "MSDGS"
            b[3] = [c2, True, 1, 0.5, list(dilations), list(fracs)]
            n += 1
    return n


print("=== build MSDGS base yaml ===")
doc = yaml.safe_load(GSDOWN_YAML.read_text(encoding="utf-8"))
assert swap_neck(doc) == 4
cfg = GEN / "o2mqfl.yaml"
cfg.write_text("# build check\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

print("=== build model ===")
model = YOLO(str(cfg), task="detect")
m = model.model
print("head_class", type(m.model[-1]).__name__)
print("end2end", getattr(m.model[-1], "end2end", None), "reg_max", getattr(m.model[-1], "reg_max", None))

print("=== init criterion + verify isolation ===")
crit = m.init_criterion()
print("crit_class", type(crit).__name__)
o2m_bce = type(crit.one2many.bce).__name__
o2o_bce = type(crit.one2one.bce).__name__
print("one2many.bce", o2m_bce, "<-- expect _O2MQualityFocal")
print("one2one.bce ", o2o_bce, "<-- expect BCEWithLogitsLoss")
assert o2m_bce == "_O2MQualityFocal", f"o2m bce is {o2m_bce}, expected _O2MQualityFocal"
assert o2o_bce == "BCEWithLogitsLoss", f"o2o bce is {o2o_bce}, expected pure BCE"
print("beta", crit.one2many.bce.beta)

# forward smoke: QFL returns same shape as BCE(reduction=none)
pred = torch.randn(2, 100, 6)
tgt = torch.rand(2, 100, 6)
qfl_out = crit.one2many.bce(pred, tgt)
bce_out = nn.functional.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
print("qfl_shape", tuple(qfl_out.shape), "bce_shape", tuple(bce_out.shape))
assert qfl_out.shape == bce_out.shape
# QFL loss should be <= BCE elementwise where scale<1 (soft-label focal downweights easy)
print("qfl<=bce_frac", float((qfl_out <= bce_out + 1e-6).float().mean()))

params = int(sum(p.numel() for p in m.parameters()))
print("params_unfused", params)
print("BUILD_O2MQFL_OK")
