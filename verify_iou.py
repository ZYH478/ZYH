import os, sys
sys.path.insert(0, "/root/autodl-tmp/neu-det-yolo26")
import torch
from ultralytics.utils.loss import BboxLoss

torch.manual_seed(0)
A = 100
anchor = torch.rand(A, 2) * 640
pred_dist = torch.rand(A, 4) * 20
tgt_ltrb  = torch.rand(A, 4) * 20
def ltrb2xyxy(ap, d):
    return torch.cat([ap - d[..., :2], ap + d[..., 2:]], -1)
pred_boxes = ltrb2xyxy(anchor, pred_dist)
tgt_boxes  = ltrb2xyxy(anchor, tgt_ltrb)
ts  = torch.rand(A, 6)
fg  = torch.rand(A) > 0.3
tss = ts.sum(-1)[fg].sum().clamp(min=1)
imgsz  = torch.tensor([640., 640.])
stride = torch.ones(A, 1)
args = (pred_dist, pred_boxes, anchor, tgt_boxes, ts, tss, fg, imgsz, stride)
def clone(a): return tuple(x.clone() if torch.is_tensor(x) else x for x in a)

# official baseline (reg_max=1 -> L1 branch)
bl = BboxLoss(reg_max=1)
off = bl.forward(*clone(args))
print("OFFICIAL      iou=%.8f dfl=%.8f" % (off[0].item(), off[1].item()))

import iou_patch
for it in ["ciou", "focaler_ciou", "wiou"]:
    if it == "focaler_ciou":
        os.environ["YOLO26_FOCALER_D"]="0.0"; os.environ["YOLO26_FOCALER_U"]="0.95"
    bl2 = BboxLoss(reg_max=1)
    bl2.forward = iou_patch.make_bbox_forward(it).__get__(bl2, BboxLoss)
    out = bl2.forward(*clone(args))
    nan = "NAN!" if (torch.isnan(out[0]) or torch.isnan(out[1])) else "ok"
    tag = ""
    if it == "ciou":
        tag = "MATCH" if abs(out[0].item()-off[0].item())<1e-6 and abs(out[1].item()-off[1].item())<1e-6 else "MISMATCH!!"
    pos = "pos" if out[0].item() > 0 else "NONPOS!"
    print("%-13s iou=%.8f dfl=%.8f  %s %s %s" % (it, out[0].item(), out[1].item(), nan, pos, tag))
