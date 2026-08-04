from pathlib import Path
import torch
from ultralytics import YOLO

ROOT=Path('/root/autodl-tmp/neu-det-yolo26')
WEIGHTS=ROOT/'yolo26n.pt'
BASE=ROOT/'generated_models_msdgs_gsdown_e250/y26n_gsdown_msdgs_135eq_e250.yaml'
B=ROOT/'generated_models_msdgs_detect_e250/msdgs_p5box32.yaml'
ckpt=torch.load(WEIGHTS, map_location='cpu', weights_only=False)
src=ckpt['model'].float().state_dict()

def inspect(name,cfg):
    tgt=YOLO(str(cfg), task='detect').model.state_dict()
    matched={k for k,v in src.items() if k in tgt and tuple(v.shape)==tuple(tgt[k].shape)}
    mismatch={k:(tuple(v.shape),tuple(tgt[k].shape)) for k,v in src.items() if k in tgt and tuple(v.shape)!=tuple(tgt[k].shape)}
    return tgt, matched, mismatch

bt,bm,bx=inspect('base',BASE)
ct,cm,cx=inspect('p5box32',B)
print('SOURCE_ITEMS',len(src))
print('BASE_TARGET_ITEMS',len(bt),'BASE_MATCHED',len(bm),'BASE_MISMATCH',len(bx))
print('P5BOX_TARGET_ITEMS',len(ct),'P5BOX_MATCHED',len(cm),'P5BOX_MISMATCH',len(cx))
print('MATCHED_LOST_VS_BASE',len(bm-cm))
for k in sorted(bm-cm):
    print('LOST',k,'src',tuple(src[k].shape),'base',tuple(bt[k].shape),'p5box',tuple(ct[k].shape) if k in ct else None)
print('P5BOX_SHAPE_MISMATCH_KEYS')
for k,(a,b) in sorted(cx.items()):
    if 'cv2.2' in k or 'one2one_cv2.2' in k:
        print('MISMATCH',k,a,b)
print('BASE_P5_KEYS')
for k in sorted(bt):
    if 'cv2.2' in k or 'one2one_cv2.2' in k:
        print('BASEKEY',k,tuple(bt[k].shape),'pretrained_match',k in bm)
