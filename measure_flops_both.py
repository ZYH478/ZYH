from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
import copy
for n in ['msdgs_p4p5_gate','msdgs_p5box32','msdgs_p4p5_gate_p5box32','msdgs_p5cafm_gate']:
 p=f'/root/autodl-tmp/neu-det-yolo26/generated_models_msdgs_detect_e250/{n}.yaml'
 m=YOLO(p,task='detect').model.eval()
 a=get_flops(m,640)
 f=copy.deepcopy(m); f.fuse(); b=get_flops(f,640)
 print(n,'unfused_eval',a,'fused_infer',b)
