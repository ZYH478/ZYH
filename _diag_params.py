import sys
from ultralytics import YOLO
import train_yolo26_stage3_gsconv as s3
import train_yolo26_module_sweep as sweep
sweep.ensure_modules()
specs = s3.stage3_specs()
import yaml, tempfile, os
for name in ["y26n_s3_gsconv_neck_e250"]:
    doc = specs[name]["doc"]
    p = os.path.join(tempfile.gettempdir(), name+".yaml")
    with open(p,"w") as f: yaml.safe_dump(doc,f,sort_keys=False)
    m = YOLO(p)
    total = sum(x.numel() for x in m.model.parameters())
    print("MODEL", name, "params", total)
    for i, layer in enumerate(m.model.model):
        t = type(layer).__name__
        np_ = sum(x.numel() for x in layer.parameters())
        extra = ""
        if t == "VoVGSCSP":
            extra = "n_bottlenecks=%d" % len(layer.m)
        print(i, t, "params", np_, extra)
