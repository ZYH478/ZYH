#!/usr/bin/env python
"""关机重开后核查自定义模块是否仍在 ultralytics 包里（系统盘可能被重置）。"""
import os
import ultralytics

p = os.path.dirname(ultralytics.__file__)
print("ultralytics_version", ultralytics.__version__)
print("exp_mod_file", os.path.exists(p + "/nn/modules/yolo26_exp.py"))
print("gsconv_mod_file", os.path.exists(p + "/nn/modules/yolo26_gsconv.py"))
tasks = open(p + "/nn/tasks.py").read()
print("exp_import_in_tasks", "yolo26_exp" in tasks)
print("gsconv_import_in_tasks", "yolo26_gsconv" in tasks)
print("VoVGSCSP_in_tasks", "VoVGSCSP" in tasks)

# 真正 build 一下 gsconv_neck，确认模块可解析
try:
    import train_yolo26_stage3_gsconv as s3
    import train_yolo26_module_sweep as sweep
    from ultralytics import YOLO
    spec = s3.stage3_specs()["y26n_s3_gsconv_neck_e250"]
    s3.set_sweep_outputs()
    cfg = sweep.dump_yaml("y26n_s3_gsconv_neck_e250", spec["doc"])
    model = YOLO(str(cfg))
    m = model.model
    print("BUILD_OK", "params", int(sum(x.numel() for x in m.parameters())),
          "end2end", getattr(m, "end2end", None), "reg_max", getattr(m.model[-1], "reg_max", None))
except Exception as exc:
    print("BUILD_FAIL", repr(exc))
