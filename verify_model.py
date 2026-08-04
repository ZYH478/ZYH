"""验证 YOLO26n 模型可加载并打印结构信息。在远程 yolo26 环境执行。"""
from ultralytics import YOLO

m = YOLO("yolo26n.pt")
print("MODEL_OK")
m.info(detailed=False, verbose=True)
