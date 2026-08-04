#!/usr/bin/env python
"""诊断 end2end 口径矛盾：加载各 best.pt，报告 end2end 状态与 val 指标。

用法: python diag_end2end.py <weights.pt> <tag> [imgsz]
定位为何魔改模型训练内 val 与重载 val 不一致：检查重载后 Detect 头的
end2end 属性是否保留（NMS-free），以及不同 conf/iou 下的 val 指标。
"""
import sys
from ultralytics import YOLO

DATA = "/root/autodl-tmp/neu-det-yolo26/dataset/neu-det.yaml"


def main():
    weights = sys.argv[1]
    tag = sys.argv[2]
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640

    model = YOLO(weights)
    net = model.model
    # 探测 Detect 头的 end2end 状态
    head = net.model[-1]
    print(f"DIAG [{tag}]")
    print(f"  head type      : {type(head).__name__}")
    print(f"  head.end2end   : {getattr(head, 'end2end', 'MISSING')}")
    print(f"  net.end2end    : {getattr(net, 'end2end', 'MISSING')}")
    print(f"  args.end2end   : {net.args.get('end2end') if hasattr(net, 'args') and isinstance(net.args, dict) else getattr(getattr(net,'args',None),'end2end','NA')}")
    print(f"  reg_max/one2one: {getattr(head, 'reg_max', 'NA')} / {hasattr(head, 'one2one_cv2')}")

    # 默认 val
    m = model.val(data=DATA, imgsz=imgsz, batch=32, device=0, verbose=False,
                  project="/root/autodl-tmp/neu-det-yolo26/runs", name=f"diag_{tag}", exist_ok=True)
    print(f"  val default    : mAP50={m.box.map50:.5f} mAP50-95={m.box.map:.5f}")
    print(f"DIAG_END [{tag}]")


if __name__ == "__main__":
    sys.exit(main())
