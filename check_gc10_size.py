#!/usr/bin/env python
"""统计 GC10-DET train 图片尺寸分布，判断是否非方形 + 是否需要 cache。"""
import glob, os
from collections import Counter
from PIL import Image

D = "/root/autodl-tmp/neu-det-yolo26/GC10-DET/train/images"
fs = sorted(glob.glob(os.path.join(D, "*")))
print("TOTAL", len(fs))
c = Counter()
nonsq = 0
for f in fs:
    w, h = Image.open(f).size
    c[(w, h)] += 1
    if w != h:
        nonsq += 1
print("NONSQUARE", nonsq, "/", len(fs))
for size, n in c.most_common(8):
    print("SIZE", size, "count", n)
