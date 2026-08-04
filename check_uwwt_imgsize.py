import glob, os
from PIL import Image
fs = sorted(glob.glob('/root/autodl-tmp/neu-det-yolo26/UWWT-Dataset-1500/images/train/*'))
print("TOTAL", len(fs))
sizes = {}
for f in fs[:60]:
    w, h = Image.open(f).size
    sizes[(w, h)] = sizes.get((w, h), 0) + 1
for k, v in sorted(sizes.items(), key=lambda x: -x[1]):
    print("SIZE", k, "count", v)
# 采样几个具体文件
for f in fs[:3]:
    print("FILE", os.path.basename(f), Image.open(f).size)
