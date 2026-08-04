#!/usr/bin/env python
"""Prepare the uploaded Wheat polygon dataset for fair YOLO26n/MSDGS detection retraining."""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import yaml

ROOT=Path(os.environ.get('YOLO26_EXP_ROOT','/root/autodl-tmp/neu-det-yolo26'))
ARCHIVE=Path(os.environ.get('WHEAT_ARCHIVE','/root/autodl-fs/zzy_wheat_yolo_599_200_100.zip'))
WORK=ROOT/'generalization_wheat_stage'
RAW=WORK/'raw'
SOURCE=RAW/'zzy_wheat_yolo_599_200_100'
PROJECT=ROOT/'runs_generalization_wheat_yolo26_msdgs_e250'
OUT=PROJECT/'datasets'
PROFILE=PROJECT/'dataset_profile.json'
NAMES=['CrownAndRootRot','HealthyWheat','LeafRust','PowderyMildew','WheatLooseSmut','WheatAphids','WheatCystNematode','WheatRedSpider','WheatScab','WheatSharpEyespot','WheatStalkRot','WheatTake-all']
SUFFIXES={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def digest(path,algo='sha256'):
    h=hashlib.new(algo)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

def extract():
    if (SOURCE/'data.yaml').is_file(): return
    if not ARCHIVE.is_file(): raise FileNotFoundError(ARCHIVE)
    RAW.mkdir(parents=True,exist_ok=True)
    with ZipFile(ARCHIVE) as z:
        rr=RAW.resolve()
        for m in z.infolist():
            target=(RAW/m.filename).resolve()
            if target!=rr and rr not in target.parents: raise RuntimeError(f'unsafe zip member: {m.filename}')
        z.extractall(RAW)
    if not (SOURCE/'data.yaml').is_file(): raise RuntimeError(f'unexpected archive root under {RAW}')

def image_files(split):
    return sorted(p.resolve() for p in (SOURCE/'images'/split).iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)

def label_path(img,split): return SOURCE/'labels'/split/(img.stem+'.txt')

def parse_polygon(path):
    counts=Counter(); field_counts=Counter()
    if not path.is_file(): raise FileNotFoundError(path)
    lines=[x.strip() for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    for lineno,line in enumerate(lines,1):
        fs=line.split(); field_counts[len(fs)]+=1
        if len(fs)<7 or len(fs)%2==0: raise ValueError(f'{path}:{lineno}: polygon fields={len(fs)}')
        vals=[float(x) for x in fs]; clsf=vals[0]; coords=vals[1:]
        cls=int(clsf)
        if not math.isfinite(clsf) or clsf!=cls or not 0<=cls<len(NAMES): raise ValueError(f'{path}:{lineno}: class={clsf}')
        if len(coords)%2 or not all(math.isfinite(v) and 0<=v<=1 for v in coords): raise ValueError(f'{path}:{lineno}: invalid polygon coordinates')
        counts[cls]+=1
    return counts,field_counts,len(lines)==0

def write_outputs():
    OUT.mkdir(parents=True,exist_ok=True)
    seen={}; summary={}; all_fields=Counter()
    expected={'train':599,'val':200,'test':100}
    for split in ('train','val','test'):
        imgs=image_files(split)
        if len(imgs)!=expected[split]: raise RuntimeError(f'{split}: expected {expected[split]} images, got {len(imgs)}')
        image_stems={p.stem for p in imgs}; labels=sorted((SOURCE/'labels'/split).glob('*.txt')); label_stems={p.stem for p in labels}
        if image_stems!=label_stems: raise RuntimeError(f'{split}: image/label mismatch missing={len(image_stems-label_stems)} orphan={len(label_stems-image_stems)}')
        inst=Counter(); empty=0; bad=0
        for img in imgs:
            try:
                with Image.open(img) as im: im.verify()
            except Exception as e: bad+=1; raise RuntimeError(f'bad image {img}: {e!r}')
            h=digest(img,'sha1')
            if h in seen and seen[h][0]!=split: raise RuntimeError(f'content overlap: {seen[h]} vs {(split,str(img))}')
            seen[h]=(split,str(img))
            c,fc,is_empty=parse_polygon(label_path(img,split)); inst.update(c); all_fields.update(fc); empty+=int(is_empty)
        list_path=OUT/f'{split}.txt'; list_path.write_text(''.join(f'{p.as_posix()}\n' for p in imgs),encoding='utf-8')
        summary[split]={'images':len(imgs),'labels':len(labels),'bad_images':bad,'empty_labels':empty,'instances':{NAMES[i]:inst[i] for i in range(len(NAMES))}}
    if not all(summary['test']['instances'][n]>0 for n in NAMES): raise RuntimeError('test does not contain all 12 classes')
    data={'path':str(SOURCE.resolve()),'train':str((OUT/'train.txt').resolve()),'val':str((OUT/'val.txt').resolve()),'test':str((OUT/'test.txt').resolve()),'nc':len(NAMES),'names':NAMES}
    yaml_path=OUT/'wheat.yaml'; yaml_path.write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return yaml_path,summary,all_fields

def loader_smoke(yaml_path):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    data=check_det_dataset(str(yaml_path),autodownload=False)
    ds=YOLODataset(img_path=data['train'],imgsz=640,batch_size=2,augment=False,hyp=get_cfg(),rect=False,cache=False,single_cls=False,stride=32,pad=0.0,prefix='wheat-gate: ',task='detect',classes=None,data=data,fraction=0.02)
    item=ds[0]
    if item['bboxes'].shape[-1]!=4 or 'segments' in item: raise RuntimeError(f'detection conversion failed: keys={item.keys()} bboxes={item["bboxes"].shape}')
    return {'status':'pass','sampled_images':len(ds),'item_bbox_shape':list(item['bboxes'].shape),'polygon_converted_to_xywh_detection_boxes':True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--skip-extract',action='store_true'); args=ap.parse_args()
    if not args.skip_extract: extract()
    if not SOURCE.is_dir(): raise FileNotFoundError(SOURCE)
    yaml_path,summary,fields=write_outputs(); loader=loader_smoke(yaml_path)
    raw_yaml=yaml.safe_load((SOURCE/'data.yaml').read_text(encoding='utf-8-sig'))
    profile={'gate_pass':True,'created_at':time.strftime('%Y-%m-%d %H:%M:%S'),'archive':str(ARCHIVE),'archive_sha256':digest(ARCHIVE) if ARCHIVE.is_file() else None,'source_root':str(SOURCE),'original_data_yaml':raw_yaml,'prepared_yaml':str(yaml_path),'label_format':'YOLO segmentation polygons; detection loader converts polygons to xywh boxes without changing uploaded labels','field_count_min':min(fields),'field_count_max':max(fields),'field_count_distinct':len(fields),'path_overlap_count':0,'content_hash_overlap_count':0,'all_classes_in_test':True,'splits':summary,'loader_smoke':loader}
    atomic_json(PROFILE,profile)
    print(f'WHEAT_DATA_GATE_PASS profile={PROFILE} yaml={yaml_path} counts=599/200/100 classes=12 polygon_to_bbox=true')
    return 0
if __name__=='__main__': raise SystemExit(main())
