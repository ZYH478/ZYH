from pathlib import Path
import json, inspect, os
print('PYTHON', os.sys.executable)
try:
    import torch
    print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception as e: print('TORCH_ERR', repr(e))
try:
    import ultralytics
    from ultralytics.nn.modules.head import Detect
    import ultralytics.nn.tasks as tasks
    print('ULTRALYTICS', ultralytics.__version__, Path(ultralytics.__file__).resolve())
    print('DETECT_INIT')
    print(inspect.getsource(Detect.__init__))
    print('DETECT_FORWARD')
    print(inspect.getsource(Detect.forward))
    print('DETECT_FORWARD_E2E')
    print(inspect.getsource(Detect.forward_end2end))
    print('DETECT_FUSE')
    print(inspect.getsource(Detect.fuse))
    tp=Path(tasks.__file__).resolve()
    txt=tp.read_text(encoding='utf-8')
    print('TASKS_FILE', tp)
    for marker in ['elif m in frozenset(', 'args.extend([reg_max, end2end', 'm.legacy = legacy']:
        i=txt.find(marker)
        print('TASKS_SNIP', marker, i)
        if i>=0: print(txt[max(0,i-500):i+1800])
except Exception as e:
    import traceback; traceback.print_exc()
root=Path('/root/autodl-tmp/neu-det-yolo26')
for rel in [
 'runs_cafm_msdgs_e250/status.json','runs_cafm_msdgs_e250/report.json',
 'generated_models_msdgs_gsdown_e250/y26n_gsdown_msdgs_135eq_e250.yaml',
 'dataset/neu-det.yaml']:
    p=root/rel
    print('FILE', p, p.exists(), p.stat().st_size if p.exists() else None)
    if p.exists():
        s=p.read_text(encoding='utf-8', errors='replace')
        print(s[:20000])
