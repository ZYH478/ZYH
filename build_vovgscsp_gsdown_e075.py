from pathlib import Path
import re, yaml

ROOT = Path('/root/autodl-tmp/neu-det-yolo26')
GS = Path('/root/miniconda3/envs/yolo26/lib/python3.10/site-packages/ultralytics/nn/modules/yolo26_gsconv.py')
INSTALL = ROOT / 'install_gsconv_modules.py'

for path in [GS, INSTALL]:
    if path.exists():
        s = path.read_text(encoding='utf-8')
        old = 'c_ = int(c2 * 0.5)'
        new = 'c_ = int(c2 * e)'
        if old in s:
            bak = path.with_suffix(path.suffix + '.bak_eparam')
            if not bak.exists():
                bak.write_text(s, encoding='utf-8')
            s = s.replace(old, new)
            path.write_text(s, encoding='utf-8')
            print(f'PATCHED {path}')
        elif new in s:
            print(f'ALREADY_E_PARAM {path}')
        else:
            print(f'WARN pattern not found in {path}')

src = ROOT / 'generated_models_module_stage3_e250' / 'y26n_s3_vovgscsp_gsdown_e250.yaml'
out_dir = ROOT / 'generated_models_module_stage3_e250'
out = out_dir / 'y26n_s3_vovgscsp_gsdown_e075_e250.yaml'
data = yaml.safe_load(src.read_text(encoding='utf-8'))
for block in data.get('head', []):
    if len(block) >= 4 and block[2] == 'VoVGSCSP':
        c2 = block[3][0]
        # parse_model inserts repeat n at arg index 2 for repeat modules, so
        # [c2, shortcut, g, e] becomes __init__(c1,c2,n,shortcut,g,e).
        block[3] = [c2, True, 1, 0.75]
out.write_text('# generated for VoVGSCSP e=0.75 from y26n_s3_vovgscsp_gsdown_e250.yaml\n' + yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
print(f'WROTE {out}')
