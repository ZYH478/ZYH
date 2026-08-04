import inspect
from pathlib import Path
import ultralytics
from ultralytics.nn.modules.block import C3k2, C2f, Bottleneck
print('ultralytics', ultralytics.__version__, Path(ultralytics.__file__).resolve())
print('C3k2_SIGNATURE', inspect.signature(C3k2.__init__))
print(inspect.getsource(C3k2))
print('C2f_SIGNATURE', inspect.signature(C2f.__init__))
root=Path('/root/autodl-tmp/neu-det-yolo26')
print('\nMSDGS_YAML')
print((root/'generated_models_msdgs_gsdown_e250/y26n_gsdown_msdgs_135eq_e250.yaml').read_text())
print('DATA_YAML')
print((root/'dataset/neu-det.yaml').read_text())
print('DATA_COUNTS')
for s in ('train','valid','test'):
    d=root/'dataset'/s/'images'
    print(s, sum(1 for p in d.iterdir() if p.is_file()))
