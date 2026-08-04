from pathlib import Path
import inspect, re
from ultralytics.nn.modules.head import Detect
import ultralytics.nn.tasks as tasks
for name in ['forward_head','fuse','bias_init']:
    print('METHOD', name)
    obj=getattr(Detect,name,None)
    print(inspect.getsource(obj) if obj else 'MISSING')
tp=Path(tasks.__file__).resolve(); txt=tp.read_text(encoding='utf-8')
print('TASKS_FILE',tp)
patterns=[
 r'elif m in frozenset\(\s*\{\s*\n\s*Detect,',
 r'args\.extend\(\[reg_max, end2end, \[ch\[x\] for x in f\]\]\)',
 r'if m in \{[^\n]+\}:\s*\n\s*m\.legacy = legacy',
]
for pat in patterns:
    m=re.search(pat,txt,re.S)
    print('PATTERN',pat,'FOUND',bool(m),'POS',m.start() if m else None)
    if m: print(txt[max(0,m.start()-900):m.start()+2600])
print('IMPORT_TAIL')
for line in txt.splitlines():
    if any(x in line for x in ['BoxHead32Detect','UBHead','CAFM','AuxDetect']): print(line)
