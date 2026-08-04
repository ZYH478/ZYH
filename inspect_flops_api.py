import inspect
import ultralytics.utils.torch_utils as t
for n in ['get_flops','get_flops_with_torch_profiler','model_info']:
 o=getattr(t,n,None); print(n,o); print(inspect.signature(o) if o else ''); print(inspect.getsource(o)[:4000] if o else '')
