import inspect
from ultralytics.nn.modules.block import C3k2, C2f, C3k, Bottleneck

print("=== C2f full source ===")
print(inspect.getsource(C2f))
print("=== C3k full source ===")
print(inspect.getsource(C3k))
print("=== Bottleneck full source ===")
print(inspect.getsource(Bottleneck))
