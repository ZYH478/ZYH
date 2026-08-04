import re, pathlib, ultralytics
t = pathlib.Path(ultralytics.__file__).resolve().parent / "nn" / "tasks.py"
s = t.read_text(encoding="utf-8")
i = s.find("repeat_modules")
print("=== around repeat_modules ===")
print(s[i-50:i+400])
print("=== base_modules already has? ===")
for m in ("DCNv2Conv", "PKIC3k2", "DWRC3k2"):
    print(m, m in s)
