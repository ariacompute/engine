# aria-engine (PyPI)

```python
from aria_engine import AriaEngine
eng = AriaEngine("/path/to/afm-de", "encoder")
print(eng.systemone({"state": {}, "questions": {}}))
```

Wheels bundle `libaria_ffi` via cibuildwheel (`scripts/build-python-ffi.sh`).
