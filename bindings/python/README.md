# aria-engine (Python)

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
pip install -e .
```

```python
from aria_engine import Engine
with Engine("/path/to/bundle") as eng:
    print(eng.complete([{"role":"user","content":"hi"}], {"max_tokens": 16}))
```

Publish to PyPI via `release.yml` (`twine` / wheel).
