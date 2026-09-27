"""Python SDK: ctypes over libaria-engine_ffi → System One."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any


def aria_home() -> Path:
    override = os.environ.get("ARIA_COMPUTE_HOME")
    if override:
        return Path(override)
    return Path.home() / ".ariacompute"


def _candidate_libs() -> list[Path]:
    env = os.environ.get("ARIA_FFI_LIB")
    if env:
        return [Path(env)]
    lib = aria_home() / "lib"
    names = [
        "libaria-engine_ffi.so",
        "libaria-engine_ffi.dylib",
        "aria-engine_ffi.dll",
        "libaria_ffi.so",
        "libaria_ffi.dylib",
        "aria_ffi.dll",
    ]
    pkg = Path(__file__).resolve().parent / "lib"
    out: list[Path] = []
    for n in names:
        out.append(lib / n)
        out.append(pkg / n)
    return out


def _load_lib() -> ctypes.CDLL:
    for path in _candidate_libs():
        if path.is_file():
            return ctypes.CDLL(str(path))
    raise FileNotFoundError(
        "libaria-engine_ffi not found; run `aria-engine upgrade` or set ARIA_FFI_LIB"
    )


class AriaEngine:
    def __init__(self, checkpoint: str, track: str = "encoder") -> None:
        self._lib = _load_lib()
        self._lib.aria_last_error.restype = ctypes.c_char_p
        self._lib.aria_model_init.restype = ctypes.c_void_p
        self._lib.aria_model_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.aria_model_destroy.argtypes = [ctypes.c_void_p]
        self._lib.aria_systemone.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.aria_systemone.restype = ctypes.c_int
        handle = self._lib.aria_model_init(checkpoint.encode(), track.encode())
        if not handle:
            err = self._lib.aria_last_error() or b"init failed"
            raise RuntimeError(err.decode())
        self._handle = handle

    def systemone(self, request: dict[str, Any]) -> dict[str, Any]:
        buf = ctypes.create_string_buffer(1 << 20)
        raw = json.dumps(request).encode()
        rc = self._lib.aria_systemone(self._handle, raw, buf, len(buf))
        if rc != 0:
            err = self._lib.aria_last_error() or f"rc={rc}".encode()
            raise RuntimeError(err.decode() if isinstance(err, bytes) else str(err))
        return json.loads(buf.value.decode())

    def destroy(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.aria_model_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass
