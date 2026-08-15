"""Aria Engine Python binding (ctypes over libaria_ffi)."""
from __future__ import annotations

import ctypes
import json
import os
from ctypes import c_char_p, c_int, c_size_t, c_void_p, POINTER, c_ubyte
from typing import Any, Optional


def _load_lib():
    path = os.environ.get("ARIA_FFI_LIB")
    if not path:
        raise RuntimeError("Set ARIA_FFI_LIB to libaria_ffi.so / .dylib / .dll")
    return ctypes.CDLL(path)


class Engine:
    def __init__(self, bundle_path: str, lib=None):
        self._lib = lib or _load_lib()
        self._lib.aria_model_init.restype = c_void_p
        self._lib.aria_model_init.argtypes = [c_char_p]
        self._lib.aria_model_destroy.argtypes = [c_void_p]
        self._lib.aria_complete.restype = c_int
        self._lib.aria_complete.argtypes = [
            c_void_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_char_p,
            c_size_t,
        ]
        self._lib.aria_embed.restype = c_int
        self._lib.aria_embed.argtypes = [c_void_p, c_char_p, c_char_p, c_size_t]
        self._lib.aria_transcribe.restype = c_int
        self._lib.aria_transcribe.argtypes = [
            c_void_p,
            POINTER(c_ubyte),
            c_size_t,
            c_char_p,
            c_char_p,
            c_size_t,
        ]
        self._lib.aria_last_error.restype = c_char_p
        self._handle = self._lib.aria_model_init(bundle_path.encode())
        if not self._handle:
            err = self._lib.aria_last_error()
            raise RuntimeError(err.decode() if err else "init failed")

    def close(self):
        if self._handle:
            self._lib.aria_model_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def complete(
        self,
        messages: list[dict[str, str]],
        options: Optional[dict[str, Any]] = None,
        tools: Optional[list] = None,
    ) -> dict:
        buf = ctypes.create_string_buffer(256 * 1024)
        rc = self._lib.aria_complete(
            self._handle,
            json.dumps(messages).encode(),
            json.dumps(options or {"max_tokens": 16}).encode(),
            json.dumps(tools or []).encode(),
            buf,
            len(buf),
        )
        if rc != 0:
            err = self._lib.aria_last_error()
            raise RuntimeError(err.decode() if err else "complete failed")
        return json.loads(buf.value.decode())

    def embed(self, text: str) -> dict:
        buf = ctypes.create_string_buffer(256 * 1024)
        rc = self._lib.aria_embed(
            self._handle,
            json.dumps({"input": text}).encode(),
            buf,
            len(buf),
        )
        if rc != 0:
            err = self._lib.aria_last_error()
            raise RuntimeError(err.decode() if err else "embed failed")
        return json.loads(buf.value.decode())

    def transcribe(self, pcm: bytes) -> dict:
        buf = ctypes.create_string_buffer(64 * 1024)
        arr = (c_ubyte * len(pcm)).from_buffer_copy(pcm)
        rc = self._lib.aria_transcribe(
            self._handle, arr, len(pcm), None, buf, len(buf)
        )
        if rc != 0:
            err = self._lib.aria_last_error()
            raise RuntimeError(err.decode() if err else "transcribe failed")
        return json.loads(buf.value.decode())
