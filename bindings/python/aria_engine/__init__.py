"""Aria Engine Python binding (ctypes over libaria_ffi)."""
from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import c_char_p, c_int, c_size_t, c_void_p, POINTER, c_ubyte
from typing import Any, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

__version__ = "0.1.0"

_LIB_NAMES = {
    "win32": "aria_ffi.dll",
    "darwin": "libaria_ffi.dylib",
}

DEFAULT_SITE = "https://ariacompute.com"


def _default_lib_path(package_dir: Optional[str] = None) -> Optional[str]:
    """Locate the platform dynamic library bundled inside the wheel.

    Wheels ship the FFI under ``aria_engine/lib/`` (built by
    ``scripts/build-python-ffi.sh`` during cibuildwheel). Returns ``None``
    when the library is not present (e.g. a source checkout).
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__)) if package_dir is None else package_dir
    name = _LIB_NAMES.get(sys.platform, "libaria_ffi.so")
    candidate = os.path.join(pkg_dir, "lib", name)
    return candidate if os.path.isfile(candidate) else None


def _load_lib(path: Optional[str] = None):
    """Resolve the FFI library: explicit path > ARIA_FFI_LIB env > bundled lib."""
    if not path:
        path = os.environ.get("ARIA_FFI_LIB") or _default_lib_path()
    if not path:
        raise RuntimeError(
            "Cannot locate libaria_ffi. Install the aria-engine wheel (bundles the "
            "library) or set ARIA_FFI_LIB to libaria_ffi.so / .dylib / .dll"
        )
    return ctypes.CDLL(path)


def _aria_home() -> str:
    override = os.environ.get("ARIA_COMPUTE_HOME")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".ariacompute")


def _parse_bundle_name(model: str):
    """Return ``(slug, quant)`` from a model name like ``gemma-4-e2b-it_q4``."""
    if not model or "/" in model or "\\" in model:
        raise ValueError(f"invalid model name: {model}")
    idx = model.rfind("_q")
    if idx != -1:
        slug, suffix = model[:idx], model[idx + 2:]
        if suffix.endswith("_channel") or suffix.endswith("_group"):
            suffix = suffix.rsplit("_", 1)[0]
        quant = {
            "4": "int4",
            "8": "int8",
            "326": "int326",
            "3.26": "int326",
        }.get(suffix)
        if quant is None:
            raise ValueError(f"unknown quant suffix _q{suffix}")
    else:
        slug, quant = model, "int4"
    if not slug:
        raise ValueError(f"invalid model name: {model}")
    return slug, quant


def _is_valid_bundle(directory: str) -> bool:
    weight = os.path.join(directory, "weight.bin")
    config = os.path.join(directory, "config.json")
    if not os.path.isfile(weight) or not os.path.isfile(config):
        return False
    try:
        with open(config, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    return meta.get("format") == "aria-quant-bundle"


def download_model(model: str, token: str, site: Optional[str] = None) -> str:
    """Download ``model`` from the Dashboard private source into
    ``~/.ariacompute/models/{model}`` and return that path.

    If a valid bundle already exists at the cache path, the download is skipped.
    """
    if token is None:
        raise ValueError("api token is required to download a model")
    site = site or DEFAULT_SITE
    slug, quant = _parse_bundle_name(model)
    cache = os.path.join(_aria_home(), "models", model)
    if os.path.isdir(cache) and _is_valid_bundle(cache):
        return cache

    meta_url = (
        f"{site.rstrip('/')}/api/dashboard/models/{quote(slug)}/download"
        f"?quant={quote(quant)}&sdk=v1.0&format=json"
    )
    try:
        req = Request(meta_url, headers={"Authorization": f"Bearer {token}"})
        with urlopen(req) as resp:  # nosec - controlled URL from dashboard
            meta = json.loads(resp.read().decode("utf-8"))
    except OSError as e:
        raise RuntimeError(f"dashboard request failed: {e}") from e
    url = meta.get("url")
    if not url:
        raise RuntimeError("dashboard meta returned empty url")

    try:
        req = Request(url, headers={"Authorization": f"Bearer {token}"})
        with urlopen(req) as resp:  # nosec - dashboard-provided url
            data = resp.read()
    except OSError as e:
        raise RuntimeError(f"download stream failed: {e}") from e

    import io
    import shutil
    import zipfile

    if data[:2] != b"PK":
        raise RuntimeError("downloaded archive is not a valid zip")
    staging = os.path.join(_aria_home(), "models", f".{model}.partial")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(staging)
    # flatten a single top-level subdir (when config.json sits inside it)
    entries = [e for e in os.listdir(staging) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(staging, entries[0])):
        inner = os.path.join(staging, entries[0])
        if os.path.isfile(os.path.join(inner, "config.json")):
            for name in os.listdir(inner):
                shutil.move(os.path.join(inner, name), os.path.join(staging, name))
            os.rmdir(inner)
    if not _is_valid_bundle(staging):
        shutil.rmtree(staging)
        raise RuntimeError("downloaded archive did not contain a valid aria-quant-bundle")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
    shutil.move(staging, cache)
    return cache


class Engine:
    def __init__(
        self,
        model_ref: str,
        lib=None,
        token: Optional[str] = None,
        site: Optional[str] = None,
    ):
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

        bundle_path = model_ref
        if not (os.path.sep in model_ref or "\\" in model_ref or os.path.exists(model_ref)):
            if not token:
                raise ValueError(
                    f"model name '{model_ref}' requires an api token to download"
                )
            bundle_path = download_model(model_ref, token, site)
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
