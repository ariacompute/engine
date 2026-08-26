"""Aria Engine Python binding (ctypes over libaria_ffi)."""
from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import c_char_p, c_int, c_size_t, c_void_p, POINTER, c_ubyte
from typing import Any, Optional
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


DEFAULT_SDK = "v1.0"
_HUB_REQUIRED = ("config.json", "weight.bin")
_HUB_OPTIONAL = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def _preferred_public_hub(site: Optional[str]) -> str:
    """``.cn`` → ModelScope, otherwise Hugging Face (same as aria-engine download)."""
    if site and "ariacompute.cn" in site.lower():
        return "modelscope"
    return "huggingface"


def _is_dashboard_token(token: Optional[str]) -> bool:
    if not token:
        return False
    t = token.strip().lower()
    return t.startswith("sk-") or t.startswith("bfvk-")


def _hub_bearer(token: Optional[str]) -> Optional[str]:
    """Dashboard API keys must not be sent to Hugging Face / ModelScope."""
    if not token or not token.strip() or _is_dashboard_token(token):
        return None
    return token.strip()


def _hub_path_names(model: str) -> list[str]:
    names = [model]
    lower = model.lower()
    core = model
    for suf in ("_channel", "_group"):
        if lower.endswith(suf):
            core = model[: -len(suf)]
            lower = core.lower()
            break
    stems = [core]
    if lower.endswith("_q326"):
        stems.append(f"{core[:-5]}q3.26")
    elif lower.endswith("_q3.26"):
        stems.append(f"{core[:-6]}q326")
    for stem in stems:
        for share in ("", "_channel", "_group"):
            cand = f"{stem}{share}"
            if cand not in names:
                names.append(cand)
    return names


def _hub_file_urls(source: str, model: str, file: str, sdk: str = DEFAULT_SDK) -> list[str]:
    urls: list[str] = []
    for name in _hub_path_names(model):
        if source == "modelscope":
            for repo in (f"AriaCompute/{name}", "AriaCompute/model"):
                urls.append(
                    f"https://www.modelscope.cn/models/{repo}/resolve/master/{sdk}/{name}/{file}"
                )
                urls.append(
                    f"https://modelscope.cn/models/{repo}/resolve/master/{sdk}/{name}/{file}"
                )
        else:
            for repo in (f"ariacompute/{name}", "ariacompute/model"):
                urls.append(
                    f"https://huggingface.co/{repo}/resolve/main/{sdk}/{name}/{file}"
                )
    return urls


def _fetch_url_to_file(url: str, dest: str, token: Optional[str]) -> None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=600) as resp:  # nosec - hub URL from _hub_file_urls
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)


def _fetch_hub_file(
    source: str, model: str, file: str, dest: str, token: Optional[str], required: bool
) -> bool:
    import urllib.error

    last: Optional[BaseException] = None
    for url in _hub_file_urls(source, model, file):
        try:
            _fetch_url_to_file(url, dest, token)
            return True
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403):
                field = "modelscope_api_token" if source == "modelscope" else "hf_token"
                raise RuntimeError(
                    f"auth failed HTTP {e.code}; set {field} via aria-engine auth "
                    f"(do not pass a Dashboard sk-/bfvk- key as the hub token)"
                ) from e
            continue
        except OSError as e:
            last = e
            continue
    if required:
        raise RuntimeError(f"{source}: missing {file}") from last
    return False


def download_model(
    model: str, token: Optional[str] = None, site: Optional[str] = None
) -> str:
    """Download ``model`` from the regional public hub into
    ``~/.ariacompute/models/{model}`` and return that path.

    Matches ``aria-engine download``: ``.com`` → Hugging Face, ``.cn`` → ModelScope.
    Dashboard is not used. A Dashboard API key (``sk-`` / ``bfvk-``) is ignored for
    hub auth. If a valid bundle already exists at the cache path, the download is skipped.
    """
    import shutil

    _parse_bundle_name(model)
    site = site or DEFAULT_SITE
    source = _preferred_public_hub(site)
    hub_token = _hub_bearer(token)
    cache = os.path.join(_aria_home(), "models", model)
    if os.path.isdir(cache) and _is_valid_bundle(cache):
        return cache

    staging = os.path.join(_aria_home(), "models", f".{model}.partial")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)
    try:
        for file in _HUB_REQUIRED:
            _fetch_hub_file(
                source, model, file, os.path.join(staging, file), hub_token, required=True
            )
        for extra in _HUB_OPTIONAL:
            try:
                _fetch_hub_file(
                    source,
                    model,
                    extra,
                    os.path.join(staging, extra),
                    hub_token,
                    required=False,
                )
            except RuntimeError:
                continue
        if not _is_valid_bundle(staging):
            raise RuntimeError(
                f"{source} fetch completed but bundle invalid "
                "(need weight.bin + aria-quant-bundle config.json)"
            )
        if os.path.isdir(cache):
            shutil.rmtree(cache)
        shutil.move(staging, cache)
    except Exception:
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        raise
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
