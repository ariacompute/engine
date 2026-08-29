"""Aria Engine Python binding (ctypes over libaria_ffi)."""
from __future__ import annotations

import ctypes
import json
import os
import sys
from ctypes import c_char_p, c_int, c_size_t, c_void_p, POINTER, c_ubyte
from typing import Any, Optional
from urllib.request import Request, urlopen
import platform
import shutil
import tarfile

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


def _load_lib(path: Optional[str] = None, site: Optional[str] = None):
    """Resolve the FFI library: explicit path > ARIA_FFI_LIB env > bundled lib >
    ``~/.ariacompute/lib`` (same as ``aria-engine upgrade``) > download latest Release.
    """
    if not path:
        path = os.environ.get("ARIA_FFI_LIB") or _default_lib_path() or _cached_ffi_path()
    if not path:
        path = ensure_ffi_lib(site=site)
    return ctypes.CDLL(path)


def _aria_home() -> str:
    override = os.environ.get("ARIA_COMPUTE_HOME")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".ariacompute")


def _ffi_lib_name(plat: Optional[str] = None) -> str:
    p = plat or sys.platform
    return _LIB_NAMES.get(p, "libaria_ffi.so")


def _lib_dir() -> str:
    return os.path.join(_aria_home(), "lib")


def _cached_ffi_path() -> Optional[str]:
    candidate = os.path.join(_lib_dir(), _ffi_lib_name())
    return candidate if os.path.isfile(candidate) else None


def _ffi_asset_os(system: Optional[str] = None, machine: Optional[str] = None) -> str:
    """Match ``aria-engine upgrade`` asset suffixes."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    if system == "linux" and machine in ("x86_64", "amd64"):
        return "linux_x86_64"
    if system == "linux" and machine in ("aarch64", "arm64"):
        return "linux_arm64"
    if system in ("darwin", "macos"):
        return "macos"
    if system.startswith("win") and machine in ("x86_64", "amd64"):
        return "windows_x86_64"
    raise RuntimeError(f"unsupported platform {system}/{machine} for libaria_ffi")


def _strip_v(tag: str) -> str:
    t = tag.strip()
    if t[:1] in "vV":
        return t[1:]
    return t


def _parse_semver(tag: str) -> Optional[tuple[int, int, int]]:
    core = _strip_v(tag).split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    if not parts or not parts[0].isdigit():
        return None
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return (major, minor, patch)


def _select_latest_stable(releases: list) -> str:
    best_tag = None
    best_key = (-1, -1, -1)
    for rel in releases:
        if rel.get("draft") or rel.get("prerelease"):
            continue
        tag = str(rel.get("tag_name") or rel.get("tag") or "")
        parsed = _parse_semver(tag)
        if parsed and parsed > best_key:
            best_key = parsed
            best_tag = tag
    if not best_tag:
        raise RuntimeError("no stable release found for libaria_ffi")
    return _strip_v(best_tag)


def _upgrade_org(site: Optional[str] = None) -> str:
    cfg = _config_yml_scalar("upgrade_url")
    if cfg:
        return cfg.rstrip("/")
    hint = (site or _config_yml_scalar("site_url") or DEFAULT_SITE).lower()
    if "ariacompute.cn" in hint or "gitee.com" in hint:
        return "https://gitee.com/ariacompute"
    return "https://github.com/ariacompute"


def _releases_api_url(org: str) -> str:
    owner = org.rstrip("/").rsplit("/", 1)[-1] or "ariacompute"
    if "gitee.com" in org.lower():
        return f"https://gitee.com/api/v5/repos/{owner}/engine/releases?per_page=30"
    return f"https://api.github.com/repos/{owner}/engine/releases?per_page=30"


def _http_get_bytes(url: str, dest: Optional[str] = None) -> bytes:
    req = Request(url, headers={"User-Agent": f"aria-engine-sdk/{__version__}"})
    with urlopen(req, timeout=600) as resp:  # nosec - release/hub URL
        data = resp.read() if dest is None else None
        if dest is not None:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            return b""
        return data or b""


def _extract_ffi_archive(archive: str, dest_dir: str, lib_name: Optional[str] = None) -> str:
    want = lib_name or _ffi_lib_name()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if os.path.basename(member.name) != want:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, want)
            with open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            try:
                os.chmod(dest, 0o755)
            except OSError:
                pass
            return dest
    raise RuntimeError(f"{want} not found in {archive}")


def ensure_ffi_lib(site: Optional[str] = None) -> str:
    """Return a path to libaria_ffi, downloading the latest Release if needed."""
    env = os.environ.get("ARIA_FFI_LIB")
    if env and os.path.isfile(env):
        return env
    bundled = _default_lib_path()
    if bundled:
        return bundled
    cached = _cached_ffi_path()
    if cached:
        return cached

    org = _upgrade_org(site)
    raw = _http_get_bytes(_releases_api_url(org))
    try:
        releases = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise RuntimeError(f"invalid releases JSON from {org}") from e
    if not isinstance(releases, list):
        raise RuntimeError(f"unexpected releases payload from {org}")
    ver = _select_latest_stable(releases)
    asset_os = _ffi_asset_os()
    asset_name = f"libaria_ffi_{ver}_{asset_os}.tar.gz"
    url = None
    for rel in releases:
        tag = str(rel.get("tag_name") or rel.get("tag") or "")
        if _strip_v(tag) != ver:
            continue
        for asset in rel.get("assets") or []:
            if asset.get("name") == asset_name:
                url = asset.get("browser_download_url") or asset.get("direct_asset_url")
                break
        if url:
            break
    if not url:
        raise RuntimeError(f"release asset not found: {asset_name}")

    staging = os.path.join(_aria_home(), "tmp", f"ffi-{ver}")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)
    archive = os.path.join(staging, asset_name)
    try:
        _http_get_bytes(url, dest=archive)
        dest = _extract_ffi_archive(archive, _lib_dir(), _ffi_lib_name())
    finally:
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
    return dest


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


def _config_yml_scalar(key: str) -> Optional[str]:
    """Read a top-level scalar from ``~/.ariacompute/config.yml`` (aria-engine auth)."""
    path = os.path.join(_aria_home(), "config.yml")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line[:1].isspace():
                    continue
                s = line.strip()
                if not s or s.startswith("#") or ":" not in s:
                    continue
                k, _, v = s.partition(":")
                if k.strip() != key:
                    continue
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                return v or None
    except OSError:
        return None
    return None


def _hub_token_field(source: str) -> str:
    return "modelscope_api_token" if source == "modelscope" else "hf_token"


def _resolve_hub_token(
    source: str,
    token: Optional[str] = None,
    hf_token: Optional[str] = None,
    modelscope_api_token: Optional[str] = None,
) -> Optional[str]:
    """Named field for the active hub, then generic ``token``, then config.yml.

    Same keys as ``aria-engine auth``: ``hf_token`` (``.com``) /
    ``modelscope_api_token`` (``.cn``). Does not read ``HF_TOKEN`` /
    ``MODELSCOPE_API_TOKEN``. Dashboard ``sk-`` / ``bfvk-`` values are skipped.
    """
    named = modelscope_api_token if source == "modelscope" else hf_token
    for cand in (named, token, _config_yml_scalar(_hub_token_field(source))):
        bearer = _hub_bearer(cand)
        if bearer:
            return bearer
    return None


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
    model: str,
    token: Optional[str] = None,
    site: Optional[str] = None,
    hf_token: Optional[str] = None,
    modelscope_api_token: Optional[str] = None,
) -> str:
    """Download ``model`` from the regional public hub into
    ``~/.ariacompute/models/{model}`` and return that path.

    Matches ``aria-engine download``: ``.com`` → Hugging Face, ``.cn`` → ModelScope.
    Hub auth uses ``hf_token`` / ``modelscope_api_token`` (call args, else
    ``~/.ariacompute/config.yml`` from ``aria-engine auth``). Dashboard is not used.
    A Dashboard API key (``sk-`` / ``bfvk-``) is ignored for hub auth. If a valid
    bundle already exists at the cache path, the download is skipped.
    """
    import shutil

    _parse_bundle_name(model)
    site = site or DEFAULT_SITE
    source = _preferred_public_hub(site)
    hub_token = _resolve_hub_token(
        source,
        token=token,
        hf_token=hf_token,
        modelscope_api_token=modelscope_api_token,
    )
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


INTL_SITE = "https://ariacompute.com"
INTL_UPGRADE = "https://github.com/ariacompute"
CN_SITE = "https://ariacompute.cn"
CN_UPGRADE = "https://gitee.com/ariacompute"
_COMPUTES = ("auto", "cpu", "cuda")
_AUTH_KEYS = (
    "router",
    "site_url",
    "upgrade_url",
    "compute",
    "hf_token",
    "modelscope_api_token",
)


def default_auth_config() -> dict[str, Any]:
    return {
        "router": "",
        "site_url": "",
        "upgrade_url": "",
        "compute": "auto",
        "hf_token": "",
        "modelscope_api_token": "",
    }


def _gateway_region(url: str) -> Optional[str]:
    lower = (url or "").lower()
    if "ariacompute.cn" in lower or "gitee.com/ariacompute" in lower:
        return "cn"
    if "ariacompute.com" in lower or "github.com/ariacompute" in lower:
        return "intl"
    return None


def _pair_urls(region: str) -> tuple[str, str]:
    if region == "cn":
        return CN_SITE, CN_UPGRADE
    return INTL_SITE, INTL_UPGRADE


def fill_auth_urls(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill missing site_url / upgrade_url from a provided TLD."""
    out = dict(cfg)
    region = _gateway_region(out.get("site_url") or "") or _gateway_region(
        out.get("upgrade_url") or ""
    )
    if not region:
        return out
    site, upgrade = _pair_urls(region)
    if not out.get("site_url"):
        out["site_url"] = site
    if not out.get("upgrade_url"):
        out["upgrade_url"] = upgrade
    return out


def apply_auth(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into ``existing``. Validates; does not mutate ``existing``."""
    out = dict(existing)
    for key, val in updates.items():
        if val is None or key not in _AUTH_KEYS:
            continue
        out[key] = val
    compute = str(out["compute"])
    if compute not in _COMPUTES:
        raise ValueError(f"invalid compute: {compute}")
    for key in _AUTH_KEYS:
        if key == "compute":
            continue
        out[key] = "" if out.get(key) is None else str(out[key])
    return fill_auth_urls(out)



def _is_local_ref(ref: str) -> bool:
    return os.path.sep in ref or "\\" in ref or os.path.exists(ref)


class Engine:
    def __init__(
        self,
        model_ref: Optional[str] = None,
        lib=None,
        token: Optional[str] = None,
        site: Optional[str] = None,
        hf_token: Optional[str] = None,
        modelscope_api_token: Optional[str] = None,
    ):
        self._explicit_lib = lib
        self._lib = None
        self._handle = None
        self._cfg = default_auth_config()
        self._generic_token = token
        if site or hf_token or modelscope_api_token:
            self.auth(
                site_url=site,
                hf_token=hf_token,
                modelscope_api_token=modelscope_api_token,
            )
        if model_ref:
            self.open(model_ref)

    def auth(
        self,
        router: Optional[str] = None,
        site_url: Optional[str] = None,
        upgrade_url: Optional[str] = None,
        compute: Optional[str] = None,
        hf_token: Optional[str] = None,
        modelscope_api_token: Optional[str] = None,
    ) -> "Engine":
        """Set Config / Run fields on this instance only. Does not write config.yml."""
        self._cfg = apply_auth(
            self._cfg,
            {
                "router": router,
                "site_url": site_url,
                "upgrade_url": upgrade_url,
                "compute": compute,
                "hf_token": hf_token,
                "modelscope_api_token": modelscope_api_token,
            },
        )
        return self

    def auth_status(self) -> dict[str, Any]:
        return dict(self._cfg)

    def auth_clear(self) -> "Engine":
        """Reset instance defaults. Does not delete ~/.ariacompute/config.yml."""
        self._cfg = default_auth_config()
        return self

    def _ensure_lib(self) -> None:
        if self._lib is not None:
            return
        site = self._cfg.get("site_url") or None
        self._lib = self._explicit_lib or _load_lib(site=site)
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

    def open(self, model_ref: str) -> "Engine":
        if self._handle:
            self.close()
        self._ensure_lib()
        bundle_path = model_ref
        if not _is_local_ref(model_ref):
            bundle_path = download_model(
                model_ref,
                token=self._generic_token,
                site=self._cfg["site_url"] or DEFAULT_SITE,
                hf_token=self._cfg["hf_token"] or None,
                modelscope_api_token=self._cfg["modelscope_api_token"] or None,
            )
        self._handle = self._lib.aria_model_init(bundle_path.encode())
        if not self._handle:
            err = self._lib.aria_last_error()
            raise RuntimeError(err.decode() if err else "init failed")
        return self

    def close(self):
        if self._handle and self._lib:
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
        if not self._handle:
            raise RuntimeError("engine not opened")
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
        if not self._handle:
            raise RuntimeError("engine not opened")
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
        if not self._handle:
            raise RuntimeError("engine not opened")
        buf = ctypes.create_string_buffer(64 * 1024)
        arr = (c_ubyte * len(pcm)).from_buffer_copy(pcm)
        rc = self._lib.aria_transcribe(
            self._handle, arr, len(pcm), None, buf, len(buf)
        )
        if rc != 0:
            err = self._lib.aria_last_error()
            raise RuntimeError(err.decode() if err else "transcribe failed")
        return json.loads(buf.value.decode())
