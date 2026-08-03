"""OpenAI-compatible chat clients for aria / llamacpp / ollama / vllm."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


BACKEND_IDS = ("aria", "llamacpp", "ollama", "vllm")


@dataclass
class BackendConfig:
    id: str
    base_url: str
    api_key: str = ""
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.id not in BACKEND_IDS:
            raise ValueError(f"unsupported backend id {self.id!r}; expected one of {BACKEND_IDS}")
        self.base_url = self.base_url.rstrip("/")


@dataclass
class ChatResult:
    status: str  # ok | error
    content: str = ""
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _headers(cfg: BackendConfig) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if cfg.api_key:
        h["Authorization"] = f"Bearer {cfg.api_key}"
    return h


def _post_json(cfg: BackendConfig, path: str, body: dict[str, Any]) -> tuple[int, bytes, float]:
    url = f"{cfg.base_url}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(cfg), method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            raw = resp.read()
            code = resp.getcode()
    except urllib.error.HTTPError as e:
        raw = e.read() if e.fp else b""
        code = e.code
    except Exception as e:
        raise RuntimeError(str(e)) from e
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return code, raw, elapsed_ms


def chat_completion(
    cfg: BackendConfig,
    *,
    model: str,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
    stream: bool = False,
) -> ChatResult:
    """Non-streaming chat by default; optional SSE for TTFT."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    path = "/v1/chat/completions"
    if stream:
        return _chat_stream(cfg, path, body)
    try:
        code, raw, latency_ms = _post_json(cfg, path, body)
    except RuntimeError as e:
        return ChatResult(status="error", error=str(e))
    if code < 200 or code >= 300:
        return ChatResult(
            status="error",
            latency_ms=latency_ms,
            error=f"HTTP {code}: {raw[:400]!r}",
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        return ChatResult(status="error", latency_ms=latency_ms, error=f"bad json: {e}")
    content = ""
    choices = payload.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content") or choices[0].get("text") or ""
        if content is None:
            content = ""
    usage = payload.get("usage") or {}
    return ChatResult(
        status="ok",
        content=str(content),
        latency_ms=latency_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        raw=payload,
    )


def _chat_stream(cfg: BackendConfig, path: str, body: dict[str, Any]) -> ChatResult:
    url = f"{cfg.base_url}{path}"
    data = json.dumps(body).encode("utf-8")
    headers = _headers(cfg)
    headers["Accept"] = "text/event-stream"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    chunks: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                s = line.decode("utf-8", errors="replace").strip()
                if not s.startswith("data:"):
                    continue
                payload = s[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    chunks.append(piece)
    except Exception as e:
        return ChatResult(status="error", error=str(e))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return ChatResult(
        status="ok",
        content="".join(chunks),
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
    )


def probe(cfg: BackendConfig) -> tuple[bool, str]:
    """Cheap readiness check: GET /v1/models (optional; failures are soft)."""
    url = f"{cfg.base_url}/v1/models"
    req = urllib.request.Request(url, headers=_headers(cfg), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=min(cfg.timeout_s, 10.0)) as resp:
            if 200 <= resp.getcode() < 300:
                return True, "ok"
            return False, f"HTTP {resp.getcode()}"
    except Exception as e:
        return False, str(e)


def parse_backend_arg(spec: str) -> BackendConfig:
    """Parse ``id=http://host:port``."""
    if "=" not in spec:
        raise ValueError(f"backend must be id=url, got {spec!r}")
    bid, url = spec.split("=", 1)
    bid = bid.strip()
    url = url.strip()
    if not bid or not url:
        raise ValueError(f"backend must be id=url, got {spec!r}")
    return BackendConfig(id=bid, base_url=url)
