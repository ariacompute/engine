#!/usr/bin/env python3
"""Profile aria-engine serve load + generate (H200 / local).

`hybrid_execution=device` only disables cloud handoff. Local GEMM is selected
with `--compute auto|cpu|cuda` (orthogonal). This script POSTs the same Hello
chat used by diag_qwen3_chat.py, then reads GET /v1/engine/profile.

Examples (from engine repo root):

  python scripts/profile_qwen3_serve.py --compute cpu --spawn
  python scripts/profile_qwen3_serve.py --compute cuda --spawn \\
      --report ./out/engine_profile_qwen3.json

If serve is already running with `--profile`:

  python scripts/profile_qwen3_serve.py --url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def _get(url: str, timeout_s: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_chat(url: str, user: str, max_tokens: int, timeout_s: float) -> dict:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {e.code}: {err}") from e
    ms = (time.perf_counter() - t0) * 1000
    return {"latency_ms": round(ms, 1), "raw": raw}


def _wait_up(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            _get(url.rstrip("/") + "/v1/models", timeout_s=5.0)
            return
        except Exception as e:  # noqa: BLE001 — probe loop
            last = e
            time.sleep(0.5)
    raise SystemExit(f"serve did not become ready at {url}: {last}")


def _find_engine() -> list[str]:
    env = os.environ.get("ARIA_ENGINE_BIN")
    if env:
        return [env]
    root = Path(__file__).resolve().parents[1]
    for cand in (
        root / "target" / "release" / "aria-engine",
        root / "target" / "debug" / "aria-engine",
    ):
        if cand.is_file():
            return [str(cand)]
    cargo = shutil.which("cargo")
    if cargo:
        return [
            cargo,
            "run",
            "-q",
            "-p",
            "aria-openai",
            "--bin",
            "aria-engine",
            "--",
        ]
    raise SystemExit("aria-engine binary not found; build it or set ARIA_ENGINE_BIN")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="qwen3-0.6b_q4")
    p.add_argument("--compute", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--bind", default="127.0.0.1:8080")
    p.add_argument("--user", default="Hello")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--spawn-timeout", type=float, default=600.0)
    p.add_argument(
        "--spawn",
        action="store_true",
        help="start aria-engine serve with --profile --hybrid-execution device",
    )
    p.add_argument("--report", default="./out/engine_profile_qwen3.json")
    args = p.parse_args()

    proc = None
    if args.spawn:
        cmd = _find_engine() + [
            "serve",
            args.model,
            "--bind",
            args.bind,
            "--hybrid-execution",
            "device",
            "--compute",
            args.compute,
            "--profile",
        ]
        print("spawn:", " ".join(cmd), file=sys.stderr)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        log_lines: list[bytes] = []

        def _drain() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                log_lines.append(line)

        threading.Thread(target=_drain, daemon=True).start()
        try:
            _wait_up(args.url, args.spawn_timeout)
        except SystemExit:
            proc.kill()
            sys.stderr.write(b"".join(log_lines[-80:]).decode("utf-8", errors="replace"))
            raise

    try:
        if not args.spawn:
            _wait_up(args.url, min(args.timeout, 30.0))
        chat = _post_chat(args.url, args.user, args.max_tokens, args.timeout)
        try:
            profile = _get(args.url.rstrip("/") + "/v1/engine/profile", timeout_s=30.0)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise SystemExit(f"profile HTTP {e.code}: {err}") from e
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    raw = chat["raw"]
    choice = ((raw.get("choices") or [{}])[0].get("message") or {})
    usage = raw.get("usage") or {}
    report = {
        "side": "engine_profile",
        "ci_fail": False,
        "url": args.url,
        "model": args.model,
        "compute_request": args.compute,
        "user": args.user,
        "content": choice.get("content"),
        "finish_reason": ((raw.get("choices") or [{}])[0].get("finish_reason")),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "http_latency_ms": chat["latency_ms"],
        "profile": profile,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    outp = Path(args.report).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text, encoding="utf-8")
    print(f"wrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
