#!/usr/bin/env python3
"""Diagnose ariaengine /v1/chat/completions (engine.log Hello garbage).

Encodes the same ChatML string the Rust session uses, POSTs the OpenAI chat
payload, and optionally diffs against model/scripts/diag_qwen3_chat.py JSON.

H200 example (from engine repo root):

  ./ariaengine serve qwen3-0.6b_q4 --bind 127.0.0.1:8080
  python scripts/diag_qwen3_chat.py \\
    --url http://127.0.0.1:8080 \\
    --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \\
    --peer-report ./out/model_diag_qwen3.json \\
    --report ./out/engine_diag_qwen3.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Must match inference/src/chat.rs qwen_chatml(empty_think=true).
def engine_qwen3_template(user: str) -> str:
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


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


def _encode_bundle(bundle: Path, text: str) -> list[int] | None:
    tok_json = bundle / "tokenizer.json"
    if not tok_json.is_file():
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return None
        tok = AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=True)
        return tok(text, add_special_tokens=False)["input_ids"]
    tok = Tokenizer.from_file(str(tok_json))
    return tok.encode(text, add_special_tokens=False).ids


def _bundle_model_meta(bundle: Path) -> dict:
    cfg_path = bundle / "config.json"
    if not cfg_path.is_file():
        return {}
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    m = cfg.get("model") or {}
    return {
        "rope_theta": m.get("rope_theta"),
        "hidden_size": m.get("hidden_size"),
        "num_layers": m.get("num_layers"),
        "num_attention_heads": m.get("num_attention_heads"),
        "num_kv_heads": m.get("num_kv_heads"),
        "head_dim": m.get("head_dim"),
        "hidden_act": m.get("hidden_act"),
        "tie_word_embeddings": m.get("tie_word_embeddings"),
        "vocab_size": m.get("vocab_size"),
        "format_version": cfg.get("format_version"),
        "hadamard_seed": cfg.get("hadamard_seed"),
    }


def _compare_peer(engine: dict, peer: dict) -> list[str]:
    hints = []
    e_ids = engine.get("prompt_ids_engine_template")
    p_ids = peer.get("prompt_ids_engine_template") or peer.get("prompt_ids_hf")
    if e_ids and p_ids and e_ids != p_ids:
        hints.append(
            f"PROMPT_IDS mismatch engine={e_ids} vs model={p_ids} (tokenizer encode differs)"
        )
    if peer.get("prompt_ids_hf") and e_ids and peer["prompt_ids_hf"] != e_ids:
        hints.append("TEMPLATE: engine ChatML ids != HF apply_chat_template ids")
    recon = ((peer.get("chat") or {}).get("reconstruct") or {}).get("text_skip_special")
    eng_txt = engine.get("content")
    if recon and eng_txt:
        if recon.strip()[:20] == (eng_txt or "").strip()[:20]:
            hints.append("ENGINE matches HF+reconstruct prefix → garbage is QUANT (or shared template)")
        else:
            hints.append(
                "ENGINE text != HF+reconstruct → ENGINE_GRAPH (weights inject into HF is the teacher)"
            )
    chat = peer.get("chat") or {}
    plen = chat.get("exact_prefix_len")
    if isinstance(plen, int) and plen >= 4:
        hints.append("model: fp32 vs reconstruct chat prefix >= 4; engine should be close if graph is correct")
    return hints


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--bundle", required=True, help="same bundle ariaengine serve loaded")
    p.add_argument("--user", default="Hello")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--peer-report", default=None, help="JSON from model/scripts/diag_qwen3_chat.py")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    bundle = Path(args.bundle).expanduser().resolve()
    tmpl = engine_qwen3_template(args.user)
    prompt_ids = _encode_bundle(bundle, tmpl)

    chat = _post_chat(args.url, args.user, args.max_tokens, args.timeout)
    raw = chat["raw"]
    choice = ((raw.get("choices") or [{}])[0].get("message") or {})
    usage = raw.get("usage") or {}
    content = choice.get("content")

    report = {
        "side": "engine",
        "status": "ok",
        "url": args.url,
        "bundle": str(bundle),
        "bundle_model": _bundle_model_meta(bundle),
        "user": args.user,
        "engine_chat_prompt": tmpl,
        "prompt_ids_engine_template": prompt_ids,
        "prompt_tokens_api": usage.get("prompt_tokens"),
        "completion_tokens_api": usage.get("completion_tokens"),
        "prompt_ids_len": None if prompt_ids is None else len(prompt_ids),
        "content": content,
        "finish_reason": ((raw.get("choices") or [{}])[0].get("finish_reason")),
        "latency_ms": chat["latency_ms"],
        "raw": raw,
        "hints": [],
    }
    if prompt_ids is not None and usage.get("prompt_tokens") is not None:
        if len(prompt_ids) != usage["prompt_tokens"]:
            report["hints"].append(
                f"API prompt_tokens={usage['prompt_tokens']} != local encode len={len(prompt_ids)}"
            )

    if args.peer_report:
        peer_path = Path(args.peer_report).expanduser()
        peer = json.loads(peer_path.read_text(encoding="utf-8"))
        report["peer"] = str(peer_path)
        report["hints"].extend(_compare_peer(report, peer))

    if not report["hints"]:
        report["hints"].append(
            "No peer report: run model/scripts/diag_qwen3_chat.py on the same bundle, then --peer-report"
        )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        outp = Path(args.report).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        print(f"wrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
