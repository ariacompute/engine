#!/usr/bin/env python3
"""Diagnose aria-engine /v1/chat/completions for Gemma-4 (engine.log Hello garbage).

Encodes the same gemma4_it string the Rust session uses, POSTs the OpenAI chat
payload, and optionally diffs against model/scripts/diag_gemma4_chat.py JSON.

engine.log (H200, gemma-4-e2b-it_q4, temperature 0, Hello): multilingual garbage
with prompt_tokens=10 (template is already aligned). Peer model diag that skips
huge embed/PLE inject is an HF-fp32 teacher — do not treat that as proof the
engine unpacked hub codebook tables correctly.

H200 example (from engine repo root):

  ./aria-engine serve gemma-4-e2b-it_q4 --bind 127.0.0.1:8080 --hybrid-execution device
  python scripts/diag_gemma4_chat.py \\
    --url http://127.0.0.1:8080 \\
    --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \\
    --peer-report ../model/out/model_diag_gemma4.json \\
    --report ./out/engine_diag_gemma4.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Must match inference/src/chat.rs gemma4_it.
def engine_gemma_it_template(user: str) -> str:
    return f"<bos><|turn>user\n{user}<turn|>\n<|turn>model\n"


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
        "layer_types": m.get("layer_types"),
        "sliding_window": m.get("sliding_window"),
        "num_kv_shared_layers": m.get("num_kv_shared_layers"),
        "use_double_wide_mlp": m.get("use_double_wide_mlp"),
        "global_head_dim": m.get("global_head_dim"),
        "partial_rotary_factor": m.get("partial_rotary_factor"),
        "format_version": cfg.get("format_version"),
        "hadamard_seed": cfg.get("hadamard_seed"),
        "family": (cfg.get("model") or {}).get("family") or cfg.get("family"),
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
        hints.append("TEMPLATE: engine gemma_it ids != HF apply_chat_template ids")
    inj = peer.get("inject") or {}
    skipped_huge = inj.get("n_skipped_huge") or 0
    inject_embeddings = inj.get("inject_embeddings")
    if isinstance(skipped_huge, int) and skipped_huge > 0 and not inject_embeddings:
        hints.append(
            "peer skipped vocab/PLE inject (HF fp32 teacher); engine unpacks hub 2D "
            "codebook embed_tokens / embed_tokens_per_layer — check load unrotate+gather+PLE, "
            "not a model re-quantize"
        )
    n_injected = peer.get("n_injected")
    if isinstance(n_injected, int) and n_injected == 0:
        hints.append("model INJECT failed; do not treat reconstruct as a teacher")
    recon = ((peer.get("chat") or {}).get("reconstruct") or {}).get("text_skip_special")
    eng_txt = engine.get("content")
    if recon and eng_txt:
        if recon.strip()[:20] == (eng_txt or "").strip()[:20]:
            hints.append(
                "ENGINE matches HF+reconstruct prefix → remaining gap is elsewhere"
            )
        elif skipped_huge and not inject_embeddings:
            hints.append(
                "ENGINE text != peer reconstruct while peer kept HF embed/PLE; "
                "prefer codebook unpack / PLE load over ENGINE_GRAPH"
            )
        else:
            hints.append(
                "ENGINE text != HF+reconstruct → ENGINE_GRAPH (weights inject into HF is the teacher)"
            )
    chat = peer.get("chat") or {}
    plen = chat.get("exact_prefix_len")
    if isinstance(plen, int) and plen >= 4:
        hints.append(
            "model: fp32 vs reconstruct chat prefix >= 4; engine should be close if graph is correct"
        )
    return hints


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--bundle", required=True, help="same bundle aria-engine serve loaded")
    p.add_argument("--user", default="Hello")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument(
        "--peer-report",
        default=None,
        help="JSON from model/scripts/diag_gemma4_chat.py",
    )
    p.add_argument("--report", default=None)
    args = p.parse_args()

    bundle = Path(args.bundle).expanduser().resolve()
    tmpl = engine_gemma_it_template(args.user)
    prompt_ids = _encode_bundle(bundle, tmpl)
    prompt_ids_no_bos = _encode_bundle(bundle, tmpl.removeprefix("<bos>"))

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
        "prompt_ids_engine_without_bos_prefix": prompt_ids_no_bos,
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
    if (
        args.user == "Hello"
        and usage.get("prompt_tokens") is not None
        and usage["prompt_tokens"] != 10
    ):
        report["hints"].append(
            f"API prompt_tokens={usage['prompt_tokens']} "
            "(HF Gemma-4 Hello template is 10 ids; old <start_of_turn> encode was 28)"
        )

    if args.peer_report:
        peer_path = Path(args.peer_report).expanduser()
        peer = json.loads(peer_path.read_text(encoding="utf-8"))
        report["peer"] = str(peer_path)
        report["hints"].extend(_compare_peer(report, peer))

    if not report["hints"]:
        report["hints"].append(
            "No peer report: run model/scripts/diag_gemma4_chat.py on the same bundle, then --peer-report"
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
