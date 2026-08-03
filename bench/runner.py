"""Orchestrate multi-backend × multi-family bench runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from .backends import BackendConfig, ChatResult, chat_completion, probe
from .families import BACKEND_IDS, Family, select_families
from .metrics import (
    aggregate_latencies,
    estimate_completion_tokens,
    quality_vs_ref,
    tokens_per_sec,
)

DEFAULT_PROMPTS = [
    "Hello, how are you?",
    "Summarize in one sentence: The sky is blue and water is wet.",
]

ChatFn = Callable[..., ChatResult]


def pick_ref_backend(
    configured: list[BackendConfig],
    explicit: str | None,
) -> str | None:
    ids = [b.id for b in configured]
    if explicit:
        if explicit not in ids:
            return None
        return explicit
    if "llamacpp" in ids:
        return "llamacpp"
    for bid in ids:
        if bid != "aria":
            return bid
    return ids[0] if ids else None


def resolve_model_id(
    family: Family,
    backend_id: str,
    overrides: dict[str, str],
) -> str:
    # keys: "family_path", "backend:family_path", "backend:*"
    for key in (f"{backend_id}:{family.path}", family.path, f"{backend_id}:*"):
        if key in overrides:
            return overrides[key]
    # aria often loads a single local bundle — allow family path as model id
    if backend_id == "aria":
        return family.path
    return family.base_model


def _run_one_prompt(
    cfg: BackendConfig,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    measure_ttft: bool,
    chat_fn: ChatFn,
) -> dict[str, Any]:
    r = chat_fn(
        cfg,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=False,
    )
    out: dict[str, Any] = {
        "prompt": prompt,
        "status": r.status,
        "content": r.content,
        "latency_ms": r.latency_ms,
        "error": r.error,
    }
    n_tok, tok_src = estimate_completion_tokens(r.content, r.completion_tokens)
    out["completion_tokens"] = n_tok
    out["completion_tokens_source"] = tok_src
    out["prompt_tokens"] = r.prompt_tokens
    out["tokens_per_sec"] = tokens_per_sec(n_tok, r.latency_ms)
    if measure_ttft and r.status == "ok":
        rs = chat_fn(
            cfg,
            model=model,
            prompt=prompt,
            max_tokens=min(max_tokens, 16),
            temperature=0.0,
            stream=True,
        )
        out["ttft_ms"] = rs.ttft_ms if rs.status == "ok" else None
        if rs.status != "ok":
            out["ttft_note"] = rs.error or "stream failed"
    else:
        out["ttft_ms"] = None
    return out


def run_family_backend(
    family: Family,
    cfg: BackendConfig,
    *,
    model_id: str,
    prompts: list[str],
    max_tokens: int,
    warmup: int,
    runs: int,
    measure_ttft: bool,
    chat_fn: ChatFn = chat_completion,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "family": family.path,
        "base_model": family.base_model,
        "kind": family.kind,
        "backend": cfg.id,
        "model_id": model_id,
    }
    # Warmup (discard)
    for p in prompts[:1]:
        for _ in range(max(0, warmup)):
            chat_fn(
                cfg,
                model=model_id,
                prompt=p,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
            )

    samples: list[dict[str, Any]] = []
    errors = 0
    for p in prompts:
        for _ in range(max(1, runs)):
            sample = _run_one_prompt(
                cfg,
                model_id,
                p,
                max_tokens=max_tokens,
                measure_ttft=measure_ttft,
                chat_fn=chat_fn,
            )
            samples.append(sample)
            if sample["status"] != "ok":
                errors += 1

    if not samples or errors == len(samples):
        reason = samples[0].get("error") if samples else "no samples"
        return {
            **base,
            "status": "error" if samples else "skipped",
            "reason": reason,
            "perf": {},
            "quality": {},
            "samples": samples,
        }

    ok_lat = [s["latency_ms"] for s in samples if s["status"] == "ok"]
    ok_tps = [
        s["tokens_per_sec"]
        for s in samples
        if s["status"] == "ok" and s.get("tokens_per_sec") is not None
    ]
    ttfts = [s["ttft_ms"] for s in samples if s.get("ttft_ms") is not None]
    perf = {
        **aggregate_latencies(ok_lat),
        "tokens_per_sec_mean": (sum(ok_tps) / len(ok_tps)) if ok_tps else None,
        "ttft_ms_mean": (sum(ttfts) / len(ttfts)) if ttfts else None,
    }
    # Representative text per prompt (last ok sample for that prompt)
    by_prompt: dict[str, str] = {}
    for s in samples:
        if s["status"] == "ok":
            by_prompt[s["prompt"]] = s["content"]
    return {
        **base,
        "status": "ok" if errors == 0 else "ok_partial",
        "reason": None if errors == 0 else f"{errors}/{len(samples)} samples failed",
        "perf": perf,
        "quality": {},  # filled later vs ref
        "texts_by_prompt": by_prompt,
        "samples": samples,
    }


def attach_quality(
    results: list[dict[str, Any]],
    *,
    ref_backend: str | None,
    prompts: list[str],
) -> None:
    if not ref_backend:
        for r in results:
            if r.get("status") in ("ok", "ok_partial"):
                r["quality"] = {"status": "skipped", "reason": "no ref_backend"}
        return
    ref_texts: dict[tuple[str, str], str] = {}
    for r in results:
        if r.get("backend") != ref_backend:
            continue
        if r.get("status") not in ("ok", "ok_partial"):
            continue
        fam = r["family"]
        for p, text in (r.get("texts_by_prompt") or {}).items():
            ref_texts[(fam, p)] = text

    for r in results:
        if r.get("status") not in ("ok", "ok_partial"):
            r["quality"] = {"status": "skipped", "reason": r.get("reason") or "not ok"}
            continue
        overlaps = []
        exacts = []
        for p in prompts:
            cand = (r.get("texts_by_prompt") or {}).get(p)
            ref = ref_texts.get((r["family"], p))
            if cand is None or ref is None:
                continue
            q = quality_vs_ref(cand, ref)
            overlaps.append(q["token_overlap"])
            exacts.append(1.0 if q["exact_match"] else 0.0)
        if not overlaps:
            r["quality"] = {
                "status": "skipped",
                "reason": f"missing ref texts from {ref_backend}",
            }
            continue
        r["quality"] = {
            "status": "ok",
            "ref_backend": ref_backend,
            "token_overlap_mean": sum(overlaps) / len(overlaps),
            "exact_match_rate": sum(exacts) / len(exacts),
            "n_prompts": len(overlaps),
        }


def build_summary(results: list[dict[str, Any]], families_n: int, backends_n: int) -> dict[str, Any]:
    ok = sum(1 for r in results if r.get("status") in ("ok", "ok_partial"))
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    err = sum(1 for r in results if r.get("status") == "error")
    overlaps = [
        r["quality"]["token_overlap_mean"]
        for r in results
        if (r.get("quality") or {}).get("status") == "ok"
        and isinstance((r.get("quality") or {}).get("token_overlap_mean"), (int, float))
    ]
    return {
        "families_total": families_n,
        "backends_configured": backends_n,
        "results_ok": ok,
        "results_skipped": skipped,
        "results_error": err,
        "mean_token_overlap_vs_ref": (sum(overlaps) / len(overlaps)) if overlaps else None,
    }


def run_bench(
    *,
    backends: list[BackendConfig],
    family_paths: list[str] | None = None,
    model_overrides: dict[str, str] | None = None,
    prompts: list[str] | None = None,
    max_tokens: int = 64,
    warmup: int = 1,
    runs: int = 3,
    ref_backend: str | None = None,
    measure_ttft: bool = False,
    skip_probe: bool = False,
    chat_fn: ChatFn = chat_completion,
) -> dict[str, Any]:
    families = select_families(family_paths)
    prompts = list(prompts or DEFAULT_PROMPTS)
    overrides = model_overrides or {}

    # Deduplicate by id (last wins)
    by_id: dict[str, BackendConfig] = {}
    for b in backends:
        by_id[b.id] = b
    configured = [by_id[i] for i in BACKEND_IDS if i in by_id]

    backend_meta = []
    live: list[BackendConfig] = []
    for b in configured:
        if skip_probe:
            ok, detail = True, "probe skipped"
        else:
            ok, detail = probe(b)
        backend_meta.append(
            {"id": b.id, "base_url": b.base_url, "probe_ok": ok, "probe_detail": detail}
        )
        if ok:
            live.append(b)

    ref = pick_ref_backend(live, ref_backend)
    results: list[dict[str, Any]] = []

    # Always emit rows for configured backends × families (skipped if probe failed)
    for fam in families:
        for b in configured:
            if b not in live:
                results.append(
                    {
                        "family": fam.path,
                        "base_model": fam.base_model,
                        "kind": fam.kind,
                        "backend": b.id,
                        "model_id": resolve_model_id(fam, b.id, overrides),
                        "status": "skipped",
                        "reason": "backend probe failed or unreachable",
                        "perf": {},
                        "quality": {"status": "skipped"},
                    }
                )
                continue
            mid = resolve_model_id(fam, b.id, overrides)
            results.append(
                run_family_backend(
                    fam,
                    b,
                    model_id=mid,
                    prompts=prompts,
                    max_tokens=max_tokens,
                    warmup=warmup,
                    runs=runs,
                    measure_ttft=measure_ttft,
                    chat_fn=chat_fn,
                )
            )

    attach_quality(results, ref_backend=ref, prompts=prompts)

    # Strip bulky sample bodies from top-level export? Keep samples but drop raw if huge.
    slim = []
    for r in results:
        rr = dict(r)
        rr.pop("texts_by_prompt", None)
        samples = rr.get("samples")
        if samples:
            rr["samples"] = [
                {
                    "prompt": s.get("prompt"),
                    "status": s.get("status"),
                    "latency_ms": s.get("latency_ms"),
                    "ttft_ms": s.get("ttft_ms"),
                    "completion_tokens": s.get("completion_tokens"),
                    "tokens_per_sec": s.get("tokens_per_sec"),
                    "content_preview": (s.get("content") or "")[:200],
                    "error": s.get("error"),
                }
                for s in samples
            ]
        slim.append(rr)

    report = {
        "mode": "engine_bench",
        "ci_fail": False,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "max_tokens": max_tokens,
            "warmup": warmup,
            "runs": runs,
            "prompts": prompts,
            "ref_backend": ref,
            "measure_ttft": measure_ttft,
        },
        "families": [{"path": f.path, "base_model": f.base_model, "kind": f.kind} for f in families],
        "backends": backend_meta,
        "results": slim,
        "summary": build_summary(slim, len(families), len(configured)),
    }
    return report
