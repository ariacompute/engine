"""Performance aggregates and quality scores (aligned with model gen_compare)."""

from __future__ import annotations

import statistics
from typing import Any, Sequence


def token_overlap(a: str, b: str) -> float:
    """Whitespace-token Jaccard; same semantics as model gen_compare._token_overlap."""
    ta = a.split()
    tb = b.split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    return len(sa & sb) / max(len(sa | sb), 1)


def exact_match(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def aggregate_latencies(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "n": 0}
    s = sorted(float(x) for x in latencies_ms)
    return {
        "mean_ms": float(statistics.fmean(s)),
        "p50_ms": percentile(s, 50),
        "p95_ms": percentile(s, 95),
        "n": len(s),
    }


def tokens_per_sec(completion_tokens: int, latency_ms: float) -> float | None:
    if latency_ms <= 0 or completion_tokens <= 0:
        return None
    return completion_tokens / (latency_ms / 1000.0)


def estimate_completion_tokens(text: str, reported: int | None) -> tuple[int, str]:
    if reported is not None and reported > 0:
        return reported, "usage"
    # Heuristic: ~4 chars / token
    est = max(1, len(text) // 4) if text else 0
    return est, "char_heuristic"


def quality_vs_ref(candidate: str, reference: str) -> dict[str, Any]:
    return {
        "token_overlap": token_overlap(candidate, reference),
        "exact_match": exact_match(candidate, reference),
        "candidate_chars": len(candidate),
        "reference_chars": len(reference),
    }
