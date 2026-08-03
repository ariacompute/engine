"""§1.1 family registry — locked to model/tests/test_families.py EXPECTED."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Family:
    path: str
    base_model: str
    kind: str  # text | vl | vla


# Mirrors model EXPECTED + engine ArchClass mapping.
FAMILIES: tuple[Family, ...] = (
    Family("qwen/qwen3-0.6b", "Qwen/Qwen3-0.6B", "text"),
    Family("qwen/qwen3-1.7b", "Qwen/Qwen3-1.7B", "text"),
    Family("qwen/qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "text"),
    Family("qwen/qwen3.5-2b", "Qwen/Qwen3.5-2B", "text"),
    Family("gemma/gemma-3-270m-it", "google/gemma-3-270m-it", "text"),
    Family("gemma/gemma-3-1b-it", "google/gemma-3-1b-it", "text"),
    Family("gemma/gemma-3n-e2b-it", "google/gemma-3n-E2B-it", "vl"),
    Family("gemma/gemma-3n-e4b-it", "google/gemma-3n-E4B-it", "vl"),
    Family("gemma/gemma-4-e2b-it", "google/gemma-4-E2B-it", "vl"),
    Family("gemma/gemma-4-e4b-it", "google/gemma-4-E4B-it", "vl"),
    Family("lfm/lfm2-350m", "LiquidAI/LFM2-350M", "text"),
    Family("lfm/lfm2-700m", "LiquidAI/LFM2-700M", "text"),
    Family("lfm/lfm2-1.2b", "LiquidAI/LFM2-1.2B", "text"),
    Family("lfm/lfm2-2.6b", "LiquidAI/LFM2-2.6B", "text"),
    Family("lfm/lfm2-8b-a1b", "LiquidAI/LFM2-8B-A1B", "text"),
    Family("lfm/lfm2-vl-450m", "LiquidAI/LFM2-VL-450M", "vl"),
    Family("lfm/lfm2.5-350m", "LiquidAI/LFM2.5-350M", "text"),
    Family("lfm/lfm2.5-1.2b-instruct", "LiquidAI/LFM2.5-1.2B-Instruct", "text"),
    Family("lfm/lfm2.5-1.2b-thinking", "LiquidAI/LFM2.5-1.2B-Thinking", "text"),
    Family("lfm/lfm2.5-vl-1.6b", "LiquidAI/LFM2.5-VL-1.6B", "vl"),
    Family("nanbeige/nanbeige4.2-3b", "Nanbeige/Nanbeige4.2-3B", "text"),
    Family("bonsai/bonsai-27b", "prism-ml/Bonsai-27B-unpacked", "text"),
    Family("inkling/inkling-small", "thinkingmachines/Inkling-Small", "text"),
    Family("openvla/openvla-7b", "openvla/openvla-7b", "vla"),
    Family("openpi/openpi-pi0-3b", "lerobot/pi0_base", "vla"),
    Family("openpi/openpi-pi0.5-3b", "lerobot/pi05_base", "vla"),
    Family("lingbot/lingbot-vla-v2-6b", "robbyant/lingbot-vla-v2-6b", "vla"),
)

BACKEND_IDS = ("aria", "llamacpp", "ollama", "vllm")


def lookup(path: str) -> Family | None:
    for f in FAMILIES:
        if f.path == path:
            return f
    return None


def select_families(paths: Iterable[str] | None) -> list[Family]:
    if not paths:
        return list(FAMILIES)
    wanted = set(paths)
    out = [f for f in FAMILIES if f.path in wanted]
    missing = wanted - {f.path for f in out}
    if missing:
        raise ValueError(f"unknown family path(s): {sorted(missing)}")
    return out
