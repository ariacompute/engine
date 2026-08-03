"""Write bench_report.json + bench_report.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_md_path(json_path: Path) -> Path:
    if json_path.suffix.lower() == ".json":
        return json_path.with_suffix(".md")
    return json_path.parent / (json_path.name + ".md")


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Aria Engine Bench Report")
    lines.append("")
    lines.append(f"- mode: `{report.get('mode')}`")
    lines.append(f"- ci_fail: `{report.get('ci_fail')}`")
    meta = report.get("meta") or {}
    if meta:
        lines.append(f"- generated_at: `{meta.get('generated_at')}`")
        lines.append(f"- max_tokens: `{meta.get('max_tokens')}`")
        lines.append(f"- warmup / runs: `{meta.get('warmup')}` / `{meta.get('runs')}`")
        lines.append(f"- ref_backend: `{meta.get('ref_backend')}`")
    lines.append("")

    summary = report.get("summary") or {}
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|--------|-------|")
    for k in (
        "families_total",
        "backends_configured",
        "results_ok",
        "results_skipped",
        "results_error",
        "mean_token_overlap_vs_ref",
    ):
        if k in summary:
            lines.append(f"| {k} | {summary[k]} |")
    lines.append("")

    lines.append("## Backends")
    lines.append("")
    lines.append("| id | base_url | probe |")
    lines.append("|----|----------|-------|")
    for b in report.get("backends") or []:
        lines.append(
            f"| {b.get('id')} | `{b.get('base_url')}` | {b.get('probe_ok')} ({b.get('probe_detail')}) |"
        )
    lines.append("")

    lines.append("## Results (perf + quality)")
    lines.append("")
    lines.append(
        "| family | kind | backend | status | p50_ms | p95_ms | tok/s | overlap | exact |"
    )
    lines.append(
        "|--------|------|---------|--------|--------|--------|-------|---------|-------|"
    )
    for r in report.get("results") or []:
        perf = r.get("perf") or {}
        qual = r.get("quality") or {}
        tps = perf.get("tokens_per_sec_mean")
        tps_s = f"{tps:.2f}" if isinstance(tps, (int, float)) else "-"
        ov = qual.get("token_overlap_mean")
        ov_s = f"{ov:.3f}" if isinstance(ov, (int, float)) else "-"
        ex = qual.get("exact_match_rate")
        ex_s = f"{ex:.2f}" if isinstance(ex, (int, float)) else "-"
        lines.append(
            "| {family} | {kind} | {backend} | {status} | {p50} | {p95} | {tps} | {ov} | {ex} |".format(
                family=r.get("family"),
                kind=r.get("kind"),
                backend=r.get("backend"),
                status=r.get("status"),
                p50=perf.get("p50_ms", "-"),
                p95=perf.get("p95_ms", "-"),
                tps=tps_s,
                ov=ov_s,
                ex=ex_s,
            )
        )
        if r.get("reason"):
            lines.append(f"| | | | reason | {r['reason']} | | | | |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Report-only: missing backends or HTTP errors do not fail CI (`ci_fail: false`).")
    lines.append("- Quality scores compare each backend output to `--ref-backend` on the same prompt.")
    lines.append("- VL/VLA families use text chat probes; multimodal-only servers may skip.")
    lines.append("")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")


def write_reports(
    report: dict[str, Any],
    json_path: Path,
    md_path: Path | None = None,
) -> tuple[Path, Path]:
    write_json(report, json_path)
    md = md_path or default_md_path(json_path)
    write_markdown(report, md)
    return json_path, md
