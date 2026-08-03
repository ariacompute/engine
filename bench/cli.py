"""CLI: ``python -m bench run …`` — engine vs mainstream backends (report-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends import parse_backend_arg
from .families import BACKEND_IDS, FAMILIES
from .report import write_reports
from .runner import run_bench


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description=(
            "Compare aria-engine with llama.cpp / Ollama / vLLM over OpenAI chat "
            "(report-only JSON+MD; never fails CI on thresholds)"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run multi-backend bench and write reports")
    pr.add_argument(
        "--backend",
        action="append",
        default=[],
        metavar="ID=URL",
        help=f"repeatable; id in {BACKEND_IDS}",
    )
    pr.add_argument(
        "--family",
        action="append",
        default=None,
        help="family path filter (repeatable); default: all §1.1",
    )
    pr.add_argument(
        "--model-id",
        action="append",
        default=[],
        metavar="KEY=ID",
        help="override model id: family_path=id or backend:family_path=id or backend:*=id",
    )
    pr.add_argument("--prompt", action="append", default=None, help="prompt (repeatable)")
    pr.add_argument("--max-tokens", type=int, default=64)
    pr.add_argument("--warmup", type=int, default=1)
    pr.add_argument("--runs", type=int, default=3)
    pr.add_argument(
        "--ref-backend",
        default=None,
        help="quality reference backend id (default: llamacpp else first non-aria)",
    )
    pr.add_argument(
        "--ttft",
        action="store_true",
        help="measure time-to-first-token via SSE (extra request per sample)",
    )
    pr.add_argument(
        "--skip-probe",
        action="store_true",
        help="do not GET /v1/models before running",
    )
    pr.add_argument(
        "--report",
        default="bench_report.json",
        help="JSON report path (Markdown written alongside)",
    )
    pr.add_argument("--report-md", default=None, help="Markdown path (default: sibling .md)")
    pr.add_argument(
        "--api-key",
        action="append",
        default=[],
        metavar="ID=KEY",
        help="optional bearer token per backend",
    )
    pr.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds")

    pl = sub.add_parser("list-families", help="Print §1.1 family registry")
    pl.add_argument("--json", action="store_true")
    return p


def _parse_kv_list(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"expected KEY=VALUE, got {spec!r}")
        k, v = spec.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "list-families":
            rows = [
                {"path": f.path, "base_model": f.base_model, "kind": f.kind} for f in FAMILIES
            ]
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                for r in rows:
                    print(f"{r['path']}\t{r['kind']}\t{r['base_model']}")
            return 0

        if not args.backend:
            print(
                "error: provide at least one --backend id=url "
                "(aria / llamacpp / ollama / vllm)",
                file=sys.stderr,
            )
            return 2

        backends = [parse_backend_arg(s) for s in args.backend]
        keys = _parse_kv_list(args.api_key)
        for b in backends:
            if b.id in keys:
                b.api_key = keys[b.id]
            b.timeout_s = args.timeout

        report = run_bench(
            backends=backends,
            family_paths=args.family,
            model_overrides=_parse_kv_list(args.model_id),
            prompts=args.prompt,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            runs=args.runs,
            ref_backend=args.ref_backend,
            measure_ttft=args.ttft,
            skip_probe=args.skip_probe,
        )
        json_path = Path(args.report)
        md_path = Path(args.report_md) if args.report_md else None
        jp, mp = write_reports(report, json_path, md_path)
        print(
            json.dumps(
                {
                    "report": str(jp),
                    "report_md": str(mp),
                    "summary": report.get("summary"),
                    "ci_fail": report.get("ci_fail"),
                },
                indent=2,
            )
        )
        return 0
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
