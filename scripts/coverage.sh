#!/usr/bin/env bash
# Line-coverage gate for the engine workspace.
#
# Usage:
#   ./scripts/coverage.sh            # gate + per-file summary table
#   ./scripts/coverage.sh html       # local report: target/coverage/html/index.html
#   ./scripts/coverage.sh lcov       # gate + lcov artifact: target/coverage/lcov.info
#
# Extra arguments are forwarded to `cargo llvm-cov`.
#
# Why the gate is computed from lcov instead of `--fail-under-lines`:
# `cargo llvm-cov --all-targets` instruments the same source file several times
# (lib for unit tests, lib as a dependency of the integration/CLI tests, the
# `aria-engine` binary). llvm-cov's per-file summary counts those line *entries*
# pairwise but unions them for the textual report, so the summary can report a
# handful of "missed" lines that the per-line and per-region views show as
# covered (the repo has no uncovered line or region). Gating on the lcov
# `DA:<line>,0` records keeps the intended semantics — every line executed at
# least once by some test — and is stable across llvm-cov versions.
#
# Set COVERAGE_ALLOW_MISSES=1 to print the misses without failing (diagnostics).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Files excluded from the 100% line gate — pure candle tensor forwards whose
# branches can only be exercised with the real checkpoints (ModernBERT for the
# encoder, MiniCPM5-2B for the decoder). Everything around them (checkpoint
# layout, HF config mapping, weight-key remapping, LoRA merge, letters, packing,
# scoring) is covered by unit tests. See requirements.md §4 for the rationale.
EXCLUDE_REGEX='(de/src/model/forward\.rs|dd/src/forward\.rs)'

MODE="${1:-summary}"
shift || true

COMMON=(--workspace --all-targets --ignore-filename-regex "$EXCLUDE_REGEX")

# Fail when any covered file has a `DA:<line>,0` record.
gate() {
  local lcov="$1"
  python3 - "$lcov" <<'PY'
import sys

miss = []
total = 0
current = ""
for raw in open(sys.argv[1], encoding="utf-8"):
    line = raw.strip()
    if line.startswith("SF:"):
        current = line[3:]
    elif line.startswith("DA:"):
        number, count = line[3:].split(",")[:2]
        total += 1
        if int(count) == 0:
            miss.append((current, int(number)))

covered = total - len(miss)
pct = 100.0 * covered / total if total else 100.0
print(f"lines: {covered}/{total} ({pct:.2f}%)")
if miss:
    print(f"uncovered lines: {len(miss)}", file=sys.stderr)
    for path, number in miss[:80]:
        print(f"  {path}:{number}", file=sys.stderr)
    if len(miss) > 80:
        print(f"  … {len(miss) - 80} more", file=sys.stderr)
    sys.exit(1)
PY
}

run_lcov() {
  mkdir -p target/coverage
  cargo llvm-cov "${COMMON[@]}" --lcov --output-path target/coverage/lcov.info "$@"
  if [ "${COVERAGE_ALLOW_MISSES:-0}" = "1" ]; then
    gate target/coverage/lcov.info || true
  else
    gate target/coverage/lcov.info
  fi
}

case "$MODE" in
  summary)
    run_lcov "$@"
    echo
    echo "note: the 'Missed Lines' column below counts per-instantiation line entries"
    echo "      (see the header comment); the gate result above is authoritative."
    cargo llvm-cov report --summary-only --ignore-filename-regex "$EXCLUDE_REGEX"
    ;;
  html)
    if [ "${COVERAGE_ALLOW_MISSES:-0}" = "1" ]; then
      cargo llvm-cov "${COMMON[@]}" --lcov --output-path target/coverage/lcov.info "$@"
      gate target/coverage/lcov.info || true
    else
      run_lcov "$@"
    fi
    mkdir -p target/coverage
    cargo llvm-cov report --html --output-dir target/coverage/html \
      --ignore-filename-regex "$EXCLUDE_REGEX"
    echo "report: target/coverage/html/index.html"
    ;;
  lcov)
    run_lcov "$@"
    echo "lcov: target/coverage/lcov.info"
    ;;
  *)
    echo "usage: $0 [summary|html|lcov] [extra cargo-llvm-cov args...]" >&2
    exit 2
    ;;
esac
