#!/usr/bin/env bash
# Materialize tiny q4 fixture for binding host tests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/bindings/testdata/tiny-q4}"
mkdir -p "$OUT"
cd "$ROOT"
cargo run -q -p ariacompute-ffi --example write_fixture -- "$OUT"
test -f "$OUT/config.json"
echo "fixture dir: $OUT"
