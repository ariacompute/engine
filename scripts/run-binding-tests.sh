#!/usr/bin/env bash
# Host binding smoke: build FFI, run Rust/Python import checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cargo build -p ariacompute-ffi -p ariacompute-engine
cargo test -p ariacompute-core -p ariacompute-de -p ariacompute-dd --lib
python3 -c "import sys; sys.path.insert(0, 'bindings/python'); import aria_engine; print('python import ok')"
echo "bindings-host smoke OK (init_ok/systemone_ok need weights or golden logit path)"
