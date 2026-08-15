#!/usr/bin/env bash
# Host binding tests: build libaria_ffi, prepare fixture, run Rust/Python/Go/TS when available.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== cargo test aria-ffi / aria-sdk =="
cargo test -p aria-ffi -p aria-sdk

echo "== prepare fixture =="
FIX="$ROOT/bindings/testdata/tiny-q4"
mkdir -p "$FIX"
cargo run -q -p aria-ffi --example write_fixture -- "$FIX"
cargo build -q -p aria-ffi

export ARIA_BUNDLE="$FIX"
export ARIA_INCLUDE="$ROOT/ffi/include"
if [[ "$(uname)" == "Darwin" ]]; then
  export ARIA_FFI_LIB="$ROOT/target/debug/libaria_ffi.dylib"
elif [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
  export ARIA_FFI_LIB="$ROOT/target/debug/aria_ffi.dll"
else
  export ARIA_FFI_LIB="$ROOT/target/debug/libaria_ffi.so"
fi
export LD_LIBRARY_PATH="${ROOT}/target/debug:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${ROOT}/target/debug:${DYLD_LIBRARY_PATH:-}"

echo "ARIA_FFI_LIB=$ARIA_FFI_LIB"
test -e "$ARIA_FFI_LIB"

if command -v python3 >/dev/null; then
  echo "== python =="
  (cd bindings/python && PYTHONPATH=. python3 -m unittest tests.test_binding -v)
fi

if command -v go >/dev/null; then
  echo "== go =="
  (cd bindings/go && CGO_ENABLED=1 go test -tags aria_ffi ./...)
fi

if command -v npm >/dev/null && [[ -f bindings/typescript/package.json ]]; then
  echo "== typescript =="
  (
    cd bindings/typescript
    npm install --silent
    npm run build
    npm test
  )
fi

echo "done (mobile Swift/Kotlin/Flutter/RN: see bindings/*/README and bindings-mobile.yml)"
