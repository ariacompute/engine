#!/usr/bin/env bash
# Host binding tests: build libariaengine_ffi, prepare fixture, run Rust/Python/Go/TS when available.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== cargo test ariacompute-ariaengine-ffi / ariacompute-ariaengine =="
cargo test -p ariacompute-ariaengine-ffi -p ariacompute-ariaengine

echo "== prepare fixture =="
FIX="$ROOT/bindings/testdata/tiny-q4"
mkdir -p "$FIX"
cargo run -q -p ariacompute-ariaengine-ffi --example write_fixture -- "$FIX"
cargo build -q -p ariacompute-ariaengine-ffi

export ARIA_BUNDLE="$FIX"
export ARIA_INCLUDE="$ROOT/ffi/include"
if [[ "$(uname)" == "Darwin" ]]; then
  export ARIAENGINE_FFI_LIB="$ROOT/target/debug/libariaengine_ffi.dylib"
elif [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
  export ARIAENGINE_FFI_LIB="$ROOT/target/debug/ariaengine_ffi.dll"
else
  export ARIAENGINE_FFI_LIB="$ROOT/target/debug/libariaengine_ffi.so"
fi
export LD_LIBRARY_PATH="${ROOT}/target/debug:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${ROOT}/target/debug:${DYLD_LIBRARY_PATH:-}"

echo "ARIAENGINE_FFI_LIB=$ARIAENGINE_FFI_LIB"
test -e "$ARIAENGINE_FFI_LIB"

if command -v python3 >/dev/null; then
  echo "== python =="
  (cd bindings/python && PYTHONPATH=. python3 -m unittest discover -s tests -t . -v)
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

if command -v node >/dev/null && [[ -f bindings/react-native/package.json ]]; then
  echo "== react-native setup =="
  (cd bindings/react-native && node --test test/setup.test.cjs)
fi

echo "done (mobile Swift/Kotlin/Flutter/RN: see bindings/*/README and bindings-mobile.yml)"
