#!/usr/bin/env bash
# Build libaria_ffi and copy into bindings/python/aria_engine/lib for wheels.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cargo build --release -p ariacompute-ffi
mkdir -p bindings/python/aria_engine/lib
shopt -s nullglob
for f in target/release/libaria_ffi.so target/release/libaria_ffi.dylib target/release/aria_ffi.dll; do
  if [[ -f "$f" ]]; then
    base="$(basename "$f")"
    case "$base" in
      libaria_ffi.so) cp "$f" bindings/python/aria_engine/lib/libaria-engine_ffi.so ;;
      libaria_ffi.dylib) cp "$f" bindings/python/aria_engine/lib/libaria-engine_ffi.dylib ;;
      aria_ffi.dll) cp "$f" bindings/python/aria_engine/lib/aria-engine_ffi.dll ;;
    esac
  fi
done
echo "python ffi libs:"
ls -la bindings/python/aria_engine/lib || true
