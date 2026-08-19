#!/usr/bin/env bash
# Build libaria_ffi and copy the platform dynamic library into the Python wheel.
# Used as cibuildwheel CIBW_BEFORE_ALL so each platform wheels bundles its own lib.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# manylinux containers have no Rust toolchain; install minimal rustup if missing.
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  source "$HOME/.cargo/env"
fi

cargo build --release -p aria-ffi

DEST="bindings/python/aria_engine/lib"
mkdir -p "$DEST"
case "$(uname -s)" in
  Darwin) cp target/release/libaria_ffi.dylib "$DEST/";;
  MINGW*|MSYS*|CYGWIN*) cp target/release/aria_ffi.dll "$DEST/";;
  *) cp target/release/libaria_ffi.so "$DEST/";;
esac

echo "FFI copied -> $DEST"
