#!/usr/bin/env bash
# Build libaria_ffi and copy the platform dynamic library into the Python wheel.
# Used as cibuildwheel CIBW_BEFORE_ALL so each platform wheel bundles its own lib.
#
# Context: on Linux this runs INSIDE a manylinux container (no Rust preinstalled);
# on macOS/Windows it runs on the runner host (Rust preinstalled by the workflow).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- ensure a Rust toolchain is available -------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  echo "==> Installing rustup (minimal profile)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  # rustup installs into $HOME/.cargo; source when possible, then force PATH
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  command -v cargo >/dev/null 2>&1 || {
    echo "ERROR: cargo still not on PATH after rustup install" >&2
    exit 1
  }
fi

echo "==> cargo $(cargo --version) (CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-<default>})"

# --- refuse to build inside a musl (Alpine/musllinux) container ----------
# musl targets default to crt-static, where rustc drops the cdylib crate
# type, so no shared library is ever produced. musllinux wheels are disabled
# (CIBW_SKIP=musllinux*); if this check trips the skip is not being applied.
HOST_TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
case "$HOST_TRIPLE" in
  *-musl*)
    echo "ERROR: rust host triple is '$HOST_TRIPLE' (musl)." >&2
    echo "Rust drops the cdylib crate type on musl targets (crt-static default)," >&2
    echo "so libaria_ffi.so cannot be produced here. musllinux wheels are" >&2
    echo "disabled; make sure CIBW_SKIP=musllinux* is applied so only manylinux" >&2
    echo "builds run." >&2
    exit 1
    ;;
esac

# --- locate the built library (CARGO_TARGET_DIR aware) ------------------
case "$(uname -s)" in
  Darwin)          LIB="libaria_ffi.dylib"; DEST_NAME="libaria-engine_ffi.dylib";;
  MINGW*|MSYS*|CYGWIN*) LIB="aria_ffi.dll"; DEST_NAME="aria-engine_ffi.dll";;
  *)               LIB="libaria_ffi.so"; DEST_NAME="libaria-engine_ffi.so";;
esac

# --- build the FFI cdylib ------------------------------------------------
cargo build --release -p ariacompute-ffi

SRC=""
for c in "target/release/$LIB" "${CARGO_TARGET_DIR:+$CARGO_TARGET_DIR/release/$LIB}"; do
  if [ -n "$c" ] && [ -f "$c" ]; then
    SRC="$c"
    break
  fi
done
if [ -z "$SRC" ]; then
  # last resort: bounded search inside the repo tree
  SRC="$(find . -maxdepth 4 -type f -name "$LIB" -path '*/release/*' 2>/dev/null | head -1 || true)"
fi
if [ -z "$SRC" ]; then
  echo "ERROR: $LIB not found after 'cargo build --release -p ariacompute-ffi'" >&2
  echo "--- target/release ---" >&2
  ls -la target/release 2>/dev/null | head -20 || true
  echo "--- found libs ---" >&2
  find . -maxdepth 4 \( -name '*.so' -o -name '*.dylib' -o -name '*.dll' \) 2>/dev/null | head -20 || true
  exit 1
fi

DEST="bindings/python/aria_engine/lib"
mkdir -p "$DEST"
cp "$SRC" "$DEST/$DEST_NAME"
echo "FFI copied $SRC -> $DEST/$DEST_NAME"
