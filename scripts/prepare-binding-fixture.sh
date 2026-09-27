#!/usr/bin/env bash
# Prepare a tiny fixture dir for binding tests (no Hub weights).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$ROOT/bindings/testdata/checkpoint"
mkdir -p "$FIX"
echo '{"encoder":"test","max_len":1024,"head_max_len":512}' > "$FIX/rl_agent_config.json"
echo "fixture at $FIX"
