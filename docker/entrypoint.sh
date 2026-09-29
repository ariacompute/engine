#!/usr/bin/env bash
#
# Entrypoint for the aria-engine container.
#
# Responsibilities (in order):
#   1. Pin ARIA_COMPUTE_HOME so config + checkpoints live on the mounted volume.
#   2. Idempotently write engine.yml from env vars (never overwrite a
#      pre-mounted config that may already carry secrets).
#   3. When a checkpoint is missing and AUTO_DOWNLOAD=1, fetch it via
#      `aria-engine download` (needs network + the matching hub token).
#   4. exec `aria-engine serve` bound to 0.0.0.0 so it is reachable inside
#      the container network (the binary default is 127.0.0.1).

set -euo pipefail

ARIA_HOME="${ARIA_COMPUTE_HOME:-/data/aria}"
export ARIA_COMPUTE_HOME="$ARIA_HOME"
mkdir -p "$ARIA_HOME/models"

ENGINE_YML="$ARIA_HOME/engine.yml"

# --- 2. Idempotent engine.yml (five-field contract from core/src/config.rs) ---
if [ ! -f "$ENGINE_YML" ]; then
  cat > "$ENGINE_YML" <<EOF
site_url: "${SITE_URL:-}"
upgrade_url: "${UPGRADE_URL:-}"
compute: "${COMPUTE:-auto}"
hf_token: "${HF_TOKEN:-}"
modelscope_api_token: "${MODELSCOPE_API_TOKEN:-}"
EOF
  echo "[entrypoint] wrote $ENGINE_YML (set env vars to override; delete the file to regenerate)"
else
  echo "[entrypoint] using existing $ENGINE_YML"
fi

# --- 3. Optional model download (matches download.rs::looks_like_checkpoint) ---
CKPT_NAME="${MODEL:-afm-de}"
CKPT_DIR="$ARIA_HOME/models/$CKPT_NAME"

checkpoint_ready() {
  local d="$1"
  [ -f "$d/model.safetensors" ] || [ -f "$d/rl_agent_config.json" ] \
    || [ -f "$d/adapter_config.json" ] || [ -f "$d/dd_config.json" ]
}

if [ ! -d "$CKPT_DIR" ] || ! checkpoint_ready "$CKPT_DIR"; then
  if [ "${AUTO_DOWNLOAD:-0}" = "1" ]; then
    echo "[entrypoint] checkpoint not ready for '$CKPT_NAME'; downloading via aria-engine download ..."
    aria-engine download "$CKPT_NAME"
  else
    echo "[entrypoint] WARNING: checkpoint for '$CKPT_NAME' not found at $CKPT_DIR" \
         "(mount a volume with downloaded weights, or set AUTO_DOWNLOAD=1 to fetch on first run)"
  fi
fi

# --- 4. Serve (bound to 0.0.0.0 so the container port is reachable) ---
BIND_ADDR="0.0.0.0:${PORT:-8010}"
echo "[entrypoint] starting aria-engine track=${TRACK:-encoder} model=${CKPT_NAME} bind=${BIND_ADDR} compute=${COMPUTE:-auto}"
exec aria-engine serve \
  --track "${TRACK:-encoder}" \
  --model-name "$CKPT_NAME" \
  --bind "$BIND_ADDR" \
  --compute "${COMPUTE:-auto}"
