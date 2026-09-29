#!/usr/bin/env bash
#
# Entrypoint for the aria-engine container.
#
# Responsibilities (in order):
#   1. Pin ARIA_COMPUTE_HOME so config + checkpoints live on the mounted volume.
#   2. Write engine.yml from env vars. A pre-mounted config is respected UNLESS
#      SITE_URL is set explicitly in the environment (so .env can override the
#      hub, e.g. switch an HF-baked host config to ModelScope inside the container).
#   3. When a checkpoint is missing and AUTO_DOWNLOAD=1, fetch it via
#      `aria-engine download` (needs network + the matching hub token; see the
#      proxy env vars below for egress through a host proxy).
#   4. exec `aria-engine serve` bound to 0.0.0.0 so it is reachable inside
#      the container network (the binary default is 127.0.0.1).

# Proxy passthrough: if the host reaches the Hub through a proxy, the container
# needs the same egress. For a host-local proxy (e.g. 127.0.0.1:7897) set these
# to http://host.docker.internal:<port> (we map that name to the host gateway
# in docker-compose.yml).
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}"
export NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export no_proxy="$NO_PROXY"

set -euo pipefail

ARIA_HOME="${ARIA_COMPUTE_HOME:-/data/aria}"
export ARIA_COMPUTE_HOME="$ARIA_HOME"
mkdir -p "$ARIA_HOME/models"

ENGINE_YML="$ARIA_HOME/engine.yml"

# --- 2. engine.yml (five-field contract from core/src/config.rs) ---
# Regenerate from env when SITE_URL is explicitly provided (container config is
# driven by .env), otherwise keep a pre-mounted config untouched.
if [ ! -f "$ENGINE_YML" ] || [ -n "${SITE_URL:-}" ]; then
  cat > "$ENGINE_YML" <<EOF
site_url: "${SITE_URL:-}"
upgrade_url: "${UPGRADE_URL:-}"
compute: "${COMPUTE:-auto}"
hf_token: "${HF_TOKEN:-}"
modelscope_api_token: "${MODELSCOPE_API_TOKEN:-}"
EOF
  if [ -n "${SITE_URL:-}" ]; then
    echo "[entrypoint] wrote $ENGINE_YML from env (SITE_URL=${SITE_URL})"
  else
    echo "[entrypoint] wrote $ENGINE_YML from env defaults"
  fi
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

# Always bind-mount ARIA_DATA_DIR with downloaded weights (offline-first). Only
# when AUTO_DOWNLOAD=1 do we try to fetch; a failed/blocked fetch must NOT crash
# the container (set -e would otherwise exit 1 and trigger a restart storm). If
# the checkpoint is still missing afterward, fail once with a clear message.
ensure_checkpoint() {
  if [ -d "$CKPT_DIR" ] && checkpoint_ready "$CKPT_DIR"; then
    echo "[entrypoint] checkpoint for '$CKPT_NAME' ready at $CKPT_DIR"
    return 0
  fi

  if [ "${AUTO_DOWNLOAD:-0}" = "1" ]; then
    echo "[entrypoint] checkpoint not ready for '$CKPT_NAME'; attempting download ..."
    if aria-engine download "$CKPT_NAME"; then
      echo "[entrypoint] download complete for '$CKPT_NAME'"
    else
      echo "[entrypoint] WARNING: download failed for '$CKPT_NAME' (network or auth); continuing"
    fi
  else
    echo "[entrypoint] checkpoint for '$CKPT_NAME' not found at $CKPT_DIR" \
         "(mount ARIA_DATA_DIR with downloaded weights, or set AUTO_DOWNLOAD=1 to fetch)"
  fi

  if [ ! -d "$CKPT_DIR" ] || ! checkpoint_ready "$CKPT_DIR"; then
    echo "[entrypoint] FATAL: no usable checkpoint for '$CKPT_NAME' at $CKPT_DIR; refusing to start" >&2
    exit 1
  fi
}

ensure_checkpoint

# --- 4. Serve (bound to 0.0.0.0 so the container port is reachable) ---
BIND_ADDR="0.0.0.0:${PORT:-8010}"
echo "[entrypoint] starting aria-engine track=${TRACK:-encoder} model=${CKPT_NAME} bind=${BIND_ADDR} compute=${COMPUTE:-auto}"
exec aria-engine serve \
  --track "${TRACK:-encoder}" \
  --model-name "$CKPT_NAME" \
  --bind "$BIND_ADDR" \
  --compute "${COMPUTE:-auto}"
