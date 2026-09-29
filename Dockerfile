# syntax=docker/dockerfile:1

# Multi-stage build for the `aria-engine` System One inference server.
#
# The image is parameterized so a single Dockerfile serves both the CPU and the
# CUDA/GPU variants. Switch the build base + features via build args (see
# docker-compose.yml and docker-compose.cuda.yml):
#
#   CPU : BUILD_IMAGE=rust:1-slim           FEATURES=""  (runtime reuses this image)
#   GPU : BUILD_IMAGE=nvidia/cuda:...    FEATURES=cuda  (override RUNTIME_IMAGE too)
#
# Only the `aria-cli` crate is compiled (the `cuda` feature chains into
# ariacompute-de/dd); the FFI/shared library and language bindings are not
# needed to run `aria-engine serve`.

############################ Build stage ############################
ARG BUILD_IMAGE=rust:1-slim
# The runtime stage reuses the build image by default, so the glibc of the
# compiled binary always matches the one it runs on (no Debian-release drift).
# The CUDA override swaps in the matching slim CUDA runtime image.
ARG RUNTIME_IMAGE=${BUILD_IMAGE}
FROM ${BUILD_IMAGE} AS build

ARG FEATURES=""
ARG ARIA_ENGINE_VERSION=""

# Build-time system tools. A C toolchain is required because some engine
# dependencies (e.g. `tokenizers` with the `onig` feature) compile C sources
# via `cc`. curl/gnupg are only needed when installing rustup below.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       build-essential pkg-config ca-certificates curl gnupg git \
  && rm -rf /var/lib/apt/lists/*

# The CUDA build base (nvidia/cuda:*-devel-ubuntu) ships no Rust toolchain.
# Install a minimal rustup toolchain only when cargo is missing; the default
# rust:1-slim base already provides it, so this step is a no-op there.
RUN if ! command -v cargo >/dev/null 2>&1; then \
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
         | sh -s -- -y --profile minimal --default-toolchain stable \
      && . "$HOME/.cargo/env"; \
    fi
ENV PATH="/usr/local/cargo/bin:/root/.cargo/bin:${PATH}"

WORKDIR /src
COPY . .

# Cache the registry dir across builds (BuildKit) to speed dependency downloads.
# NOTE: the `target` dir is also cached, but cache mounts are NOT part of the
# committed image layer — so we copy the binary out to /out here (a plain layer
# path) for the later `COPY --from=build` to pick up.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p aria-cli --features "${FEATURES}" \
    && mkdir -p /out \
    && cp /src/target/release/aria-engine /out/aria-engine

############################ Runtime stage ############################
FROM ${RUNTIME_IMAGE} AS runtime

# curl is required for the /health healthcheck; ca-certificates for hub downloads.
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /out/aria-engine /usr/local/bin/aria-engine
COPY docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
  && mkdir -p /data/aria/models

# Stable env defaults; compose/dotenv override per service.
ENV ARIA_COMPUTE_HOME=/data/aria \
    RUST_LOG=info \
    PORT=8010 \
    TRACK=encoder \
    MODEL=afm-de \
    COMPUTE=auto \
    AUTO_DOWNLOAD=0

EXPOSE 8010 8011
ENTRYPOINT ["docker-entrypoint.sh"]
