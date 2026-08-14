# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute inference engine: OpenAI-compatible API, Aria bundle inference, zero-copy graph, ARM NEON / scalar kernels, hybrid router.

## Build / Test

```bash
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

## HTTP Serve

```bash
cargo run -p aria-openai --bin aria-engine -- serve \
  --model /path/to/aria-bundle \
  --bind 127.0.0.1:8080
```

Default bind is `127.0.0.1:8080`. The process listens for OpenAI-compatible HTTP.

## Hybrid Cloud

Configure via environment variables (no CLI flags):

| Variable | Meaning | Default |
|----------|---------|---------|
| `ARIA_HYBRID_CLOUD_URL` | Cloud OpenAI-compatible **base URL** (engine appends `/v1/chat/completions`) | `https://gateway.ariacompute.com` |
| `ARIA_HYBRID_CLOUD_API_KEY` | Bearer token; required for real cloud calls | _(empty → cloud errors)_ |
| `ARIA_HYBRID_THRESHOLD` | Confidence threshold in `[0,1]` | `0.0` |
| `ARIA_HYBRID_MODE` | `cost` / `balance` / `intelligence` | `balance` |
| `ARIA_HYBRID_EXECUTION` | `hybrid` (default) / `device` (on-device only) / `cloud` (cloud only) | `hybrid` |

```bash
export ARIA_HYBRID_CLOUD_URL=https://gateway.ariacompute.com
export ARIA_HYBRID_CLOUD_API_KEY=sk-...
export ARIA_HYBRID_THRESHOLD=0.5
export ARIA_HYBRID_MODE=balance
# optional: force device-only or cloud-only
# export ARIA_HYBRID_EXECUTION=device
# export ARIA_HYBRID_EXECUTION=cloud

cargo run -p aria-openai --bin aria-engine -- serve \
  --model /path/to/aria-bundle \
  --bind 127.0.0.1:8080
```

Force a cloud path in hybrid mode by including `FORCE_CLOUD` in the user message (tests / demos). `ARIA_HYBRID_EXECUTION=device` never handoffs; `=cloud` always handoffs (privacy-sensitive requests still stay local). Cloud handoff posts `model: "ariacompute/ariamodel"` to the gateway.

## OpenAI API

Assuming the server is on `http://127.0.0.1:8080`:

```bash
# List models
curl -s http://127.0.0.1:8080/v1/models | jq .

# Chat (non-stream)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens": 32,
    "temperature": 0
  }' | jq .

# Chat (SSE stream)
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens": 32,
    "stream": true
  }'

# Embeddings
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":"hello embedding"}' | jq .

# ASR stub (PCM16 LE bytes, base64)
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -H 'content-type: application/json' \
  -d '{"file_b64":"AAECAwQFBgc="}' | jq .
```

## Bench

```bash
# Start servers yourself, then:
python -m unittest discover -s bench/tests -t .
python -m bench list-families
python -m bench run \
  --backend aria=http://127.0.0.1:8080 \
  --backend llamacpp=http://127.0.0.1:8081 \
  --backend ollama=http://127.0.0.1:11434 \
  --backend vllm=http://127.0.0.1:8000 \
  --max-tokens 64 --warmup 1 --runs 3 \
  --report ./out/bench_report.json
# also writes ./out/bench_report.md
```

Override model ids with `--model-id family=path=id` or `backend:family=id`. Default aria model id is the family path; others use HF `base_model`.

## Engineering Conventions

This repository follows the Harness Engineering philosophy:

- [`AGENTS.md`](AGENTS.md): Agent engineering context entry and directory index
- [`requirements.md`](requirements.md): Requirements spec (feature boundaries/exceptions/acceptance criteria, human-review-gated)
- [`task.md`](task.md): Implementation task checklist
