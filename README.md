# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute inference engine: OpenAI-compatible API, Aria bundle inference, zero-copy graph, ARM NEON / scalar kernels, hybrid router.

## Build / Test

```bash
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

## Config / Run

Credentials and hybrid prefs live in `~/.ariacompute/config.yml` (via `aria-engine auth`). 

| Field | Meaning | Default |
|-------|---------|---------|
| `cloud_api_key` | Hybrid Bearer key | _(empty → cloud errors)_ |
| `cloud_url` | Gateway base URL (auto-detected on auth) | — |
| `site_url` | Site for downloads  (auto-detected on download) | — |
| `hybrid_mode` | `cost` / `balance` / `intelligence` | `balance` |
| `hybrid_execution` | `hybrid` / `device` / `cloud` | `hybrid` |

```bash
# Auth
aria-engine auth
aria-engine auth --status

# Download
aria-engine download gemma-4-e2b-it_q4
aria-engine list
aria-engine clean gemma-4-e2b-it_q4

# Serve
# or: serve /path/to/aria-bundle
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-mode balance \
  --hybrid-execution hybrid
```

`download` probes Serve (if keyed) / HuggingFace / ModelScope each run and picks the fastest reachable source.

`serve` flags override config for that process only (no rewrite). `serve <model>` uses a filesystem path if it exists, otherwise `~/.ariacompute/models/<model>`.

In `--hybrid-execution hybrid`, routing uses prompt complexity / context overflow / modality / local failures / `FORCE_CLOUD`. `cost` prefers on-device; `intelligence` prefers cloud; `balance` is neutral auto. Include `FORCE_CLOUD` in the user message to force cloud (tests / demos). `--hybrid-execution device` never handoffs; `--hybrid-execution cloud` always handoffs (privacy-sensitive requests still stay local). Cloud handoff posts `model: "ariacompute/ariamodel"` to the gateway.

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
