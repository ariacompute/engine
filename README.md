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
| `cloud_url` | Gateway base URL (auto-detected from API key region) | — |
| `site_url` | Site for downloads (same region as `cloud_url`) | — |
| `upgrade_url` | Org root for CLI/FFI upgrades (`.com`→GitHub, `.cn`→Gitee) | — |
| `hybrid_mode` | `cost` / `balance` / `intelligence` | `balance` |
| `hybrid_execution` | `hybrid` / `device` / `cloud` | `hybrid` |
| `hybrid_semantic` | Semantic routing layer switch (auto-off without cloud credentials) | `true` |
| `hybrid_semantic_timeout_ms` | Semantic routing per-consult timeout | `800` |
| `hybrid_semantic_cache_size` | Semantic decision cache capacity (TTL 60s) | `512` |
| `compute` | `auto` / `cpu` / `cuda` (local GEMM; **not** a hybrid switch) | `auto` |

```bash
# Auth
aria-engine auth
aria-engine auth --status

# Download models
aria-engine download gemma-4-e2b-it_q4
aria-engine list
aria-engine clean gemma-4-e2b-it_q4

# Upgrade CLI + libaria_ffi (latest stable, or a tag)
# FFI lands in ~/.ariacompute/lib/ — set ARIA_FFI_LIB if needed
aria-engine upgrade
aria-engine upgrade 0.7.2

# Serve
# or: serve /path/to/aria-bundle
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-mode balance \
  --hybrid-execution hybrid \
  --compute auto
```

`download` probes Serve (if keyed) / HuggingFace / ModelScope each run and picks the fastest reachable source. On a TTY it shows a green progress bar for the transfer.

`list` queries `{site_url}/api/dashboard/models` (requires `aria-engine auth`) and prints each downloadable bundle as `downloaded` / `not downloaded` (plus local-only caches).

`serve` flags override config for that process only (no rewrite). `serve <model>` uses a filesystem path if it exists, otherwise `~/.ariacompute/models/<model>`.

`--hybrid-execution` only controls cloud handoff (`device` never leaves the box). `--compute auto|cpu|cuda` selects **local** GEMM: `auto` uses CUDA when `libcudart`/`libcublas` load and `cudaGetDeviceCount>0`, otherwise CPU (AVX2+FMA on x86_64, NEON on aarch64). `--compute cuda` **fails** if no NVIDIA device — it does not silently fall back. CUDA is runtime-loaded (no toolkit at compile time); on H200 you can still pass `--features cuda` as a documentation flag:

```bash
cargo build -p aria-openai --release --features cuda
aria-engine serve qwen3-0.6b_q4 --hybrid-execution device --compute auto --profile
```

`--profile` records load/generate timings. Read them with `GET /v1/engine/profile` or:

```bash
python scripts/profile_qwen3_serve.py --compute cpu --spawn --report ./out/engine_profile_qwen3.json
```

In `--hybrid-execution hybrid`, routing uses prompt complexity / context overflow / modality / local failures / `FORCE_CLOUD`. `cost` prefers on-device; `intelligence` prefers cloud; `balance` is neutral auto. Include `FORCE_CLOUD` in the user message to force cloud (tests / demos). `--hybrid-execution device` never handoffs; `--hybrid-execution cloud` always handoffs (privacy-sensitive requests still stay local). Cloud handoff posts `model: "ariacompute/ariamodel"` to the gateway.

Routing is two-layer (rule layer fast path + semantic layer slow path). Deterministic rules decide most requests in <5ms; only uncertain ones (complexity near the mode cutoff, agentic/long-context prompts) consult the **semantic layer**, which asks the cloud gateway for a structured JSON intent decision (cached 60s, ≤800ms). Semantic disabled / timeout / failure silently falls back to the rule layer — never errors. Backend health scores (success/failure/timeout) feed a fallback flip; hard constraints (device/privacy) are never flipped. Inspect recent decisions and health via `GET /v1/engine/routes?n=20`; disable the semantic layer per process with `--hybrid-semantic off`.

## OpenAI API

Assuming the server is on `http://127.0.0.1:8080`:

```bash
# List models
curl -s http://127.0.0.1:8080/v1/models | jq .

# Load / generate timings (requires serve --profile)
curl -s http://127.0.0.1:8080/v1/engine/profile | jq .

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

## Qwen3 chat diagnostic

Use this pair when `/v1/chat/completions` looks like garbage (e.g. Hello → `"olum啦…"`) but the Aria bundle is fine under HF. Both sides share the same ChatML (`enable_thinking=False` / empty `<think>`), default user `Hello`, greedy `max_tokens=32`.

| Script | What it isolates |
|--------|------------------|
| [`../model/scripts/diag_qwen3_chat.py`](../model/scripts/diag_qwen3_chat.py) | **Quant + template**: HF fp32 vs `reconstruct_weight` inject into the same HF graph |
| [`scripts/diag_qwen3_chat.py`](scripts/diag_qwen3_chat.py) | **Engine graph**: this server vs the model JSON (`--peer-report`) |

**1. Model teacher** (sibling `model` repo; GPU recommended; tokenizer from `--hf`, not the bundle):

```bash
# from ../model
pip install torch transformers
python scripts/diag_qwen3_chat.py \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --hf Qwen/Qwen3-0.6B \
  --device cuda \
  --report ./out/model_diag_qwen3.json
```

`--device auto` picks CUDA when available. Attention is eager (no hub CUDA JIT). If compile still fails: `sudo apt install python3-dev`.

Read `chat.fp32` vs `chat.reconstruct`, plus `template_string_match` / `prompt_ids_match`. A reconstruct greeting such as `"Hello! How can I assist you today?"` with `exact_prefix_len >= 4` means the **bundle is usable in HF**; leftover engine garbage is not a quant bug.

**2. Engine** (serve the **same** bundle, no cloud handoff):

```bash
# from this repo; keep --hybrid-execution device so hybrid cannot mask local decode
./aria-engine serve qwen3-0.6b_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-execution device \
  --compute auto --profile
python scripts/diag_qwen3_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --peer-report ../model/out/model_diag_qwen3.json \
  --report ./out/engine_diag_qwen3.json
```

`--timeout` defaults to 300s (CPU decode can be slow). Optional: `pip install tokenizers` so prompt ids are encoded from `bundle/tokenizer.json`.

**How to read `hints`**

| Observation | Likely cause |
|-------------|--------------|
| `template_string_match` / `prompt_ids_match` is false | ChatML / tokenizer encode drift |
| reconstruct greedy already diverges from fp32 (`QUANT:…`) | codebook / Hadamard / inject |
| reconstruct chat looks ok, engine `content` does not (`ENGINE_GRAPH`) | Rust forward (HDM, embed row gather, RoPE, QK-norm, …) |

Teacher for the engine is **HF + reconstruct inject**, not raw fp32. More detail: [`../model/README.md`](../model/README.md) (Quality audit).

## Gemma-4 chat diagnostic

Use this pair when `gemma-4-e2b-it_q4` `/v1/chat/completions` is garbage (engine.log Hello → `"uhnyaчь…"`) while Qwen3 on the same host is fine. Both sides share `gemma4_it` (`<bos><|turn>user…<turn|>\n<|turn>model\n`), default user `Hello`, greedy `max_tokens=32`. Official Hello prompt is **10** tokens.

| Script | What it isolates |
|--------|------------------|
| [`../model/scripts/diag_gemma4_chat.py`](../model/scripts/diag_gemma4_chat.py) | **Quant + template**: HF fp32 vs `reconstruct_weight` inject into the same HF graph |
| [`scripts/diag_gemma4_chat.py`](scripts/diag_gemma4_chat.py) | **Engine graph**: this server vs the model JSON (`--peer-report`) |

**1. Model teacher** (sibling `model` repo; GPU recommended):

```bash
# from ../model
pip install torch transformers
python scripts/diag_gemma4_chat.py \
  --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \
  --hf google/gemma-4-E2B-it \
  --device cuda \
  --report ./out/model_diag_gemma4.json
```

**2. Engine** (serve the **same** bundle, no cloud handoff):

```bash
# from this repo; keep --hybrid-execution device so hybrid cannot mask local decode
./aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-execution device \
  --compute auto --profile
python scripts/diag_gemma4_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \
  --peer-report ../model/out/model_diag_gemma4.json \
  --report ./out/engine_diag_gemma4.json
```

Read `hints` like Qwen3 (`TEMPLATE` / `QUANT` / `ENGINE_GRAPH`). Hello `prompt_tokens` should be **10** with the Gemma-4 `<|turn>` template.

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

## SDK Bindings

Native C ABI (`aria-ffi` / `libaria_ffi`) plus thin wrappers under `bindings/`.

| Binding | Path | Registry |
|---------|------|----------|
| Rust | `bindings/rust` (`ariacompute-engine`) | crates.io |
| Python | `bindings/python` | PyPI |
| Go | `bindings/go` | Go module |
| TypeScript | `bindings/typescript` | npm `@ariacompute/engine-ts` |
| React Native | `bindings/react-native` | npm `@ariacompute/engine-rn` |
| Flutter | `bindings/flutter` | pub.dev |
| Swift | `bindings/swift` | CocoaPods |
| Kotlin | `bindings/kotlin` | Maven |

C header: [`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`, `aria_complete` / stream, `aria_embed`, `aria_transcribe`, tools JSON, `aria_model_destroy`, `aria_last_error`.

```bash
cargo test -p aria-ffi -p ariacompute-engine
./scripts/run-binding-tests.sh   # host matrix (Rust / Python / Go / TS)
```

Mobile e2e (Flutter + React Native, iOS + Android): [`.github/workflows/bindings-mobile.yml`](.github/workflows/bindings-mobile.yml).

### libaria_ffi (Release assets)

On each GitHub Release, [`.github/workflows/release.yml`](.github/workflows/release.yml) uploads platform archives next to the CLI:

| Asset | Contents |
|-------|----------|
| `libaria_ffi_<ver>_linux_x86_64.tar.gz` | `libaria_ffi.so` |
| `libaria_ffi_<ver>_linux_arm64.tar.gz` | `libaria_ffi.so` |
| `libaria_ffi_<ver>_macos.tar.gz` | `libaria_ffi.dylib` |
| `libaria_ffi_<ver>_windows_x86_64.tar.gz` | `aria_ffi.dll` (+ optional import libs) |

```bash
# Example: Linux x86_64
tar -xzf libaria_ffi_0.7.1_linux_x86_64.tar.gz
export ARIA_FFI_LIB="$PWD/libaria_ffi.so"
# optional: export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
```

Point bindings at a local Aria bundle (`weight.bin` + `config.json` + tokenizer), e.g. from `aria-engine download …` under `~/.ariacompute/models/`.

### Examples

**Python** (loads `libaria_ffi` via `ARIA_FFI_LIB`):

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
pip install aria-engine
```

```python
from aria_engine import Engine

with Engine("/path/to/aria-bundle") as eng:
    out = eng.complete(
        [{"role": "user", "content": "Hello"}],
        {"max_tokens": 32},
    )
    print(out["response"])
    # also: eng.embed("hi"), eng.transcribe(pcm_bytes)
```

**TypeScript / Node** (`@ariacompute/engine-ts`):

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
npm install @ariacompute/engine-ts
```

```ts
import { Engine } from "@ariacompute/engine-ts";

const eng = new Engine("/path/to/aria-bundle");
const out = eng.complete(
  [{ role: "user", content: "Hello" }],
  { max_tokens: 32 },
);
console.log(out.response);
eng.close();
```

**Go** (cgo; link against `libaria_ffi`):

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
export CGO_ENABLED=1
# Ensure the linker can find the library (or use -L via cgo in the module).
go get github.com/ariacompute/engine/bindings/go@latest
```

```go
package main

import (
	"fmt"
	aria "github.com/ariacompute/engine/bindings/go"
)

func main() {
	eng, err := aria.Open("/path/to/aria-bundle")
	if err != nil {
		panic(err)
	}
	defer eng.Close()
	out, err := eng.Complete(
		[]map[string]string{{"role": "user", "content": "Hello"}},
		map[string]any{"max_tokens": 32},
		nil,
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(out["response"])
}
```

**Rust** (`ariacompute-engine` crate — native API; does not require unpacking `libaria_ffi`):

```bash
cargo add ariacompute-engine
```

```rust
use aria_engine::{Engine, GenerateOpts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut eng = Engine::open("/path/to/aria-bundle")?;
    let g = eng.complete("Hello", &GenerateOpts {
        max_tokens: 32,
        temperature: 0.0,
    })?;
    println!("{}", g.text);
    Ok(())
}
```

More detail per language: `bindings/*/README.md`.

### Install from registries

Cut a **GitHub Release** — [`.github/workflows/release.yml`](.github/workflows/release.yml) builds CLI + `libaria_ffi` archives and attempts package publish (npm / pub.dev / Maven / CocoaPods / crates.io / PyPI). **Publish failures are fail-pass** and do not block CLI/`libaria_ffi` assets. crates.io (`ariacompute-engine`) is published by [`.github/workflows/publish-cargo.yml`](.github/workflows/publish-cargo.yml). Secrets: `NPM_TOKEN`, pub credentials, Maven + GPG, `COCOAPODS_TRUNK_TOKEN`, `CARGO_REGISTRY_TOKEN`, `PYPI_TOKEN`, plus `ARIACOMPUTE_TOKEN` for Release uploads.

Version = release tag without leading `v`.

## Engineering Conventions

This repository follows the Harness Engineering philosophy:

- [`AGENTS.md`](AGENTS.md): Agent engineering context entry and directory index
- [`requirements.md`](requirements.md): Requirements spec (feature boundaries/exceptions/acceptance criteria, human-review-gated)
- [`task.md`](task.md): Implementation task checklist
