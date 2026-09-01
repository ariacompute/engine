# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute inference engine: OpenAI-compatible API, Aria bundle inference, zero-copy graph, ARM NEON / scalar kernels. Mixture-of-Models routing lives in the **router** repo (`aria-router`).

## Build / Test

```bash
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

## Config / Run

Credentials live in `~/.ariacompute/engine.yml` (via `aria-engine setup`).

| Field | Meaning | Default |
|-------|---------|---------|
| `router` | Optional aria-router management URL | _(empty → local-only serve)_ |
| `router_api_key` | Dashboard-issued secret for provider registration Bearer | _(empty)_ |
| `site_url` | Site for hub region (`.com` / `.cn`) | — |
| `upgrade_url` | Org root for CLI/FFI upgrades (`.com`→GitHub, `.cn`→Gitee) | — |
| `compute` | `auto` / `cpu` / `cuda` (local GEMM) | `auto` |
| `hf_token` | Hugging Face hub token (optional; `.com` gated files) | _(empty)_ |
| `modelscope_api_token` | ModelScope hub token (optional; `.cn` gated files) | _(empty)_ |

```bash
# Setup
aria-engine setup
aria-engine setup --status

# Download models
# Optional: aria-engine setup prompts hf_token (.com) or modelscope_api_token (.cn)
aria-engine download gemma-4-e2b-it_q4
aria-engine list
aria-engine check gemma-4-e2b-it_q4
# or: aria-engine check   # all cached models
aria-engine clean gemma-4-e2b-it_q4

# Upgrade CLI + libaria_ffi (latest stable, or a tag)
# FFI lands in ~/.ariacompute/lib/ — set ARIA_FFI_LIB if needed
aria-engine upgrade
aria-engine upgrade 0.7.2

# Serve (local only)
# or: serve /path/to/aria-bundle
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --compute auto

# Serve and register as a local provider on aria-router (process override; does not write engine.yml)
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --router http://127.0.0.1:8090 \
  --router-api-key sk-aria_… \
  --compute auto
```

`download` probes the regional public hub each run (`.com` → Hugging Face, `.cn` → ModelScope). Gated/private hub files return `auth failed HTTP 401` unless `aria-engine setup` has stored the matching token (`hf_token` on `.com`, `modelscope_api_token` on `.cn`) in `~/.ariacompute/engine.yml`.

`list` scans local `~/.ariacompute/models` only.

`check [model]` compares local file count, names, and SHA-256 against the regional hub (same source as `download`). Omit the model to check every cached bundle. Exit 1 on mismatch; `weight.bin` is hashed locally and compared to hub metadata (not re-downloaded).

`serve` flags override config for that process only (no rewrite). `serve <model>` uses a filesystem path if it exists, otherwise `~/.ariacompute/models/<model>`. `--router URL` / `--router-api-key SECRET` register this process as a local provider on aria-router (do not write `engine.yml`). Use the secret from router Dashboard → API keys when the gateway has `require_api_key: true`.

`--compute auto|cpu|cuda` selects **local** GEMM: `auto` uses CUDA when `libcudart`/`libcublas` load and `cudaGetDeviceCount>0`, otherwise CPU (AVX2+FMA on x86_64, NEON on aarch64). `--compute cuda` **fails** if no NVIDIA device — it does not silently fall back. CUDA is runtime-loaded (no toolkit at compile time); on H200 you can still pass `--features cuda` as a documentation flag:

```bash
cargo build -p aria-openai --release --features cuda
aria-engine serve qwen3-0.6b_q4 --compute auto --profile
```

`--profile` records load/generate timings. Read them with `GET /v1/engine/profile` or:

```bash
python scripts/profile_qwen3_serve.py --compute cpu --spawn --report ./out/engine_profile_qwen3.json
```

`--compute auto|cpu|cuda` selects **local** GEMM. Mixture-of-Models / edge-cloud routing lives in the **router** repo (`aria-router`).

## Register with aria-router

This process never routes. If `router` is set (config or `--router`), `serve` registers as a local provider before it accepts traffic:

`PUT {router}/v1/router/providers` with `{name, endpoint, provider_model_id, locality}` and optional `Authorization: Bearer` from `router_api_key`. Failure **exits** (no silent local-only fallback). `--router` / `--router-api-key` override `engine.yml` for this process only.

Use **different ports**: engine data (`--bind`) vs router management (`--mgmt-bind`, default `127.0.0.1:8080`). Clients then talk to the **router data plane**, not engine.

```bash
# 1. router repo — data :8899, management :8090
cd /path/to/router
cargo run -p aria-router -- serve \
  --config config/examples/semantic-tiny.yaml \
  --bind 127.0.0.1:8899 \
  --mgmt-bind 127.0.0.1:8090

# 2. engine repo — OpenAI on :8080, then PUT to management
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --router http://127.0.0.1:8090 \
  --router-api-key sk-aria_… \
  --compute auto

# Persist the URL / key instead of passing flags each time:
#   aria-engine setup  # optional router URL + router API key (from Dashboard)
#   # or in ~/.ariacompute/engine.yml:
#   # router: http://127.0.0.1:8090
#   # router_api_key: sk-aria_…

# 3. Chat via the gateway (concrete name = bypass; forwards to this engine)
curl -s http://127.0.0.1:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma-4-e2b-it_q4",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens": 32
  }' | jq .

# Semantic / agent entrypoints (aria/semantic-auto, aria/agent-auto) only
# reach this engine if the YAML modelRefs / default_model use the same
# registered name. Inspect the last hop:
curl -sD - -o /dev/null http://127.0.0.1:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gemma-4-e2b-it_q4","messages":[{"role":"user","content":"Hello"}]}' \
  | grep -i x-aria-router
```

`x-aria-router-layer` is `bypass` for a registered bundle name, or `semantic` / `agent` for recipe entrypoints.

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
    "temperature": 0
  }' | jq .

# Chat (SSE stream)
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"Hello"}],
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
| [`model/scripts/diag_qwen3_chat.py`](https://github.com/ariacompute/model/tree/main/scripts/diag_qwen3_chat.py) | **Quant + template**: HF fp32 vs `reconstruct_weight` inject into the same HF graph |
| [`scripts/diag_qwen3_chat.py`](scripts/diag_qwen3_chat.py) | **Engine graph**: this server vs the model JSON (`--peer-report`) |

**1. Model teacher** (sibling `model` repo; GPU recommended; tokenizer from `--hf`, not the bundle):

```bash
# from model
pip install torch transformers
python scripts/diag_qwen3_chat.py \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --hf Qwen/Qwen3-0.6B \
  --device cuda \
  --report ./out/model_diag_qwen3.json
```

`--device auto` picks CUDA when available. Attention is eager (no hub CUDA JIT). If compile still fails: `sudo apt install python3-dev`.

Read `chat.fp32` vs `chat.reconstruct`, plus `template_string_match` / `prompt_ids_match`. A reconstruct greeting such as `"Hello! How can I assist you today?"` with `exact_prefix_len >= 4` means the **bundle is usable in HF**; leftover engine garbage is not a quant bug.

**2. Engine** (serve the **same** bundle):

```bash
# from engine
./aria-engine serve qwen3-0.6b_q4 \
  --bind 127.0.0.1:8080 \
  --compute auto --profile
python scripts/diag_qwen3_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --peer-report /path/to/model/out/model_diag_qwen3.json \
  --report ./out/engine_diag_qwen3.json
```

`--timeout` defaults to 300s (CPU decode can be slow). Optional: `pip install tokenizers` so prompt ids are encoded from `bundle/tokenizer.json`.

**How to read `hints`**

| Observation | Likely cause |
|-------------|--------------|
| `template_string_match` / `prompt_ids_match` is false | ChatML / tokenizer encode drift |
| reconstruct greedy already diverges from fp32 (`QUANT:…`) | codebook / Hadamard / inject |
| reconstruct chat looks ok, engine `content` does not (`ENGINE_GRAPH`) | Rust forward (HDM, embed row gather, RoPE, QK-norm, …) |

Teacher for the engine is **HF + reconstruct inject**, not raw fp32.

## Gemma-4 chat diagnostic

Use this pair when `gemma-4-e2b-it_q4` `/v1/chat/completions` is garbage (engine.log Hello → `"uhnyaчь…"`) while Qwen3 on the same host is fine. Both sides share `gemma4_it` (`<bos><|turn>user…<turn|>\n<|turn>model\n`), default user `Hello`, greedy `max_tokens=32`. Official Hello prompt is **10** tokens.

| Script | What it isolates |
|--------|------------------|
| [`model/scripts/diag_gemma4_chat.py`](https://github.com/ariacompute/model/tree/main/scripts/diag_gemma4_chat.py) | **Quant + template**: HF fp32 vs `reconstruct_weight` inject into the same HF graph |
| [`scripts/diag_gemma4_chat.py`](scripts/diag_gemma4_chat.py) | **Engine graph**: this server vs the model JSON (`--peer-report`) |

**1. Model teacher** (sibling `model` repo; GPU recommended):

```bash
# from model
pip install torch transformers
python scripts/diag_gemma4_chat.py \
  --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \
  --hf google/gemma-4-E2B-it \
  --device cuda \
  --report ./out/model_diag_gemma4.json
```

**2. Engine** (serve the **same** bundle):

```bash
# from engine
./aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --compute auto --profile
python scripts/diag_gemma4_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \
  --peer-report /path/to/model/out/model_diag_gemma4.json \
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
  --backend sglang=http://127.0.0.1:30000 \
  --max-tokens 64 --warmup 1 --runs 3 \
  --report ./out/bench_report.json
# also writes ./out/bench_report.md
```

Override model ids with `--model-id family=path=id` or `backend:family=id`. Default aria model id is the family path; others use HF `base_model`.

## SDK Bindings

Native C ABI (`ariacompute-ffi` / `libaria_ffi`) plus thin wrappers under `bindings/`.

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

C header: [`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`, `aria_complete` / stream, `aria_embed`, `aria_transcribe`, tools JSON, `aria_model_destroy`, `aria_last_error`. New helpers for SDK auto-download: `aria_model_cache_dir(model)` (returns `~/.ariacompute/models/{model}`) and `aria_is_local_path(ref)` (1 = local path, 0 = model name, -1 = error).

### Auto-download by model name

Every binding now accepts **either** a local bundle path **or** an Aria model name. A value containing `/` (or already on disk) is treated as a local path and loaded directly; otherwise it is a model name. All language SDKs download from the regional public hub (same as `aria-engine download`: `.com` → Hugging Face, `.cn` → ModelScope) and do **not** call Dashboard. Gated hub files use the same fields as `aria-engine setup` via instance `setup` (empty construct → `setup` → `open`); this is in-memory only and **never** writes `engine.yml` (CLI `aria-engine setup` still does). Empty instance fields still fall back to reading `~/.ariacompute/engine.yml`. Dashboard `sk-`/`bfvk-` tokens are ignored. Env `HF_TOKEN` / `MODELSCOPE_API_TOKEN` are not used. Token is optional for public models. A valid cached bundle at `~/.ariacompute/models/{model}` is reused without re-downloading. Download failures raise a clear error — they never fail silently.

```bash
cargo test -p ariacompute-ffi -p ariacompute-engine
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

Language SDKs auto-install the matching archive into `~/.ariacompute/lib/` on first `Engine.open` / equivalent if the library is not already on `ARIA_FFI_LIB`, bundled in the package, or cached. `aria-engine download` only fetches the model bundle (the CLI binary does not dlopen FFI).

```bash
# Example: Linux x86_64 (manual unpack; SDKs do this automatically)
tar -xzf libaria_ffi_0.7.1_linux_x86_64.tar.gz
export ARIA_FFI_LIB="$PWD/libaria_ffi.so"
# optional: export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
```

Point bindings at a local Aria bundle (`weight.bin` + `config.json` + tokenizer), e.g. from `aria-engine download …` under `~/.ariacompute/models/`.

### Examples

**Python** (auto-installs `libaria_ffi`; `ARIA_FFI_LIB` optional):

```bash
pip install aria-engine
# optional override: export ARIA_FFI_LIB=/path/to/libaria_ffi.so
```

```python
from aria_engine import Engine

# Local bundle path:
with Engine("/path/to/aria-bundle") as eng:
    out = eng.complete(
        [{"role": "user", "content": "Hello"}],
        {"max_tokens": 32},
    )
    print(out["response"])

# Or by model name — auto-downloads from Hugging Face / ModelScope (not Dashboard):
with Engine("gemma-4-e2b-it_q4") as eng:
    print(eng.complete([{"role": "user", "content": "Hello"}], {"max_tokens": 32})["response"])
    # also: eng.embed("hi"), eng.transcribe(pcm_bytes)

# Gated hub files — instance setup (does not write ~/.ariacompute/engine.yml):
eng = Engine()
eng.setup(hf_token="hf_...")  # .com → Hugging Face
eng.open("gemma-4-e2b-it_q4")
print(eng.complete([{"role": "user", "content": "Hello"}], {"max_tokens": 32})["response"])
eng_ms = Engine()
eng_ms.setup(modelscope_api_token="ms_...", site_url="https://ariacompute.cn")
eng_ms.open("gemma-4-e2b-it_q4")
print(eng_ms.complete([{"role": "user", "content": "Hello"}], {"max_tokens": 32})["response"])
```

**TypeScript / Node** (`@ariacompute/engine-ts`):

```bash
npm install @ariacompute/engine-ts
# optional override: export ARIA_FFI_LIB=/path/to/libaria_ffi.so
```

```ts
import { Engine } from "@ariacompute/engine-ts";

// Local bundle path:
const eng = new Engine("/path/to/aria-bundle");
const out = eng.complete(
  [{ role: "user", content: "Hello" }],
  { max_tokens: 32 },
);
console.log(out.response);
eng.close();

// Or by model name — auto-downloads from Hugging Face / ModelScope (not Dashboard):
const eng2 = await Engine.open("gemma-4-e2b-it_q4");
console.log((await eng2.complete([{ role: "user", content: "Hello" }])).response);
eng2.close();

// Gated hub files — instance setup (does not write ~/.ariacompute/engine.yml):
const engHf = new Engine();
engHf.setup({ hf_token: "hf_..." }); // .com
await engHf.open("gemma-4-e2b-it_q4");
const engMs = new Engine();
engMs.setup({ modelscope_api_token: "ms_...", site_url: "https://ariacompute.cn" });
await engMs.open("gemma-4-e2b-it_q4");
engHf.close();
engMs.close();
```

**Go** (cgo; link against `libaria_ffi`):

```bash
export CGO_ENABLED=1
# optional: export ARIA_FFI_LIB=/path/to/libaria_ffi.so
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

	// Or by model name — auto-downloads from Hugging Face / ModelScope (not Dashboard):
	eng2, err := aria.OpenModel("gemma-4-e2b-it_q4", "", "")
	if err != nil {
		panic(err)
	}
	defer eng2.Close()
	_ = eng2

	// Gated hub files — instance setup (does not write ~/.ariacompute/engine.yml):
	engHf := aria.NewEngine()
	hf := "hf_..."
	if err := engHf.Setup(aria.SetupUpdates{HFToken: &hf}); err != nil {
		panic(err)
	}
	if err := engHf.Open("gemma-4-e2b-it_q4"); err != nil {
		panic(err)
	}
	defer engHf.Close()
	engMs := aria.NewEngine()
	ms, site := "ms_...", "https://ariacompute.cn"
	if err := engMs.Setup(aria.SetupUpdates{ModelScopeAPIToken: &ms, SiteURL: &site}); err != nil {
		panic(err)
	}
	if err := engMs.Open("gemma-4-e2b-it_q4"); err != nil {
		panic(err)
	}
	defer engMs.Close()
	_ = engHf
	_ = engMs
}
```

**Rust** (`ariacompute-engine` crate — native API; does not require unpacking `libaria_ffi`):

```bash
cargo add ariacompute-engine
```

```rust
use aria_engine::{SetupUpdates, Engine, GenerateOpts, OpenOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut eng = Engine::open("/path/to/aria-bundle")?;
    let g = eng.complete("Hello", &GenerateOpts {
        max_tokens: 32,
        temperature: 0.0,
    })?;
    println!("{}", g.text);

    // Or by model name — auto-downloads from Hugging Face / ModelScope (not Dashboard):
    let mut eng2 = Engine::open_model("gemma-4-e2b-it_q4", &OpenOptions::default())?;
    let g2 = eng2.complete("Hello", &GenerateOpts { max_tokens: 32, temperature: 0.0 })?;
    println!("{}", g2.text);

    // Gated hub files — instance setup (does not write ~/.ariacompute/engine.yml):
    let mut eng_hf = Engine::new();
    eng_hf.setup(&SetupUpdates { hf_token: Some("hf_...".into()), ..Default::default() })?;
    eng_hf.open_named("gemma-4-e2b-it_q4")?;
    println!("{}", eng_hf.complete("Hello", &GenerateOpts { max_tokens: 32, temperature: 0.0 })?.text);
    let mut eng_ms = Engine::new();
    eng_ms.setup(&SetupUpdates {
        modelscope_api_token: Some("ms_...".into()),
        site_url: Some("https://ariacompute.cn".into()),
        ..Default::default()
    })?;
    eng_ms.open_named("gemma-4-e2b-it_q4")?;
    println!("{}", eng_ms.complete("Hello", &GenerateOpts { max_tokens: 32, temperature: 0.0 })?.text);
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
