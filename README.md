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

`download` probes Serve (if keyed) / HuggingFace / ModelScope each run and picks the fastest reachable source. On a TTY it shows a green progress bar for the transfer.

`list` queries `{site_url}/api/dashboard/models` (requires `aria-engine auth`) and prints each downloadable bundle as `downloaded` / `not downloaded` (plus local-only caches).

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

## SDK Bindings

Native C ABI (`aria-ffi` / `libaria_ffi`) plus thin wrappers under `bindings/`.

| Binding | Path | Registry |
|---------|------|----------|
| Rust | `bindings/rust` (`aria-engine`) | crates.io |
| Python | `bindings/python` | PyPI |
| Go | `bindings/go` | Go module |
| TypeScript | `bindings/typescript` | npm `@ariacompute/engine-ts` |
| React Native | `bindings/react-native` | npm `@ariacompute/engine-rn` |
| Flutter | `bindings/flutter` | pub.dev |
| Swift | `bindings/swift` | CocoaPods |
| Kotlin | `bindings/kotlin` | Maven |

C header: [`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`, `aria_complete` / stream, `aria_embed`, `aria_transcribe`, tools JSON, `aria_model_destroy`, `aria_last_error`.

```bash
cargo test -p aria-ffi -p aria-engine
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

**Rust** (`aria-engine` crate — native API; does not require unpacking `libaria_ffi`):

```bash
cargo add aria-engine
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

Cut a **GitHub Release** — [`.github/workflows/release.yml`](.github/workflows/release.yml) builds CLI + `libaria_ffi` archives and attempts package publish (npm / pub.dev / Maven / CocoaPods / crates.io / PyPI). **Publish failures are fail-pass** and do not block CLI/`libaria_ffi` assets. Secrets: `NPM_TOKEN`, pub credentials, Maven + GPG, `COCOAPODS_TRUNK_TOKEN`, `CARGO_REGISTRY_TOKEN`, `PYPI_TOKEN`, plus `ARIACOMPUTE_TOKEN` for Release uploads.

Version = release tag without leading `v`.

## Engineering Conventions

This repository follows the Harness Engineering philosophy:

- [`AGENTS.md`](AGENTS.md): Agent engineering context entry and directory index
- [`requirements.md`](requirements.md): Requirements spec (feature boundaries/exceptions/acceptance criteria, human-review-gated)
- [`task.md`](task.md): Implementation task checklist
