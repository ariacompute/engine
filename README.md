# Aria Engine

Pure Rust inference runtime for **AFM**:

| Track | Crate | Model |
| --- | --- | --- |
| Encoder (`afm_de`) | `ariacompute-de` | Laya-layout ModernBERT DecisionModel |
| Decoder (`afm_dd`) | `ariacompute-dd` | MiniCPM5-2B ± PEFT LoRA (SemIf direct) |

Surfaces: **`aria-engine` CLI** · **`POST /v1/systemone`** · **`libaria_ffi` C ABI** · language SDKs under `bindings/`.

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo build --release -p aria-cli -p ariacompute-ffi
```

Optional CUDA feature: `cargo build -p aria-cli --features cuda` (propagates to de/dd).

## Config (`aria-engine setup`)

Writes `~/.ariacompute/engine.yml` (no router fields):

| Field | Meaning | Default |
| --- | --- | --- |
| `site_url` | Site (`.com` / `.cn`) → hub region | locale (`LANG=zh*` → `.cn`) |
| `upgrade_url` | Releases org root (GitHub / Gitee) | from `site_url` |
| `compute` | `auto` \| `cpu` \| `cuda` | `auto` |
| `hf_token` | Hugging Face token (`.com`) | _(empty)_ |
| `modelscope_api_token` | ModelScope token (`.cn`) | _(empty)_ |

```bash
aria-engine setup                 # interactive
aria-engine setup --status        # secrets redacted
aria-engine setup --clear
aria-engine setup --site-url https://ariacompute.cn --compute cuda
```

Regional hub token prompt: `.cn` → ModelScope only; otherwise Hugging Face only. Interactive token input is masked with `*`.

## CLI

```bash
aria-engine download afm-de       # HF/ModelScope HTTP API (site_url + setup token); no hub CLI
                                  # .cn → ModelScope /api/v1/.../repo?FilePath=… ; .com → HF resolve
aria-engine list | check | clean  # list = local ∪ Hub org (ariacompute / AriaCompute); status downloaded|not downloaded
                                  # check/clean: complete checkpoints only; failed downloads cleaned up
aria-engine upgrade [version]     # needs upgrade_url
aria-engine serve …               # see Serve below
aria-engine decide …              # see Decide below
aria-engine -v
```

## Serve (`aria-engine serve`)

Start a local System One HTTP server over a downloaded AFM checkpoint. No router registration.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--track` | `encoder` | `encoder` \| `decoder` |
| `--model-name` | `afm-de` / `afm-dd` (from `--track`) | Reported model id; also used as checkpoint when `--checkpoint` omitted |
| `--checkpoint` | `~/.ariacompute/models/<model-name>` | Checkpoint directory, or cache name under `~/.ariacompute/models/` |
| `--bind` | `127.0.0.1:8010` (encoder) / `127.0.0.1:8011` (decoder) | Listen address |
| `--compute` | from `engine.yml` (`auto`) | `auto` \| `cpu` \| `cuda` (overrides setup; logged for device selection) |

Provide either `--model-name` or `--checkpoint` (or both). Soft layout check rejects an encoder track pointed at a decoder-looking dir and vice versa.

### Endpoints

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/health` or `/` | `{ "ok": true, "service": "afm-engine", "model": "<model-name>" }` |
| `GET` | `/v1/models` | OpenAI-style list with one entry for `--model-name` |
| `POST` | `/v1/systemone` | `{ "answers": { … }, "model": "<model-name>" }` |

`POST /v1/systemone` body matches Decide’s System One shape: `{ "state": …, "questions": { "<qid>": { "type", "instructions", "criteria" } } }` (`type` = `choice` \| `score` \| `noul`). Invalid questions → HTTP **422**; scoring errors → **500**.

### Examples

```bash
# Encoder (default track) from cache name
aria-engine download afm-de
aria-engine serve --model-name afm-de
# → http://127.0.0.1:8010

# Explicit track + checkpoint path
aria-engine serve --track decoder --checkpoint ~/.ariacompute/models/afm-dd

# Custom bind + compute override
aria-engine serve --model-name afm-de --bind 0.0.0.0:8010 --compute cpu

# Health / models
curl -s http://127.0.0.1:8010/health
curl -s http://127.0.0.1:8010/v1/models

# System One — choice
curl -s http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "user wants a refund",
    "questions": {
      "q1": {
        "type": "choice",
        "instructions": "Pick the best action",
        "criteria": {
          "refund": "issue a full refund",
          "deny": "deny the request"
        }
      }
    }
  }'

# System One — score (criteria: 2–10 ordered levels)
curl -s http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "ticket sat in queue for three days with no reply",
    "questions": {
      "q1": {
        "type": "score",
        "instructions": "Rate urgency from 0 (low) to 3 (critical)",
        "criteria": [
          "can wait until next week",
          "should be handled this week",
          "needs attention today",
          "page on-call now"
        ]
      }
    }
  }'

# System One — noul (true / false)
curl -s http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "customer wrote: this is unacceptable, I want a manager",
    "questions": {
      "q1": {
        "type": "noul",
        "instructions": "Is the user angry?",
        "criteria": {
          "false": "calm or neutral tone",
          "true": "angry or escalating"
        }
      }
    }
  }'
```

Real weight forward requires a complete local checkpoint (see Stage B in [`task.md`](task.md)). Until then, use `decide` with `--logits` / `--semif-out` for golden-path scoring.

## Decide (`aria-engine decide`)

One-shot typed decision over a JSON body (stdin or `--file`). Prints one pretty JSON answer per question/record. Same packing / typed-answer contract as `POST /v1/systemone`.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--track` | `encoder` | `encoder` \| `decoder` |
| `--model-name` | `afm-de` / `afm-dd` (from `--track`) | Cache name; used as checkpoint when `--checkpoint` omitted |
| `--checkpoint` | `~/.ariacompute/models/<model-name>` | Checkpoint directory, or cache name under `~/.ariacompute/models/` |
| `--file` | _(stdin)_ | Path to input JSON |
| `--logits` | — | Encoder **golden path**: JSON array of raw option logits (skips weight forward) |
| `--semif-out` | — | Decoder **golden path**: SemIf-style JSON (skips weight forward) |

Without `--logits` / `--semif-out`, the scorer runs a real forward on the checkpoint (needs a complete local model; see Stage B in [`task.md`](task.md)).

### Input shapes

**1. AFM-D record** (single question) — `task` + `options` + `state`:

```json
{
  "id": "choice-1",
  "task": "choice",
  "instructions": "Select the fruit",
  "options": [
    {"name": "apple", "description": "a red fruit"},
    {"name": "carrot", "description": "an orange vegetable"}
  ],
  "state": "The user wants something sweet and red."
}
```

```json
{
  "id": "score-1",
  "task": "score",
  "instructions": "Rate urgency from 0 (low) to 3 (critical)",
  "options": [
    {"name": "level-0", "description": "can wait until next week"},
    {"name": "level-1", "description": "should be handled this week"},
    {"name": "level-2", "description": "needs attention today"},
    {"name": "level-3", "description": "page on-call now"}
  ],
  "state": "ticket sat in queue for three days with no reply"
}
```

```json
{
  "id": "noul-1",
  "task": "noul",
  "instructions": "Is the user angry?",
  "options": [
    {"name": "false", "description": "calm or neutral tone"},
    {"name": "true", "description": "angry or escalating"}
  ],
  "state": "customer wrote: this is unacceptable, I want a manager"
}
```

`task` is `choice` \| `score` \| `noul`. For `noul`, option names normalize to `true` / `false` (aliases `yes` / `no` accepted). Repo fixtures: [`tests/fixtures/encoder_choice.json`](tests/fixtures/encoder_choice.json), [`encoder_noul.json`](tests/fixtures/encoder_noul.json), [`decoder_choice.json`](tests/fixtures/decoder_choice.json).

**2. System One body** (one or more questions) — same as HTTP serve:

```json
{
  "state": "user wants a refund",
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "Pick the best action",
      "criteria": {
        "refund": "issue a full refund",
        "deny": "deny the request"
      }
    },
    "q2": {
      "type": "score",
      "instructions": "Rate urgency from 0 (low) to 3 (critical)",
      "criteria": [
        "can wait until next week",
        "should be handled this week",
        "needs attention today",
        "page on-call now"
      ]
    },
    "q3": {
      "type": "noul",
      "instructions": "Is the user angry?",
      "criteria": {
        "false": "calm",
        "true": "angry"
      }
    }
  }
}
```

| `type` | `criteria` |
| --- | --- |
| `choice` | object: option name → description |
| `score` | array of 2–10 ordered level strings |
| `noul` | object with `true` / `false` (optional; defaults apply) |

### Examples

```bash
# Encoder golden path (no weight forward) — logits length = option count
aria-engine decide \
  --track encoder \
  --model-name afm-de \
  --logits '[2.5, 0.1]' \
  --file tests/fixtures/encoder_choice.json

# Same via stdin; --checkpoint overrides model-name path
aria-engine decide --track encoder --checkpoint afm-de --logits '[2.5,0.1]' \
  < tests/fixtures/encoder_choice.json

# Decoder golden path — SemIf out JSON
aria-engine decide \
  --track decoder \
  --model-name afm-dd \
  --semif-out '{"option_ids":["a","b"],"probabilities":[0.1,0.9]}' \
  --file tests/fixtures/decoder_choice.json

# System One — choice (encoder golden path; two options → two logits)
cat <<'EOF' | aria-engine decide --track encoder --model-name afm-de --logits '[1.0, 0.2]'
{
  "state": "user wants a refund",
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "Pick the best action",
      "criteria": {"refund": "issue a full refund", "deny": "deny the request"}
    }
  }
}
EOF

# System One — score (four levels → four logits)
cat <<'EOF' | aria-engine decide --track encoder --model-name afm-de --logits '[0.1, 0.4, 2.0, 0.3]'
{
  "state": "ticket sat in queue for three days with no reply",
  "questions": {
    "q1": {
      "type": "score",
      "instructions": "Rate urgency from 0 (low) to 3 (critical)",
      "criteria": [
        "can wait until next week",
        "should be handled this week",
        "needs attention today",
        "page on-call now"
      ]
    }
  }
}
EOF

# System One — noul (false/true → two logits; fixture also works)
aria-engine decide \
  --track encoder \
  --model-name afm-de \
  --logits '[0.2, 1.8]' \
  --file tests/fixtures/encoder_noul.json

cat <<'EOF' | aria-engine decide --track encoder --model-name afm-de --logits '[0.2, 1.8]'
{
  "state": "customer wrote: this is unacceptable, I want a manager",
  "questions": {
    "q1": {
      "type": "noul",
      "instructions": "Is the user angry?",
      "criteria": {"false": "calm or neutral tone", "true": "angry or escalating"}
    }
  }
}
EOF

# Real forward (needs complete downloaded checkpoint)
aria-engine decide --track encoder --model-name afm-de --file record.json
```

`--model-name afm-de` (or `--checkpoint afm-de`) resolves to `~/.ariacompute/models/afm-de` when that path exists (same rules as `serve`).

### Output

One pretty-printed typed answer object per record (e.g. `type`, `choice` / `score` / `noul`, `probabilities`, `confidence`). For System One multi-question input, answers are printed sequentially (one JSON document per question).

## FFI

Header: [`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`, `aria_systemone`, `aria_model_destroy`.

Release assets: `aria-engine_${VERSION}_${OS}.tar.gz|zip`, `libaria-engine_ffi_${VERSION}_${OS}.tar.gz`.

## Language SDKs

Native C ABI (`ariacompute-ffi` / `libaria-engine_ffi`) plus thin wrappers under `bindings/`. API surface is **System One** (`systemone`), not chat/complete.

| Binding | Path | Registry |
| --- | --- | --- |
| Rust | `bindings/rust` (`ariacompute-engine`) | crates.io |
| Python | `bindings/python` | PyPI `aria-engine` |
| Go | `bindings/go` | Go module |
| TypeScript | `bindings/typescript` | npm `@ariacompute/engine-ts` |
| React Native | `bindings/react-native` | npm `@ariacompute/engine-rn` |
| Flutter | `bindings/flutter` | pub.dev `aria_engine` |
| Swift | `bindings/swift` | CocoaPods `AriaEngine` |
| Kotlin | `bindings/kotlin` | Maven Central `com.ariacompute:engine` |

C header: [`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`, `aria_systemone`, `aria_model_destroy`, `aria_last_error`.

FFI-backed SDKs need `libaria-engine_ffi` (via `aria-engine upgrade`, package bundle, or `ARIA_FFI_LIB`). Point `checkpoint` at a downloaded AFM dir (e.g. `~/.ariacompute/models/afm-de` after `aria-engine download afm-de`). Rust crate links crates natively and does not require unpacking the shared library.

```bash
cargo test -p ariacompute-ffi -p ariacompute-engine
./scripts/run-binding-tests.sh   # host matrix (Rust / Python / Go / TS)
```

### Examples

**Python** (`aria-engine`; `ARIA_FFI_LIB` optional):

```bash
pip install aria-engine
# optional: export ARIA_FFI_LIB=/path/to/libaria-engine_ffi.so
```

```python
from aria_engine import AriaEngine

eng = AriaEngine("/path/to/afm-de", "encoder")  # or ~/.ariacompute/models/afm-de
out = eng.systemone({
    "state": "user wants a refund",
    "questions": {
        "q1": {
            "type": "choice",
            "instructions": "Pick the best action",
            "criteria": {
                "refund": "issue a full refund",
                "deny": "deny the request",
            },
        }
    },
})
print(out["answers"]["q1"])
eng.destroy()
```

**TypeScript / Node** (`@ariacompute/engine-ts`):

```bash
npm install @ariacompute/engine-ts
# optional: export ARIA_FFI_LIB=/path/to/libaria-engine_ffi.so
```

```ts
import { AriaEngine } from "@ariacompute/engine-ts";

const eng = new AriaEngine("/path/to/afm-de", "encoder");
const out = eng.systemone({
  state: "user wants a refund",
  questions: {
    q1: {
      type: "choice",
      instructions: "Pick the best action",
      criteria: {
        refund: "issue a full refund",
        deny: "deny the request",
      },
    },
  },
}) as { answers: Record<string, unknown> };
console.log(out.answers.q1);
eng.destroy();
```

**React Native** (`@ariacompute/engine-rn`):

```bash
npm install @ariacompute/engine-rn
```

```js
const { AriaEngine } = require("@ariacompute/engine-rn");

const eng = new AriaEngine("/path/to/afm-de", "encoder");
const out = await eng.systemone({
  state: "user wants a refund",
  questions: {
    q1: {
      type: "choice",
      instructions: "Pick the best action",
      criteria: {
        refund: "issue a full refund",
        deny: "deny the request",
      },
    },
  },
});
console.log(out.answers.q1);
eng.destroy();
```

**Go** (cgo; link `libaria_ffi` / `libaria-engine_ffi`):

```bash
export CGO_ENABLED=1
# optional: export ARIA_FFI_LIB=/path/to/libaria-engine_ffi.so
go get github.com/ariacompute/engine/bindings/go@latest
```

```go
package main

import (
	"fmt"

	aria "github.com/ariacompute/engine/bindings/go"
)

func main() {
	eng, err := aria.Open("/path/to/afm-de", "encoder")
	if err != nil {
		panic(err)
	}
	defer eng.Destroy()
	out, err := eng.SystemOne(map[string]any{
		"state": "user wants a refund",
		"questions": map[string]any{
			"q1": map[string]any{
				"type":         "choice",
				"instructions": "Pick the best action",
				"criteria": map[string]any{
					"refund": "issue a full refund",
					"deny":   "deny the request",
				},
			},
		},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(out["answers"])
}
```

**Rust** (`ariacompute-engine` — native; no `libaria-engine_ffi` unpack required):

```bash
cargo add ariacompute-engine
```

```rust
use aria_engine::{Engine, EngineTrack, SystemOneRequest};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eng = Engine::open("/path/to/afm-de", EngineTrack::Encoder)?;
    let req: SystemOneRequest = serde_json::from_value(json!({
        "state": "user wants a refund",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Pick the best action",
                "criteria": {
                    "refund": "issue a full refund",
                    "deny": "deny the request"
                }
            }
        }
    }))?;
    let out = eng.systemone(&req)?;
    println!("{out}");
    Ok(())
}
```

**Flutter** (`package:aria_engine`):

```yaml
# pubspec.yaml
dependencies:
  aria_engine: ^0.1.0
```

```dart
import 'package:aria_engine/aria_engine.dart';

final eng = AriaEngine('/path/to/afm-de', track: 'encoder');
final out = eng.systemone({
  'state': 'user wants a refund',
  'questions': {
    'q1': {
      'type': 'choice',
      'instructions': 'Pick the best action',
      'criteria': {
        'refund': 'issue a full refund',
        'deny': 'deny the request',
      },
    },
  },
});
print(out['answers']);
eng.destroy();
```

**Swift** (CocoaPods `AriaEngine`):

```ruby
# Podfile
pod 'AriaEngine'
```

```swift
import AriaEngine

let eng = AriaEngine(checkpoint: "/path/to/afm-de", track: "encoder")
let req = """
{
  "state": "user wants a refund",
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "Pick the best action",
      "criteria": {
        "refund": "issue a full refund",
        "deny": "deny the request"
      }
    }
  }
}
"""
let out = try eng.systemone(requestJson: req)
print(out)
eng.destroy()
```

**Kotlin** (`com.ariacompute:engine`):

```kotlin
// build.gradle.kts
implementation("com.ariacompute:engine:0.1.0")
```

```kotlin
import com.ariacompute.engine.AriaEngine
import org.json.JSONObject

val eng = AriaEngine("/path/to/afm-de", "encoder")
val out = eng.systemone(
    JSONObject(
        """
        {
          "state": "user wants a refund",
          "questions": {
            "q1": {
              "type": "choice",
              "instructions": "Pick the best action",
              "criteria": {
                "refund": "issue a full refund",
                "deny": "deny the request"
              }
            }
          }
        }
        """.trimIndent()
    )
)
println(out.getJSONObject("answers"))
eng.destroy()
```

More detail per language: `bindings/*/README.md`.

## Publish workflows

On GitHub Release: `release.yml` (CLI+FFI assets) plus `publish-cargo.yml`, `publish-npm.yml`, `publish-pub.yml`, `publish-pypi.yml`, `publish-maven.yml`, `publish-cocoapods.yml`.

## Docker / Docker Compose deployment

Run the System One HTTP server as containers. The repo ships a parameterized
`Dockerfile`, a `docker/entrypoint.sh` orchestrator, a CPU `docker-compose.yml`
(two services: `engine-encoder` on **8010** and `engine-decoder` on **8011**), and
a `docker-compose.cuda.yml` GPU override.

### Build & run (CPU)

```bash
cp .env.example .env          # set SITE_URL / HF_TOKEN or MODELSCOPE_API_TOKEN / COMPUTE
docker compose build
docker compose up -d
```

Both services share the `aria-data` named volume at `/data/aria`, which holds
`engine.yml` (config) and `models/` (checkpoints). The entrypoint writes
`engine.yml` from env vars **only when absent**, so a pre-mounted config is
respected.

### Models

- **Offline (default):** pre-populate the volume with downloaded weights
  (`aria-engine download afm-de && aria-engine download afm-dd` on the host, or
  mount a host dir that contains `~/.ariacompute/models/*`).
- **Auto-download:** set `AUTO_DOWNLOAD=1` in `.env`; the entrypoint runs
  `aria-engine download <model>` on first run (needs network + the matching hub
  token). This is off by default.

### GPU (CUDA)

Requires the NVIDIA driver + `nvidia-container-toolkit`. The override swaps the
build base to a CUDA image, compiles with `--features cuda`, and requests a GPU
per service:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml build
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```

Set `COMPUTE=cuda` (done by the override) so candle selects the CUDA backend.

### Configuration (`.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENGINE_IMAGE` | `aria-engine:latest` | Tag for the built image |
| `BUILD_IMAGE` / `RUNTIME_IMAGE` | `rust:1-slim` / *(defaults to `BUILD_IMAGE`)* | CPU build base; the runtime reuses it so glibc always matches. Only the CUDA override sets `RUNTIME_IMAGE` |
| `FEATURES` | _(empty)_ | Build features; the CUDA override sets `cuda` |
| `SITE_URL` / `UPGRADE_URL` | _(empty)_ | Region hub (`.com`→HF, `.cn`→ModelScope) |
| `HF_TOKEN` / `MODELSCOPE_API_TOKEN` | _(empty)_ | Hub token for the active region |
| `COMPUTE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `ENCODER_MODEL` / `ENCODER_PORT` | `afm-de` / `8010` | encoder service model + port |
| `DECODER_MODEL` / `DECODER_PORT` | `afm-dd` / `8011` | decoder service model + port |
| `AUTO_DOWNLOAD` | `0` | `1` to fetch weights on first run |
| `RUST_LOG` | `info` | Rust tracing filter |

### Verify

```bash
curl -s http://localhost:8010/health     # {"ok":true,"service":"afm-engine","model":"afm-de"}
curl -s http://localhost:8011/health     # decoder
docker compose ps                        # both services healthy
```

The container binds `0.0.0.0` internally; `serve` keeps its `127.0.0.1` default
outside containers. **Do not commit `.env`** — it may contain hub tokens.

## License

MIT.

## Engineering Conventions

This repository follows the Harness Engineering philosophy:

- [`AGENTS.md`](AGENTS.md): Agent engineering context entry and directory index
- [`requirements.md`](requirements.md): Requirements spec (feature boundaries/exceptions/acceptance criteria, human-review-gated)
- [`task.md`](task.md): Implementation task checklist
