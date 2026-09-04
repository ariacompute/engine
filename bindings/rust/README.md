# ariacompute-engine (Rust)

Thin SDK over `ariacompute-inference` (and re-exports of `ariacompute-ffi` for embedding tests).
crates.io package name is `ariacompute-engine`; the Rust crate is still `aria_engine`.

```bash
cargo add ariacompute-engine
```

```rust
use aria_engine::{Engine, GenerateOpts};

let mut eng = Engine::open("/path/to/bundle")?;
let g = eng.complete("hi", &GenerateOpts { max_tokens: 16, temperature: 0.0 })?;
```

`Engine::open_model` also installs `libaria_ffi` into `~/.ariacompute/lib/` when
missing (same Releases asset as `aria-engine upgrade`), so other language
bindings can reuse it. Native `Engine::open` does not dlopen that library.

## Auto-download by model name

`Engine::open_model(model_ref, &OpenOptions { token, site })` accepts either a
local bundle path or an Aria model name (e.g. `gemma-4-e2b-it_q4`). A value
containing `/` or already on disk is loaded directly; otherwise the SDK
downloads it from the regional public hub (same as `aria-engine download`:
`.com` → Hugging Face, `.cn` → ModelScope; `site` defaults to
`https://ariacompute.com`) into `~/.ariacompute/models/{model}` and loads it.
Dashboard is not used. A Dashboard `sk-` / `sk-bf-` token is ignored for hub auth.
Token is optional for public models. A valid cached bundle is reused without
re-downloading.

Gated files: `Engine::new` → `setup` → `open_named` (instance memory; does not
write `engine.yml`). Same keys as `aria-engine setup`. If omitted, the SDK reads
`~/.ariacompute/engine.yml`. Associated `open_model(..., &OpenOptions)` remains.

```rust
use aria_engine::{SetupUpdates, Engine, OpenOptions};
let mut eng = Engine::open_model("gemma-4-e2b-it_q4", &OpenOptions::default())?;
let mut gated = Engine::new();
gated.setup(&SetupUpdates { hf_token: Some("hf_...".into()), ..Default::default() })?;
gated.open_named("gemma-4-e2b-it_q4")?;
```
