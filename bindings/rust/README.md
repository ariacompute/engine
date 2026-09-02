# ariacompute-ariaengine (Rust)

Thin SDK over `ariaengine-inference` (and re-exports of `ariacompute-ariaengine-ffi` for embedding tests).
crates.io package name is `ariacompute-ariaengine`; the Rust crate is `ariaengine`.

```bash
cargo add ariacompute-ariaengine
```

```rust
use ariaengine::{Engine, GenerateOpts};

let mut eng = Engine::open("/path/to/bundle")?;
let g = eng.complete("hi", &GenerateOpts { max_tokens: 16, temperature: 0.0 })?;
```

`Engine::open_model` also installs `libariaengine_ffi` into `~/.ariacompute/lib/` when
missing (same Releases asset as `ariaengine upgrade`), so other language
bindings can reuse it. Native `Engine::open` does not dlopen that library.

## Auto-download by model name

`Engine::open_model(model_ref, &OpenOptions { token, site })` accepts either a
local bundle path or an Aria model name (e.g. `gemma-4-e2b-it_q4`). A value
containing `/` or already on disk is loaded directly; otherwise the SDK
downloads it from the regional public hub (same as `ariaengine download`:
`.com` → Hugging Face, `.cn` → ModelScope; `site` defaults to
`https://ariacompute.com`) into `~/.ariacompute/models/{model}` and loads it.
Dashboard is not used. A Dashboard `sk-` / `bfvk-` token is ignored for hub auth.
Token is optional for public models. A valid cached bundle is reused without
re-downloading.

Gated files: `Engine::new` → `setup` → `open_named` (instance memory; does not
write `engine.yml`). Same keys as `ariaengine setup`. If omitted, the SDK reads
`~/.ariacompute/engine.yml`. Associated `open_model(..., &OpenOptions)` remains.

```rust
use ariaengine::{SetupUpdates, Engine, OpenOptions};
let mut eng = Engine::open_model("gemma-4-e2b-it_q4", &OpenOptions::default())?;
let mut gated = Engine::new();
gated.setup(&SetupUpdates { hf_token: Some("hf_...".into()), ..Default::default() })?;
gated.open_named("gemma-4-e2b-it_q4")?;
```
