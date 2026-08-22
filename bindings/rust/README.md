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

Publish: `cargo publish -p ariacompute-engine` (via [`.github/workflows/publish-cargo.yml`](../../.github/workflows/publish-cargo.yml) on GitHub Release).

## Auto-download by model name

`Engine::open_model(model_ref, &OpenOptions { token, site })` accepts either a
local bundle path or an Aria model name (e.g. `gemma-4-e2b-it_q4`). A value
containing `/` or already on disk is loaded directly; otherwise the SDK
downloads it from the Dashboard private source (requires `token`; `site`
defaults to `https://ariacompute.com`) into `~/.ariacompute/models/{model}` and
loads it. A valid cached bundle is reused without re-downloading.

```rust
use aria_engine::{Engine, OpenOptions};
let opts = OpenOptions { token: Some("API_TOKEN".into()), site: None };
let mut eng = Engine::open_model("gemma-4-e2b-it_q4", &opts)?;
```
