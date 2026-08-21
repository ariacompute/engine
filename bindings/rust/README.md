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
