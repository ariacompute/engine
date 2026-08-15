# aria-engine (Rust)

Thin SDK over `aria-inference` (and re-exports of `aria-ffi` for embedding tests).

```rust
use aria_engine::{Engine, GenerateOpts};

let mut eng = Engine::open("/path/to/bundle")?;
let g = eng.complete("hi", &GenerateOpts { max_tokens: 16, temperature: 0.0 })?;
```

Publish: `cargo publish -p aria-engine` (via `release.yml` on GitHub Release).
