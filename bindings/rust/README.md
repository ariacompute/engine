# aria-sdk (Rust)

Rust binding for Aria Engine.

```rust
use aria_sdk::{Engine, GenerateOpts};

let mut eng = Engine::open("/path/to/aria-bundle")?;
let out = eng.complete("Hello", &GenerateOpts { max_tokens: 32, temperature: 0.0 })?;
println!("{}", out.text);
```

Publish: `cargo publish -p aria-sdk` (via `release.yml` on GitHub Release).
