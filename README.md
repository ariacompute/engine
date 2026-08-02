# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute inference engine (Rust): OpenAI-compatible API, Aria bundle inference, zero-copy graph, ARM NEON / scalar kernels, hybrid router.

## Docs

- [`AGENTS.md`](AGENTS.md) — agent entry / directory index  
- [`requirements.md`](requirements.md) — audited Spec  
- [`task.md`](task.md) — implementation checklist  

## Build / test

```bash
cargo test
cargo run -p aria-openai --bin aria-engine -- serve --model /path/to/aria-bundle
```

Weights: **only** `aria-quant-bundle` (`config.json` + `weight.bin`). No GGUF.
