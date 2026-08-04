# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute inference engine: OpenAI-compatible API, Aria bundle inference, zero-copy graph, ARM NEON / scalar kernels, hybrid router.

## Build

```bash
cargo test
cargo run -p aria-openai --bin aria-engine -- serve --model /path/to/aria-bundle
```

Weights: `aria-quant-bundle` (`config.json` + `weight.bin`).

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

## Engineering Conventions

This repository follows the Harness Engineering philosophy:

- [`AGENTS.md`](AGENTS.md): Agent engineering context entry and directory index
- [`requirements.md`](requirements.md): Requirements spec (feature boundaries/exceptions/acceptance criteria, human-review-gated)
- [`task.md`](task.md): Implementation task checklist
