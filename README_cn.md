# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute 推理引擎：OpenAI 兼容 API、Aria bundle 推理、零拷贝计算图、ARM NEON / scalar 内核、Hybrid 路由。

## 构建

```bash
cargo test
cargo run -p aria-openai --bin aria-engine -- serve --model /path/to/aria-bundle
```

权重：`aria-quant-bundle`（`config.json` + `weight.bin`）。

## 评测

```bash
# 请先自行启动各推理服务，然后：
python -m unittest discover -s bench/tests -t .
python -m bench list-families
python -m bench run \
  --backend aria=http://127.0.0.1:8080 \
  --backend llamacpp=http://127.0.0.1:8081 \
  --backend ollama=http://127.0.0.1:11434 \
  --backend vllm=http://127.0.0.1:8000 \
  --max-tokens 64 --warmup 1 --runs 3 \
  --report ./out/bench_report.json
# 同时会写出 ./out/bench_report.md
```

可用 `--model-id family=path=id` 或 `backend:family=id` 覆盖模型 id。aria 默认使用家族 path；其余后端默认使用 HF `base_model`。
