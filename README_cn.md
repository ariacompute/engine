# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute 推理引擎（Rust）：OpenAI 兼容 API、Aria bundle 推理、零拷贝计算图、ARM NEON / scalar 内核、hybrid 路由。

## 文档

- [`AGENTS.md`](AGENTS.md) — Agent 工程入口 / 目录索引  
- [`requirements.md`](requirements.md) — 已审核 Spec  
- [`task.md`](task.md) — 实施清单  

## 构建 / 测试

```bash
cargo test
cargo run -p aria-openai --bin aria-engine -- serve --model /path/to/aria-bundle
```

权重格式：**仅** `aria-quant-bundle`（`config.json` + `weight.bin`）。不支持 GGUF。
