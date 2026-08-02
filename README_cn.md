# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute 推理引擎：OpenAI 兼容 API、Aria bundle 推理、零拷贝计算图、ARM NEON / scalar 内核、hybrid 路由。

## 构建 / 测试

```bash
cargo test
cargo run -p aria-openai --bin aria-engine -- serve --model /path/to/aria-bundle
```

权重格式：`aria-quant-bundle`（`config.json` + `weight.bin`）。
