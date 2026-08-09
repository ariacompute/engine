# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute 推理引擎：OpenAI 兼容 API、Aria bundle 推理、零拷贝计算图、ARM NEON / scalar 内核、Hybrid 路由。

## 构建 / 测试

```bash
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

## HTTP 服务

```bash
cargo run -p aria-openai --bin aria-engine -- serve \
  --model /path/to/aria-bundle \
  --bind 127.0.0.1:8080
```

默认监听 `127.0.0.1:8080`，对外提供 OpenAI 兼容 HTTP。

## Hybrid 云端

通过环境变量配置（无 CLI 参数）：

| 变量 | 含义 | 默认 |
|------|------|------|
| `ARIA_HYBRID_CLOUD_URL` | 云端 OpenAI 兼容 **base URL**（引擎会追加 `/v1/chat/completions`） | `http://127.0.0.1:9` |
| `ARIA_HYBRID_CLOUD_API_KEY` | Bearer Token；真实云调用必填 | _(空 → 云调用报错)_ |
| `ARIA_HYBRID_THRESHOLD` | 置信度阈值，`[0,1]` | `0.0` |
| `ARIA_HYBRID_MODE` | `cost` / `balance` / `intelligence` | `balance` |
| `ARIA_ON_DEVICE_ONLY` | `1` = 禁止卸载到云端 | 未设置 |

```bash
export ARIA_HYBRID_CLOUD_URL=https://api.openai.com
export ARIA_HYBRID_CLOUD_API_KEY=sk-...
export ARIA_HYBRID_THRESHOLD=0.5
export ARIA_HYBRID_MODE=balance

cargo run -p aria-openai --bin aria-engine -- serve \
  --model /path/to/aria-bundle \
  --bind 127.0.0.1:8080
```

用户消息中包含 `FORCE_CLOUD` 可强制走云端（测试 / 演示）。设置 `ARIA_ON_DEVICE_ONLY=1` 时始终本地。

## OpenAI API

假设服务在 `http://127.0.0.1:8080`：

```bash
# 列出模型
curl -s http://127.0.0.1:8080/v1/models | jq .

# Chat（非流式）
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"你好"}],
    "max_tokens": 32,
    "temperature": 0
  }' | jq .

# Chat（SSE 流式）
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"你好"}],
    "max_tokens": 32,
    "stream": true
  }'

# Embeddings
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":"hello embedding"}' | jq .

# ASR stub（PCM16 LE 字节的 base64）
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -H 'content-type: application/json' \
  -d '{"file_b64":"AAECAwQFBgc="}' | jq .
```

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

## 工程规范

本仓库遵循 Harness Engineering 理念：

- [`AGENTS.md`](AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](requirements.md)：需求规格（功能边界/异常/验收标准，人工审核制）
- [`task.md`](task.md)：实施任务清单
