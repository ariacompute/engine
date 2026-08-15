# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute 推理引擎：OpenAI 兼容 API、Aria bundle 推理、零拷贝计算图、ARM NEON / scalar 内核、Hybrid 路由。

## 构建 / 测试

```bash
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

## 配置 / 运行

凭证与 hybrid 偏好保存在 `~/.ariacompute/config.yml`（通过 `aria-engine auth` 写入）。

| 字段 | 含义 | 默认 |
|------|------|------|
| `cloud_api_key` | Hybrid Bearer 密钥 | _(空 → 云调用报错)_ |
| `cloud_url` | Gateway base URL（由 API key 所属区域自动探测） | — |
| `site_url` | 下载用站点（与 `cloud_url` 同区） | — |
| `hybrid_mode` | `cost` / `balance` / `intelligence` | `balance` |
| `hybrid_execution` | `hybrid` / `device` / `cloud` | `hybrid` |

```bash
# 认证
aria-engine auth
aria-engine auth --status

# 下载
aria-engine download gemma-4-e2b-it_q4
aria-engine list
aria-engine clean gemma-4-e2b-it_q4

# 服务
# 或：serve /path/to/aria-bundle
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-mode balance \
  --hybrid-execution hybrid
```

`download` 每次运行会探测 Serve（若已配置密钥）/ HuggingFace / ModelScope，并选择当前可达且最快的源。

`list` 查询 `{site_url}/api/dashboard/models`（需先 `aria-engine auth`），按可下载 bundle 列出并标记 `downloaded` / `not downloaded`（另附仅本地缓存项）。

`serve` 旗标仅覆盖本进程配置（不回写文件）。`serve <model>`：若为现存路径则用之，否则使用 `~/.ariacompute/models/<model>`。

在 `--hybrid-execution hybrid` 下，路由依据提示复杂度 / 上下文溢出 / modality / 本地失败 / `FORCE_CLOUD`。`cost` 更偏端侧；`intelligence` 更偏云端；`balance` 为中性自动。用户消息包含 `FORCE_CLOUD` 可强制走云端（测试 / 演示）。`--hybrid-execution device` 永不切换云端；`--hybrid-execution cloud` 始终云端推理（隐私敏感请求仍留本地）。云端 handoff 请求的 `model` 固定为 `ariacompute/ariamodel`。

## OpenAI API

假设服务在 `http://127.0.0.1:8080`：

```bash
# 列出模型
curl -s http://127.0.0.1:8080/v1/models | jq .

# Chat（非流式）
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens": 32,
    "temperature": 0
  }' | jq .

# Chat（SSE 流式）
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens": 32,
    "stream": true
  }'

# Embeddings
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":"hello embedding"}' | jq .

# ASR stub（PCM16 LE 字节，base64）
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -H 'content-type: application/json' \
  -d '{"file_b64":"AAECAwQFBgc="}' | jq .
```

## 评测（Bench）

```bash
# 请自行启动各后端服务，然后：
python -m unittest discover -s bench/tests -t .
python -m bench list-families
python -m bench run \
  --backend aria=http://127.0.0.1:8080 \
  --backend llamacpp=http://127.0.0.1:8081 \
  --backend ollama=http://127.0.0.1:11434 \
  --backend vllm=http://127.0.0.1:8000 \
  --max-tokens 64 --warmup 1 --runs 3 \
  --report ./out/bench_report.json
# 同时会写入 ./out/bench_report.md
```

可用 `--model-id family=path=id` 或 `backend:family=id` 覆盖模型 id。aria 默认使用家族 path；其余后端默认使用 HF `base_model`。

## 工程约定

本仓库遵循 Harness Engineering 理念：

- [`AGENTS.md`](AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](requirements.md)：需求 Spec（功能边界 / 例外 / 验收标准，需人工审核）
- [`task.md`](task.md)：实施任务清单
