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
| `upgrade_url` | CLI/FFI 升级组织根（`.com`→GitHub，`.cn`→Gitee） | — |
| `hybrid_mode` | `cost` / `balance` / `intelligence` | `balance` |
| `hybrid_execution` | `hybrid` / `device` / `cloud` | `hybrid` |
| `compute` | `auto` / `cpu` / `cuda`（本机 GEMM；**不是** hybrid 开关） | `auto` |

```bash
# 认证
aria-engine auth
aria-engine auth --status

# 下载模型
aria-engine download gemma-4-e2b-it_q4
aria-engine list
aria-engine clean gemma-4-e2b-it_q4

# 升级 CLI + libaria_ffi（最新正式版，或指定版本）
# FFI 安装到 ~/.ariacompute/lib/，必要时设置 ARIA_FFI_LIB
aria-engine upgrade
aria-engine upgrade 0.7.2

# 服务
# 或：serve /path/to/aria-bundle
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-mode balance \
  --hybrid-execution hybrid \
  --compute auto
```

`download` 每次运行会探测 Serve（若已配置密钥）/ HuggingFace / ModelScope，并选择当前可达且最快的源。在 TTY 下会显示绿色下载进度条。

`list` 查询 `{site_url}/api/dashboard/models`（需先 `aria-engine auth`），按可下载 bundle 列出并标记 `downloaded` / `not downloaded`（另附仅本地缓存项）。

`serve` 旗标仅覆盖本进程配置（不回写文件）。`serve <model>`：若为现存路径则用之，否则使用 `~/.ariacompute/models/<model>`。

`--hybrid-execution` 只控制云 handoff（`device` 永不离机）。`--compute auto|cpu|cuda` 选择**本机** GEMM：`auto` 在能加载 `libcudart`/`libcublas` 且 `cudaGetDeviceCount>0` 时用 CUDA，否则 CPU（x86_64 AVX2+FMA，aarch64 NEON）。`--compute cuda` 在无 NVIDIA 设备时**硬失败**，不会静默降到 CPU。CUDA 为运行时 libloading（编译不依赖 CUDA toolkit）；H200 上仍可用 `--features cuda` 作为文档旗标：

```bash
cargo build -p aria-openai --release --features cuda
aria-engine serve qwen3-0.6b_q4 --hybrid-execution device --compute auto --profile
```

`--profile` 记录加载/生成分段计时。用 `GET /v1/engine/profile` 读取，或：

```bash
python scripts/profile_qwen3_serve.py --compute cpu --spawn --report ./out/engine_profile_qwen3.json
```

在 `--hybrid-execution hybrid` 下，路由依据提示复杂度 / 上下文溢出 / modality / 本地失败 / `FORCE_CLOUD`。`cost` 更偏端侧；`intelligence` 更偏云端；`balance` 为中性自动。用户消息包含 `FORCE_CLOUD` 可强制走云端（测试 / 演示）。`--hybrid-execution device` 永不切换云端；`--hybrid-execution cloud` 始终云端推理（隐私敏感请求仍留本地）。云端 handoff 请求的 `model` 固定为 `ariacompute/ariamodel`。

## OpenAI API

假设服务在 `http://127.0.0.1:8080`：

```bash
# 列出模型
curl -s http://127.0.0.1:8080/v1/models | jq .

# 加载 / 生成分段计时（需 serve --profile）
curl -s http://127.0.0.1:8080/v1/engine/profile | jq .

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

## Qwen3 对话诊断

当 `/v1/chat/completions` 乱码（例如 Hello → `"olum啦…"`）但同一份 Aria bundle 在 HF 上正常时，用这一对脚本拆分问题。两侧共用同一段 ChatML（`enable_thinking=False` / 空 `<think>`），默认 user `Hello`，greedy `max_tokens=32`。

| 脚本 | 隔离什么 |
|------|----------|
| [`../model/scripts/diag_qwen3_chat.py`](../model/scripts/diag_qwen3_chat.py) | **量化 + 模板**：HF fp32 vs 把 `reconstruct_weight` 注入同一张 HF 图 |
| [`scripts/diag_qwen3_chat.py`](scripts/diag_qwen3_chat.py) | **引擎图**：本服务 vs model 侧 JSON（`--peer-report`） |

**1. model 教师**（并列的 `model` 仓库；建议 GPU；tokenizer 从 `--hf` 加载，不用 bundle 里的）：

```bash
# 在 ../model
pip install torch transformers
python scripts/diag_qwen3_chat.py \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --hf Qwen/Qwen3-0.6B \
  --device cuda \
  --report ./out/model_diag_qwen3.json
```

`--device auto` 在有 CUDA 时选用 GPU。Attention 走 eager（关闭 hub CUDA JIT）。若仍编译失败：`sudo apt install python3-dev`。

对照 `chat.fp32` 与 `chat.reconstruct`，以及 `template_string_match` / `prompt_ids_match`。若 reconstruct 已能打出类似 `"Hello! How can I assist you today?"` 且 `exact_prefix_len >= 4`，说明 **bundle 在 HF 上可用**，引擎乱码不是量化问题。

**2. engine**（必须 `serve` **同一份** bundle，且不走云 handoff）：

```bash
# 在本仓库；用 --hybrid-execution device，避免 hybrid 掩盖本地 decode
./aria-engine serve qwen3-0.6b_q4 \
  --bind 127.0.0.1:8080 \
  --hybrid-execution device \
  --compute auto --profile
python scripts/diag_qwen3_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --peer-report ../model/out/model_diag_qwen3.json \
  --report ./out/engine_diag_qwen3.json
```

`--timeout` 默认 300s（CPU decode 可能很慢）。可选 `pip install tokenizers`，以便用 `bundle/tokenizer.json` 编 prompt ids。

**如何读 `hints`**

| 现象 | 更可能的原因 |
|------|----------------|
| `template_string_match` / `prompt_ids_match` 为 false | ChatML / tokenizer 编码不一致 |
| reconstruct greedy 已相对 fp32 发散（`QUANT:…`） | 码本 / Hadamard / 注入 |
| reconstruct 对话正常，引擎 `content` 仍乱（`ENGINE_GRAPH`） | Rust 前向（HDM、embed 按行取值、RoPE、QK-norm 等） |

引擎的教师信号是 **HF + reconstruct 注入**，不是裸 fp32。更多说明见 [`../model/README_cn.md`](../model/README_cn.md)（质量审计）。

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

## SDK Bindings

原生 C ABI（`aria-ffi` / `libaria_ffi`）与 `bindings/` 下薄封装

| Binding | 路径 | Registry |
|---------|------|----------|
| Rust | `bindings/rust`（`aria-engine`） | crates.io |
| Python | `bindings/python` | PyPI |
| Go | `bindings/go` | Go module |
| TypeScript | `bindings/typescript` | npm `@ariacompute/engine-ts` |
| React Native | `bindings/react-native` | npm `@ariacompute/engine-rn` |
| Flutter | `bindings/flutter` | pub.dev |
| Swift | `bindings/swift` | CocoaPods |
| Kotlin | `bindings/kotlin` | Maven |

C 头文件：[`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`、`aria_complete` / stream、`aria_embed`、`aria_transcribe`、tools JSON、`aria_model_destroy`、`aria_last_error`。

```bash
cargo test -p aria-ffi -p aria-engine
./scripts/run-binding-tests.sh   # 主机矩阵（Rust / Python / Go / TS）
```

移动端 e2e（Flutter + React Native，iOS + Android）：[`.github/workflows/bindings-mobile.yml`](.github/workflows/bindings-mobile.yml)。

### libaria_ffi（Release 资产）

每次 GitHub Release，[`.github/workflows/release.yml`](.github/workflows/release.yml) 会在 CLI 包之外上传各平台归档：

| 资产 | 内容 |
|------|------|
| `libaria_ffi_<ver>_linux_x86_64.tar.gz` | `libaria_ffi.so` |
| `libaria_ffi_<ver>_linux_arm64.tar.gz` | `libaria_ffi.so` |
| `libaria_ffi_<ver>_macos.tar.gz` | `libaria_ffi.dylib` |
| `libaria_ffi_<ver>_windows_x86_64.tar.gz` | `aria_ffi.dll`（及可选 import lib） |

```bash
# 示例：Linux x86_64
tar -xzf libaria_ffi_0.7.1_linux_x86_64.tar.gz
export ARIA_FFI_LIB="$PWD/libaria_ffi.so"
# 可选：export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
```

绑定还需本地 Aria bundle（`weight.bin` + `config.json` + tokenizer），例如经 `aria-engine download …` 落到 `~/.ariacompute/models/`。

### 示例

**Python**（通过 `ARIA_FFI_LIB` 加载 `libaria_ffi`）：

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
pip install aria-engine
```

```python
from aria_engine import Engine

with Engine("/path/to/aria-bundle") as eng:
    out = eng.complete(
        [{"role": "user", "content": "Hello"}],
        {"max_tokens": 32},
    )
    print(out["response"])
    # 也可：eng.embed("hi"), eng.transcribe(pcm_bytes)
```

**TypeScript / Node**（`@ariacompute/engine-ts`）：

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
npm install @ariacompute/engine-ts
```

```ts
import { Engine } from "@ariacompute/engine-ts";

const eng = new Engine("/path/to/aria-bundle");
const out = eng.complete(
  [{ role: "user", content: "Hello" }],
  { max_tokens: 32 },
);
console.log(out.response);
eng.close();
```

**Go**（cgo；链接 `libaria_ffi`）：

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
export CGO_ENABLED=1
# 确保链接器能找到该库（或在模块 cgo 中指定 -L）。
go get github.com/ariacompute/engine/bindings/go@latest
```

```go
package main

import (
	"fmt"
	aria "github.com/ariacompute/engine/bindings/go"
)

func main() {
	eng, err := aria.Open("/path/to/aria-bundle")
	if err != nil {
		panic(err)
	}
	defer eng.Close()
	out, err := eng.Complete(
		[]map[string]string{{"role": "user", "content": "Hello"}},
		map[string]any{"max_tokens": 32},
		nil,
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(out["response"])
}
```

**Rust**（`aria-engine` crate — 原生 API；一般无需解压 `libaria_ffi`）：

```bash
cargo add aria-engine
```

```rust
use aria_engine::{Engine, GenerateOpts};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut eng = Engine::open("/path/to/aria-bundle")?;
    let g = eng.complete("Hello", &GenerateOpts {
        max_tokens: 32,
        temperature: 0.0,
    })?;
    println!("{}", g.text);
    Ok(())
}
```

各语言更多说明见 `bindings/*/README.md`。

### 从各 Registry 安装

创建 **GitHub Release** — [`.github/workflows/release.yml`](.github/workflows/release.yml) 构建 CLI + `libaria_ffi` 包并尝试发布语言包（npm / pub.dev / Maven / CocoaPods / crates.io / PyPI）。**发布失败为 fail-pass**，不阻塞 CLI / `libaria_ffi` 资产。所需 secrets：`NPM_TOKEN`、pub 凭证、Maven + GPG、`COCOAPODS_TRUNK_TOKEN`、`CARGO_REGISTRY_TOKEN`、`PYPI_TOKEN`，以及 Release 上传用的 `ARIACOMPUTE_TOKEN`。

版本 = release tag 去掉前导 `v`。

## 工程约定

本仓库遵循 Harness Engineering 理念：

- [`AGENTS.md`](AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](requirements.md)：需求 Spec（功能边界 / 例外 / 验收标准，需人工审核）
- [`task.md`](task.md)：实施任务清单
