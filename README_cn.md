# engine

[English](README.md) | [中文](README_cn.md)

Aria Compute 推理引擎：OpenAI 兼容 API、Aria bundle 推理、零拷贝计算图、ARM NEON / scalar 内核。端云 / Mixture-of-Models 路由在独立 **router** 仓。

## 构建 / 测试

```bash
cargo test
cargo clippy --workspace --all-targets -- -D warnings
```

## 配置 / 运行

凭证保存在 `~/.ariacompute/engine.yml`（通过 `aria-engine setup` 写入）。

| 字段 | 含义 | 默认 |
|------|------|------|
| `router` | 可选 aria-router 管理面 URL | _(空 → 纯本地 serve)_ |
| `router_api_key` | `sk-aria_…` 或 `sk-bf-…`（router 按前缀关联） | _(空)_ |
| `site_url` | 站点（`.com` / `.cn`），用于 hub 分区 | — |
| `upgrade_url` | 组织根（`.com`→GitHub，`.cn`→Gitee） | — |
| `compute` | `auto` / `cpu` / `cuda`（本机 GEMM） | `auto` |
| `hf_token` | Hugging Face hub token（可选） | _(空)_ |
| `modelscope_api_token` | ModelScope hub token（可选） | _(空)_ |

```bash
# 配置 — hub / compute / 可选 router URL + router API key（sk-aria_ 或 sk-bf-）
aria-engine setup
aria-engine setup --status
# Flags: --router --router-api-key

# 下载模型
# 可选：aria-engine setup 按区提示 hf_token（.com）或 modelscope_api_token（.cn）
aria-engine download gemma-4-e2b-it_q4
aria-engine list
aria-engine check gemma-4-e2b-it_q4
# 或：aria-engine check   # 校验全部本地缓存
aria-engine clean gemma-4-e2b-it_q4

# 升级 CLI + libaria_ffi（最新正式版，或指定版本）
# FFI 安装到 ~/.ariacompute/lib/，必要时设置 ARIA_FFI_LIB
aria-engine upgrade
aria-engine upgrade 0.7.2

# 服务（纯本地）
# 或：serve /path/to/aria-bundle
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --compute auto

# 服务并向 aria-router 注册为本地 provider（仅覆盖本进程，不写 engine.yml）
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --router http://127.0.0.1:8090 \
  --router-api-key sk-aria_… \
  --compute auto
```

`download` 每次运行只探测**本区**公开 hub（`.com`→Hugging Face，`.cn`→ModelScope），**不**走 Dashboard，也**不**直连公开 S3/COS registry。私有/需授权的 hub 文件在未配置 token 时会报 `auth failed HTTP 401`：用 `aria-engine setup` 按区写入对应 token（`.com` → `hf_token`，`.cn` → `modelscope_api_token`）到 `~/.ariacompute/engine.yml`。

`list` 只扫描本地 `~/.ariacompute/models`。

`check [model]` 对照本区 hub（与 `download` 相同）校验本地文件数目、文件名与 SHA-256；省略 model 则检查全部缓存。不一致 exit 1；`weight.bin` 只在本地哈希并与 hub 元数据比对，不重新下载。

`serve` 旗标仅覆盖本进程配置（不回写文件）。`serve <model>`：若为现存路径则用之，否则使用 `~/.ariacompute/models/<model>`。`--router URL` / `--router-api-key SECRET` 向 aria-router 注册本进程为本地 provider（不写 `engine.yml`）。网关开启 `require_api_key` 时使用 Dashboard「API 密钥」签发的 secret。

`--compute auto|cpu|cuda` 选择**本机** GEMM：`auto` 在能加载 `libcudart`/`libcublas` 且 `cudaGetDeviceCount>0` 时用 CUDA，否则 CPU（x86_64 AVX2+FMA，aarch64 NEON）。`--compute cuda` 在无 NVIDIA 设备时**硬失败**，不会静默降到 CPU。CUDA 为运行时 libloading（编译不依赖 CUDA toolkit）；H200 上仍可用 `--features cuda` 作为文档旗标：

```bash
cargo build -p aria-openai --release --features cuda
aria-engine serve qwen3-0.6b_q4 --compute auto --profile
```

`--profile` 记录加载/生成分段计时。用 `GET /v1/engine/profile` 读取，或：

```bash
python scripts/profile_qwen3_serve.py --compute cpu --spawn --report ./out/engine_profile_qwen3.json
```

`--compute auto|cpu|cuda` 只决定本机 GEMM。端云 / Mixture-of-Models 路由在独立 **router** 仓。

## 向 aria-router 注册

本进程不做路由。若配置了 `router`（`engine.yml` 或 `--router`），`serve` 在接受请求前向网关注册为本地 provider：

`PUT {router}/v1/router/providers`，body 为 `{name, endpoint, provider_model_id, locality}`，可选 `Authorization: Bearer`（来自 `router_api_key`）。失败则 **退出**（不会静默改成纯本地）。`--router` / `--router-api-key` 只覆盖本进程，不回写 `engine.yml`。

端口不要撞车：engine 数据面（`--bind`）与 router 管理面（`--mgmt-bind`，默认 `127.0.0.1:8080`）。客户端应打 **router 数据面**，而不是 engine。

从 GitHub/Gitee Releases 安装 `aria-router`（`aria-router_<ver>_<os>.tar.gz` 或 `.zip`；包内为 `aria-router` 二进制）。

```bash
# 1. router 仓 — 数据面 :8899，管理面 :8090
cd /path/to/router
cargo run -p aria-router -- serve \
  --config config/examples/semantic-tiny.yaml \
  --bind 127.0.0.1:8899 \
  --mgmt-bind 127.0.0.1:8090

# 2. engine 仓 — OpenAI 在 :8080，再 PUT 到管理面
aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --router http://127.0.0.1:8090 \
  --router-api-key sk-aria_… \
  --compute auto

# 也可写入配置，不必每次带旗标：
#   aria-engine setup  # 可选填写 router URL + router API key（来自 Dashboard）
#   # 或 ~/.ariacompute/engine.yml：
#   # router: http://127.0.0.1:8090
#   # router_api_key: sk-aria_…

# 3. 经网关对话（实名 = bypass，转发到本 engine）
curl -s http://127.0.0.1:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma-4-e2b-it_q4",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens": 32
  }' | jq .

# aria/semantic-auto、aria/agent-auto 只有在 YAML 的 modelRefs /
# default_model 写成同一注册名时才会打到本 engine。看上一跳：
curl -sD - -o /dev/null http://127.0.0.1:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gemma-4-e2b-it_q4","messages":[{"role":"user","content":"Hello"}]}' \
  | grep -i x-aria-router
```

`x-aria-router-layer`：已注册 bundle 名为 `bypass`；走 recipe 入口则为 `semantic` 或 `agent`。

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
    "temperature": 0
  }' | jq .

# Chat（SSE 流式）
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"Hello"}],
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
| [`model/scripts/diag_qwen3_chat.py`](https://github.com/ariacompute/model/tree/main/scripts/diag_qwen3_chat.py) | **量化 + 模板**：HF fp32 vs 把 `reconstruct_weight` 注入同一张 HF 图 |
| [`scripts/diag_qwen3_chat.py`](scripts/diag_qwen3_chat.py) | **引擎图**：本服务 vs model 侧 JSON（`--peer-report`） |

**1. model 教师**（并列的 `model` 仓库；建议 GPU；tokenizer 从 `--hf` 加载，不用 bundle 里的）：

```bash
# 在 model
pip install torch transformers
python scripts/diag_qwen3_chat.py \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --hf Qwen/Qwen3-0.6B \
  --device cuda \
  --report ./out/model_diag_qwen3.json
```

`--device auto` 在有 CUDA 时选用 GPU。Attention 走 eager（关闭 hub CUDA JIT）。若仍编译失败：`sudo apt install python3-dev`。

对照 `chat.fp32` 与 `chat.reconstruct`，以及 `template_string_match` / `prompt_ids_match`。若 reconstruct 已能打出类似 `"Hello! How can I assist you today?"` 且 `exact_prefix_len >= 4`，说明 **bundle 在 HF 上可用**，引擎乱码不是量化问题。

**2. engine**（必须 `serve` **同一份** bundle）：

```bash
# 在 engine
./aria-engine serve qwen3-0.6b_q4 \
  --bind 127.0.0.1:8080 \
  --compute auto --profile
python scripts/diag_qwen3_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/qwen3-0.6b_q4 \
  --peer-report /path/to/model/out/model_diag_qwen3.json \
  --report ./out/engine_diag_qwen3.json
```

`--timeout` 默认 300s（CPU decode 可能很慢）。可选 `pip install tokenizers`，以便用 `bundle/tokenizer.json` 编 prompt ids。

**如何读 `hints`**

| 现象 | 更可能的原因 |
|------|----------------|
| `template_string_match` / `prompt_ids_match` 为 false | ChatML / tokenizer 编码不一致 |
| reconstruct greedy 已相对 fp32 发散（`QUANT:…`） | 码本 / Hadamard / 注入 |
| reconstruct 对话正常，引擎 `content` 仍乱（`ENGINE_GRAPH`） | Rust 前向（HDM、embed 按行取值、RoPE、QK-norm 等） |

引擎的教师信号是 **HF + reconstruct 注入**，不是裸 fp32。

## Gemma-4 对话诊断

当 `gemma-4-e2b-it_q4` 的 `/v1/chat/completions` 乱码（`engine.log` Hello → `"uhnyaчь…"`），而同机 Qwen3 已正常时，用这一对脚本拆分。两侧共用 `gemma4_it`（`<bos><|turn>user…<turn|>\n<|turn>model\n`），默认 user `Hello`，greedy `max_tokens=32`。官方 Hello prompt 为 **10** token。

| 脚本 | 隔离什么 |
|------|----------|
| [`model/scripts/diag_gemma4_chat.py`](https://github.com/ariacompute/model/tree/main/scripts/diag_gemma4_chat.py) | **量化 + 模板**：HF fp32 vs 把 `reconstruct_weight` 注入同一张 HF 图 |
| [`scripts/diag_gemma4_chat.py`](scripts/diag_gemma4_chat.py) | **引擎图**：本服务 vs model 侧 JSON（`--peer-report`） |

**1. model 教师**（并列的 `model` 仓库；建议 GPU）：

```bash
# 在 model
pip install torch transformers
python scripts/diag_gemma4_chat.py \
  --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \
  --hf google/gemma-4-E2B-it \
  --device cuda \
  --report ./out/model_diag_gemma4.json
```

**2. engine**（必须 `serve` **同一份** bundle）：

```bash
# 在 engine
./aria-engine serve gemma-4-e2b-it_q4 \
  --bind 127.0.0.1:8080 \
  --compute auto --profile
python scripts/diag_gemma4_chat.py \
  --url http://127.0.0.1:8080 \
  --bundle ~/.ariacompute/models/gemma-4-e2b-it_q4 \
  --peer-report /path/to/model/out/model_diag_gemma4.json \
  --report ./out/engine_diag_gemma4.json
```

`hints` 读法与 Qwen3 相同（`TEMPLATE` / `QUANT` / `ENGINE_GRAPH`）。Hello 的 `prompt_tokens` 在 Gemma-4 `<|turn>` 模板下应为 **10**。

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
  --backend sglang=http://127.0.0.1:30000 \
  --max-tokens 64 --warmup 1 --runs 3 \
  --report ./out/bench_report.json
# 同时会写入 ./out/bench_report.md
```

可用 `--model-id family=path=id` 或 `backend:family=id` 覆盖模型 id。aria 默认使用家族 path；其余后端默认使用 HF `base_model`。

## SDK Bindings

原生 C ABI（`ariacompute-ffi` / `libaria_ffi`）与 `bindings/` 下薄封装。**不要**与 `libaria-router_ffi` / `ariacompute-router` 混用（见 **router** 仓）。

| Binding | 路径 | Registry |
|---------|------|----------|
| Rust | `bindings/rust`（`ariacompute-engine`） | crates.io |
| Python | `bindings/python` | PyPI |
| Go | `bindings/go` | Go module |
| TypeScript | `bindings/typescript` | npm `@ariacompute/engine-ts` |
| React Native | `bindings/react-native` | npm `@ariacompute/engine-rn` |
| Flutter | `bindings/flutter` | pub.dev |
| Swift | `bindings/swift` | CocoaPods |
| Kotlin | `bindings/kotlin` | Maven |

C 头文件：[`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`、`aria_complete` / stream、`aria_embed`、`aria_transcribe`、tools JSON、`aria_model_destroy`、`aria_last_error`。新增自动下载辅助：`aria_model_cache_dir(model)`（返回 `~/.ariacompute/models/{model}`）与 `aria_is_local_path(ref)`（1=本地路径，0=模型名，-1=错误）。

### 按模型名自动下载

所有语言 SDK 现在同时接受**本地 bundle 路径**或 **Aria 模型名**。含 `/`（或本地已存在）的值视为本地路径直接加载；否则视为模型名。各语言 SDK 均从本区公开 hub 下载（与 `aria-engine download` 相同：`.com`→Hugging Face，`.cn`→ModelScope），**不再请求 Dashboard**。需授权的 hub 文件使用与 `aria-engine setup` 相同的字段，通过实例 `setup`（空构造 → `setup` → `open`）设置；**仅内存**，绝不写入 `engine.yml`（CLI `aria-engine setup` 仍写该文件）。实例字段为空时下载仍可读 `~/.ariacompute/engine.yml`。Dashboard 的 `sk-` token 不会当作 hub 凭证。不读环境变量 `HF_TOKEN` / `MODELSCOPE_API_TOKEN`。公开模型无需 token。缓存中已存在有效 bundle 时直接复用、不重复下载。下载失败抛出明确错误——绝不静默吞掉。

```bash
cargo test -p ariacompute-ffi -p ariacompute-engine
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

各语言 SDK 在首次 `Engine.open` / 等价入口时，若 `ARIA_FFI_LIB`、语言包捆绑路径、`~/.ariacompute/lib/` 均不存在，会自动下载对应归档并解压到 `~/.ariacompute/lib/`。`aria-engine download` 只拉模型 bundle（CLI 二进制不 dlopen FFI）。

```bash
# 示例：Linux x86_64（手动解压；SDK 会自动完成）
tar -xzf libaria_ffi_0.7.1_linux_x86_64.tar.gz
export ARIA_FFI_LIB="$PWD/libaria_ffi.so"
# 可选：export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
```

绑定还需本地 Aria bundle（`weight.bin` + `config.json` + tokenizer），例如经 `aria-engine download …` 落到 `~/.ariacompute/models/`。

### 示例

**Python**（自动安装 `libaria_ffi`；`ARIA_FFI_LIB` 可覆盖）：

```bash
pip install aria-engine
# 可选覆盖：export ARIA_FFI_LIB=/path/to/libaria_ffi.so
```

```python
from aria_engine import Engine

# 本地 bundle 路径：
with Engine("/path/to/aria-bundle") as eng:
    out = eng.complete(
        [{"role": "user", "content": "Hello"}],
        {"max_tokens": 32},
    )
    print(out["response"])
    # 也可：eng.embed("hi"), eng.transcribe(pcm_bytes)

# 或按模型名 —— 从本区公开 hub 自动下载（Hugging Face / ModelScope，不走 Dashboard）：
with Engine("gemma-4-e2b-it_q4") as eng:
    print(eng.complete([{"role": "user", "content": "Hello"}], {"max_tokens": 32})["response"])

# 需授权的 hub 文件 —— 实例 setup（不写 ~/.ariacompute/engine.yml）：
eng = Engine()
eng.setup(hf_token="hf_...")  # .com → Hugging Face
eng.open("gemma-4-e2b-it_q4")
print(eng.complete([{"role": "user", "content": "Hello"}], {"max_tokens": 32})["response"])
eng_ms = Engine()
eng_ms.setup(modelscope_api_token="ms_...", site_url="https://ariacompute.cn")
eng_ms.open("gemma-4-e2b-it_q4")
print(eng_ms.complete([{"role": "user", "content": "Hello"}], {"max_tokens": 32})["response"])
```

**TypeScript / Node**（`@ariacompute/engine-ts`）：

```bash
npm install @ariacompute/engine-ts
# 可选覆盖：export ARIA_FFI_LIB=/path/to/libaria_ffi.so
```

```ts
import { Engine } from "@ariacompute/engine-ts";

// 本地 bundle 路径：
const eng = new Engine("/path/to/aria-bundle");
const out = eng.complete(
  [{ role: "user", content: "Hello" }],
  { max_tokens: 32 },
);
console.log(out.response);
eng.close();

// 或按模型名 —— 自动从本区公开 hub 下载（Hugging Face / ModelScope，不走 Dashboard）：
const eng2 = await Engine.open("gemma-4-e2b-it_q4");
console.log((await eng2.complete([{ role: "user", content: "Hello" }])).response);
eng2.close();

// 需授权的 hub 文件 —— 实例 setup（不写 ~/.ariacompute/engine.yml）：
const engHf = new Engine();
engHf.setup({ hf_token: "hf_..." }); // .com
await engHf.open("gemma-4-e2b-it_q4");
const engMs = new Engine();
engMs.setup({ modelscope_api_token: "ms_...", site_url: "https://ariacompute.cn" });
await engMs.open("gemma-4-e2b-it_q4");
engHf.close();
engMs.close();
```

**Go**（cgo；链接 `libaria_ffi`）：

```bash
export CGO_ENABLED=1
# 可选：export ARIA_FFI_LIB=/path/to/libaria_ffi.so
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
	// 本地 bundle 路径：
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

	// 或按模型名 —— 自动从本区公开 hub 下载（Hugging Face / ModelScope，不走 Dashboard）：
	eng2, err := aria.OpenModel("gemma-4-e2b-it_q4", "", "")
	if err != nil {
		panic(err)
	}
	defer eng2.Close()
	_ = eng2

	// 需授权的 hub 文件 —— 实例 setup（不写 ~/.ariacompute/engine.yml）：
	engHf := aria.NewEngine()
	hf := "hf_..."
	if err := engHf.Setup(aria.SetupUpdates{HFToken: &hf}); err != nil {
		panic(err)
	}
	if err := engHf.Open("gemma-4-e2b-it_q4"); err != nil {
		panic(err)
	}
	defer engHf.Close()
	engMs := aria.NewEngine()
	ms, site := "ms_...", "https://ariacompute.cn"
	if err := engMs.Setup(aria.SetupUpdates{ModelScopeAPIToken: &ms, SiteURL: &site}); err != nil {
		panic(err)
	}
	if err := engMs.Open("gemma-4-e2b-it_q4"); err != nil {
		panic(err)
	}
	defer engMs.Close()
	_ = engHf
	_ = engMs
}
```

**Rust**（`ariacompute-engine` crate — 原生 API；一般无需解压 `libaria_ffi`）：

```bash
cargo add ariacompute-engine
```

```rust
use aria_engine::{SetupUpdates, Engine, GenerateOpts, OpenOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 本地 bundle 路径：
    let mut eng = Engine::open("/path/to/aria-bundle")?;
    let g = eng.complete("Hello", &GenerateOpts {
        max_tokens: 32,
        temperature: 0.0,
    })?;
    println!("{}", g.text);

    // 或按模型名 —— 自动从本区公开 hub 下载（Hugging Face / ModelScope，不走 Dashboard）：
    let mut eng2 = Engine::open_model("gemma-4-e2b-it_q4", &OpenOptions::default())?;
    let g2 = eng2.complete("Hello", &GenerateOpts { max_tokens: 32, temperature: 0.0 })?;
    println!("{}", g2.text);

    // 需授权的 hub 文件 —— 实例 setup（不写 ~/.ariacompute/engine.yml）：
    let mut eng_hf = Engine::new();
    eng_hf.setup(&SetupUpdates { hf_token: Some("hf_...".into()), ..Default::default() })?;
    eng_hf.open_named("gemma-4-e2b-it_q4")?;
    println!("{}", eng_hf.complete("Hello", &GenerateOpts { max_tokens: 32, temperature: 0.0 })?.text);
    let mut eng_ms = Engine::new();
    eng_ms.setup(&SetupUpdates {
        modelscope_api_token: Some("ms_...".into()),
        site_url: Some("https://ariacompute.cn".into()),
        ..Default::default()
    })?;
    eng_ms.open_named("gemma-4-e2b-it_q4")?;
    println!("{}", eng_ms.complete("Hello", &GenerateOpts { max_tokens: 32, temperature: 0.0 })?.text);
    Ok(())
}
```

各语言更多说明见 `bindings/*/README.md`。

### 从各 Registry 安装

创建 **GitHub Release** — [`.github/workflows/release.yml`](.github/workflows/release.yml) 构建 CLI + `libaria_ffi` 包并尝试发布语言包（npm / pub.dev / Maven / CocoaPods / crates.io / PyPI）。**发布失败为 fail-pass**，不阻塞 CLI / `libaria_ffi` 资产。crates.io（`ariacompute-engine`）由 [`.github/workflows/publish-cargo.yml`](.github/workflows/publish-cargo.yml) 发布。所需 secrets：`NPM_TOKEN`、pub 凭证、Maven + GPG、`COCOAPODS_TRUNK_TOKEN`、`CARGO_REGISTRY_TOKEN`、`PYPI_TOKEN`，以及 Release 上传用的 `ARIACOMPUTE_TOKEN`。

版本 = release tag 去掉前导 `v`。

## 工程约定

本仓库遵循 Harness Engineering 理念：

- [`AGENTS.md`](AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](requirements.md)：需求 Spec（功能边界 / 例外 / 验收标准，需人工审核）
- [`task.md`](task.md)：实施任务清单
