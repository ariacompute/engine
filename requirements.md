# requirements.md — aria engine（Rust）

> 本文件为 `engine` 仓库 **OpenAI 兼容 API / 全家族推理 / 零拷贝计算图 / ARM NEON 内核 / Hybrid 路由** 的功能边界、API、数据布局、异常与验收标准。**须经人工逐项审核**，审核通过后方可据其生成 / 执行 `task.md`。
>
> 架构参考：Cactus Engine / Graph / Kernels / Hybrid；Aria《技术架构-大模型》Engine / Graph / Kernels；权重契约对齐 `model` 仓 `aria-quant-bundle` v1。

## 1. 目标与范围

用 **Rust** 实现端侧推理引擎，消费 `model` 仓导出的 **Aria model bundle**，提供本地推理与 OpenAI 兼容 HTTP，并在置信度不足时路由至云端。

- **五层产品面**：`openai` / `inference` / `graph` / `kernel` / `hybrid`（与仓库目录 1:1，Cargo workspace crate：`aria-openai` / `aria-inference` / `aria-graph` / `aria-kernel` / `aria-hybrid`）。
- **权重格式**：仅 `aria-quant-bundle`（`format_version: 1|2`）— `config.json` + `weight.bin` + tokenizer 侧车；**禁止**解析 / 导出 GGUF 与 Cactus 专有权重格式。
- **位宽**：消费 `q1`–`q4` / `q8` / `q1.5` / `q2.54` / `q3.26`（与 `model` 产物一致）。
- **模型覆盖**：§1.1 全部家族（与 `model/requirements.md` §1.1 对齐）。
- **SIMD**：主路径 **ARM NEON**（`aarch64`）；可移植 **scalar** 覆盖 x86_64（WSL/CI）；AVX2 为后续优化，不阻塞 MVP。
- 本增量与 **model** 协同 blocked Hadamard。

### 1.1 家族注册表（推理目标）

| 目录（相对 `model/`） | `base_model` | 架构类 | MVP |
|----------------------|--------------|--------|-----|
| `qwen/qwen3-0.6b` | `Qwen/Qwen3-0.6B` | text LLM | 阶段 B |
| `qwen/qwen3-1.7b` | `Qwen/Qwen3-1.7B` | text LLM | 阶段 B |
| `qwen/qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | text LLM | 阶段 B |
| `qwen/qwen3.5-2b` | `Qwen/Qwen3.5-2B` | text LLM | 阶段 B |
| `gemma/gemma-3-270m-it` | `google/gemma-3-270m-it` | text LLM | 阶段 B |
| `gemma/gemma-3-1b-it` | `google/gemma-3-1b-it` | text LLM | 阶段 B |
| `gemma/gemma-3n-e2b-it` | `google/gemma-3n-E2B-it` | multimodal | 阶段 C |
| `gemma/gemma-3n-e4b-it` | `google/gemma-3n-E4B-it` | multimodal | 阶段 C |
| `gemma/gemma-4-e2b-it` | `google/gemma-4-E2B-it` | multimodal | **阶段 A 黄金路径（tiny 文本）**；全量/ VL 阶段 C |
| `gemma/gemma-4-e4b-it` | `google/gemma-4-E4B-it` | multimodal | 阶段 C |
| `lfm/lfm2-350m` | `LiquidAI/LFM2-350M` | text LLM | 阶段 B |
| `lfm/lfm2-700m` | `LiquidAI/LFM2-700M` | text LLM | 阶段 B |
| `lfm/lfm2-1.2b` | `LiquidAI/LFM2-1.2B` | text LLM | 阶段 B |
| `lfm/lfm2-2.6b` | `LiquidAI/LFM2-2.6B` | text LLM | 阶段 B |
| `lfm/lfm2-8b-a1b` | `LiquidAI/LFM2-8B-A1B` | MoE text | 阶段 B |
| `lfm/lfm2-vl-450m` | `LiquidAI/LFM2-VL-450M` | VL | 阶段 C |
| `lfm/lfm2.5-350m` | `LiquidAI/LFM2.5-350M` | text LLM | 阶段 B |
| `lfm/lfm2.5-1.2b-instruct` | `LiquidAI/LFM2.5-1.2B-Instruct` | text LLM | 阶段 B |
| `lfm/lfm2.5-1.2b-thinking` | `LiquidAI/LFM2.5-1.2B-Thinking` | text LLM | 阶段 B |
| `lfm/lfm2.5-vl-1.6b` | `LiquidAI/LFM2.5-VL-1.6B` | VL | 阶段 C |
| `nanbeige/nanbeige4.2-3b` | `Nanbeige/Nanbeige4.2-3B` | text LLM | 阶段 B |
| `bonsai/bonsai-27b` | `prism-ml/Bonsai-27B-unpacked` | text LLM | 阶段 B |
| `inkling/inkling-small` | `thinkingmachines/Inkling-Small` | text LLM | 阶段 B |
| `openvla/openvla-7b` | `openvla/openvla-7b` | VLA | 阶段 C |
| `openpi/openpi-pi0-3b` | `lerobot/pi0_base` | VLA | 阶段 C |
| `openpi/openpi-pi0.5-3b` | `lerobot/pi05_base` | VLA | 阶段 C |
| `lingbot/lingbot-vla-v2-6b` | `robbyant/lingbot-vla-v2-6b` | VLA | 阶段 C |

**阶段定义**：

| 阶段 | 名称 | 交付深度 |
|------|------|----------|
| **A（MVP）** | 骨架 + 黄金路径 | workspace、五 crate、scalar kernel、graph 调度、bundle 加载、tiny 文本 LLM E2E、OpenAI chat(+SSE)、hybrid mock |
| **B** | 全文本家族 | §1.1 全部 text/MoE 家族可加载并文本生成（可用 tiny fixture 按架构类验收后再接全量权重） |
| **C** | 多模态 / Agent 面 | VL / VLA、ASR、embeddings/RAG、tool_calls；NEON 精度对拍；on-device-only 锁 |

### 1.2 交付风格

完整产品 Spec（本文件覆盖 openai / inference / graph / kernel / hybrid 全能力）；**验收按阶段 A→B→C** 切开。未达阶段的能力须在代码中以明确 `Unsupported` / 功能门控失败，禁止静默空实现冒充完成。

## 2. 功能边界

| # | 特性 | 实现深度 |
|---|------|----------|
| 1 | **kernel** | matmul、attention、RMSNorm、RoPE、Softmax、SwiGLU、码本查表反量化、FWHT；`SimdMode::{Scalar, Neon}`；阶段 A：scalar 完整 + NEON `#[cfg(target_arch="aarch64")]` 入口；阶段 C：NEON 与 scalar 数值对拍 |
| 2 | **graph** | Layer → Op → Tensor IR；`BufferPool`；mmap / `external` 零拷贝输入；融合节点 **HDM**（Hadamard + Dequant + MatMul）；阶段 A：LLM decode 所需 ops 可调度；图序列化可选（阶段 B+） |
| 3 | **inference** | `load_bundle`（mmap `weight.bin`）；KV cache；greedy / 基础 sampling；tokenizer；§1.1 家族注册与图构建钩子；阶段 A：`gemma-4-e2b-it` **tiny q4** 黄金路径；阶段 B/C：见上表 |
| 4 | **openai** | `POST /v1/chat/completions`（含 SSE streaming）、`GET /v1/models`；阶段 C：`/v1/audio/transcriptions`、`/v1/embeddings`、tool_calls / RAG 编排；CLI：`auth` / `download` / `list` / `clean` / `serve` |
| 5 | **hybrid** | 信号→投影→决策：`RouteSignal` → `ProjectionBand` → `RouteDecision{action, reason, policy_version, fallback}`；Pareto 模式 `Cost`/`Balance`/`Intelligence`；会话粘性与失败升级；`RouteOutcome` 内存落盘；云端 OpenAI 兼容 POST；凭证与模式来自 `~/.ariacompute/config.yml`（`CloudClient::new`）；`execution`=`hybrid`\|`device`\|`cloud`；阶段 A：mock + 单测；`device` / privacy 强制 Local，`cloud` 强制 Handoff |
| 6 | **反量化语义** | 与 Python `model.common.quant.dequantize` 一致：**rotated-space** 重建；推理优先融合 HDM，**不**要求加载时整表逆 Hadamard 物化 |
| 7 | **HTTP** | **axum** 实现本地 serve |

### 2.1 非目标

- 解析或导出 **GGUF**；Cactus `CACT` / `.graph` 等专有二进制格式
- 通过 FFI 嵌入 Cactus C++（本仓为 **Rust 原生**重写，仅借鉴架构）
- Metal / Vulkan / ANE / NPU（roadmap，不进本 Spec 验收）
- 生产计费、云端密钥校验服务
- 剪枝 / 蒸馏 / 对称 int8 旁路量化

## 3. API 边界

### 3.1 `aria-kernel`

- `SimdMode::{Scalar, Neon}`；测试可强制 Scalar。
- 算子（至少）：`matmul`、`attention`、`rms_norm`、`rope`、`softmax`、`swiglu`、`dequant_lookup` / `dequant_gemm`、`fwht`。
- 正常 + 维度不匹配异常路径单测。

### 3.2 `aria-graph`

- `Graph` / `Node` / `Op` / `TensorView` / `BufferPool` / `execute`。
- `TensorView`：dtype、shape、strides、底层字节为 mmap 切片或外部借用（零拷贝）。
- 支持将权重 blob 以 `external` 引用挂入图，禁止无谓 `memcpy`。

### 3.3 `aria-inference`

- `load_bundle(path) -> Result<Bundle, EngineError>`：校验 `format == "aria-quant-bundle"` 且 `format_version` ∈ `{1, 2}`。
- `Session` / `SessionBuilder` / `GenerateOpts` / `generate`（prefill + decode）。
- `Family` 注册表：覆盖 §1.1 每一行；未知家族 → `UnsupportedFamily`。
- 阶段 A CLI / 库入口可加载 tiny 目录并产出 token 序列。

### 3.4 `aria-openai`

- `serve(bind, model_dir, hybrid_opts)`（axum）。
- `POST /v1/chat/completions`：非流式 JSON + `stream: true` 时 SSE。
- `GET /v1/models`：按 `hybrid_execution` 列出模型 id——`cloud` 仅 `ariacompute/ariamodel`；`device` 为本地 bundle 目录名；`hybrid` 为本地目录名，若已配置 cloud 凭证则另附 `ariacompute/ariamodel`（不再用内部家族 path 如 `gemma/gemma-4-e2b-it` 冒充云端 id）。
- 请求/响应字段对齐 OpenAI Chat Completions 常用子集（`messages`、`temperature`、`max_tokens`、`stream`）。
- **CLI（`aria-engine`）**
  - 缓存根：`~/.ariacompute/`（`config.yml` + `models/<model>/`）。
  - 子命令：`auth [--status|--clear]`、`download <model>`、`list`、`clean [model]`、`serve <model> [--bind] [--hybrid-mode] [--hybrid-execution]`；`-h` / `-v`。
  - `list`：`GET {site_url}/api/dashboard/models`（Bearer `cloud_api_key`）展开为可下载 bundle（`_q4`/`_q8`/`_q326`），对照本地缓存标记 `downloaded` / `not downloaded` / `incomplete`；另附 catalog 外本地项。
  - `serve <model>`：若为现存路径则用之，否则 `~/.ariacompute/models/<model>`；CLI 旗标仅覆盖本进程，不回写 config。
  - **禁止** `ARIA_HYBRID_*` 环境变量；仅保留编译期 `ARIA_ENGINE_VERSION`。
  - **下载源**（恰三）：Dashboard 认证 API、Hugging Face、ModelScope；**禁止**引擎直连公开 S3/COS registry URL。
  - 每次 `download` 探测连通性 + 短速率采样，选最优可达源；中途失败回退次优已探测源；不持久化强制源。
  - HF/MS 布局对齐 `serve/scripts/upload-model-hub.sh`：`{sdk}/{bundle}/{file}`，默认 `sdk=v1.0`。
  - Bundle 名解析：`*_q4`→int4、`*_q8`→int8、`*_q326`/`*_q3.26`→int326；否则整名 + 默认 int4。

### 3.4.1 Config（`~/.ariacompute/config.yml`）

| 字段 | 含义 |
|------|------|
| `cloud_api_key` | Dashboard / hybrid 同一 API key |
| `cloud_url` | Gateway base（`.com` 或 `.cn`） |
| `site_url` | Dashboard site（`.com` 或 `.cn`） |
| `hybrid_mode` | `cost` \| `balance` \| `intelligence` |
| `hybrid_execution` | `hybrid` \| `device` \| `cloud` |

- `auth`：提示 API key → **用 key 探测** Dashboard（`GET /api/dashboard/models`）判定 `.com` / `.cn`，写入匹配的 `cloud_url`/`site_url`（同 TLD）；两端均失败时回退 locale + 连通性。`list`/`download` 启动时若 URL 不一致或 key 被当前 site 拒绝会自动纠偏并回写 config。
- Gateway / site 始终成对（同为 `.com` 或同为 `.cn`）；下载源每次运行时探针选择。

### 3.5 `aria-hybrid`

- **P0**
  - `RouteAction::{Local, CloudHandoff}`；`RouteDecision` 含 `action` / `reason` / `policy_version` / `fallback` / `projection` / `mode`。
  - `ParetoMode::{Cost, Balance, Intelligence}`：调节复杂度 handoff 阈值（Cost=`0.90` 偏 Local，Balance=`0.75`，Intelligence=`0.40` 更易 Handoff）；`confidence` 保留字段但不参与 handoff。
  - 硬约束：`execution=device`、`privacy_sensitive` → 强制 Local；`=cloud` → 强制 CloudHandoff（云不可用时仍走 handoff 路径并报错，禁止静默本地）；`!cloud_available` 时 hybrid 模式不得 Handoff；本地不支持 modality / 上下文超限且云可用 → PreferCloud。
  - 会话粘性：同 `session_id` 默认保持上次 `action`；仅硬约束或 `consecutive_local_failures >= upgrade_after_failures` 允许 Local→Cloud 升级（高复杂度为软升级，粘性可拦住）。
  - `RouteOutcome` + `OutcomeStore`（进程内）：记录 action、reason、tokens、latency、handoff、可选 user_corrected。
- **P1（薄信号面）**
  - `RouteSignal`：`confidence`（兼容保留）、`complexity`、`context_tokens`/`context_limit`、`modality_unsupported_locally`、`consecutive_local_failures`、`privacy_sensitive`、`cloud_available`、`session_id`、`force_cloud`。
  - `ProjectionBand::{MustLocal, LocalOk, PreferCloud}`；决策由投影 + 模式复杂度阈值合成。
  - chat 路径用 `estimate_route_signals(prompt, context_limit)` 填充 `complexity` / `context_tokens`。
- `Router::new() -> Result<Self, EngineError>`；`route(&self, &RouteSignal) -> RouteDecision`（兼容 `route_confidence(f32)`）。
- `CloudClient::new(base_url, api_key)`：OpenAI 兼容 HTTP；超时与非 2xx → `EngineError::Cloud`；handoff 请求 `model` 固定为 `ariacompute/ariamodel`（`CLOUD_GATEWAY_MODEL`）。**无** `from_env` / `ARIA_HYBRID_*`。
- 单测：模式复杂度阈值、硬约束、粘性升级、投影、Outcome、Cloud mock 成功/失败。

### 3.6 错误类型

统一 `EngineError`（或等价枚举，各 crate `thiserror` / `From` 贯通）：

| 变体 | 含义 |
|------|------|
| `Io` | 文件 / 网络 IO |
| `Format` | bundle 损坏、缺字段、offset 越界、`format` 不匹配 |
| `ShapeMismatch` | 算子维度不匹配 |
| `Quant` | 不支持的 bits / 码本布局 / pack 错误 |
| `UnsupportedFamily` | 未注册或未实现的家族 |
| `Cloud` | 云卸载失败（超时、非 2xx、JSON 解析） |
| `InvalidParam` | 参数越界（max_tokens=0、非法 `hybrid_execution` 等） |
| `Unsupported` | 未实现算子 / 阶段未开通的 API |

禁止 `panic` 作为预期错误路径；禁止吞掉错误。

### 3.7 SDK / Bindings（C ABI）

跨语言唯一契约为 **C ABI**（`aria-ffi`：`cdylib` + `staticlib`，头文件 `ffi/include/aria.h`）。禁止嵌入 Cactus C++（§2.1）；本仓自有 ABI 允许。

**入口（OpenAI 面 parity）：**

| C API | 语义 |
|------|------|
| `aria_model_init(path)` | 加载 Aria bundle → opaque handle |
| `aria_complete(…, messages_json, options_json, tools_json, out, …)` | chat；可选 tools；JSON 出参 |
| `aria_complete_stream(…, callback)` | 流式 token/chunk 回调 |
| `aria_embed(…, input_json, out)` | embeddings |
| `aria_transcribe(…, pcm, len, options_json, out)` | ASR |
| `aria_model_destroy` / `aria_last_error` | 生命周期 / 错误 |

**语言包：** Python、Go、Rust（`aria-sdk`）、Swift、Kotlin、Flutter、React Native（npm `@ariacompute/engine-rn`）、TypeScript（npm `@ariacompute/engine-ts`）。布局：`ffi/` + `bindings/<lang>/` + `bindings/testdata/`。

**测试：** 共享 `cases.json`（lifecycle / chat / stream / tools / embed / ASR）；`cargo test -p aria-ffi`；`./scripts/run-binding-tests.sh`。Flutter/RN：iOS+Android device-farm/emulator CI（`.github/workflows/bindings-mobile.yml`）。

**发布：** GitHub Release 触发 `release.yml`：CLI 资产 + 尝试发布 Maven / CocoaPods / npm / pub.dev / crates.io / PyPI；**publish fail-pass**（不阻断 CLI/资产上传）。版本 = tag 去 `v`。

**Serve：** 主页与文档（全站语言）Tab 展示各 binding 调用示例。

## 4. 数据布局

### 4.1 Aria bundle（加载契约）

与 `model/common/bundle.py` 对齐：

**`config.json` 顶层**：

| 字段 | 类型 | 约束 |
|------|------|------|
| `format` | string | 必须 `"aria-quant-bundle"` |
| `format_version` | int | `1`（legacy pad-crop）或 **`2`（blocked Hadamard）** |
| `quantization` | string | `q1`…`q4` / `q8` / `q1.5` / `q2.54` / `q3.26` |
| `group_size_default` | int | 通常 32 |
| `hadamard_seed` | int \| null | 全局种子；**v2** 用 portable SplitMix 派生每块 ±1（与 `model.common.hadamard.portable_block_signs` 一致） |
| `model` | object | 见下 |
| `tensors` | object | name → 张量元数据 |
| 可选 | `bit_policy` 等 | `q1.5` 等 extras |

**`model` 对象（常用）**：`hidden_size`、`num_layers`、`num_attention_heads`、`num_kv_heads`、`intermediate_size`、`vocab_size`、`context_length`、`rope_theta`。

**张量 `kind == "codebook"`**：`bits`、`group_size`、`shape` `[K,N]`、`row_pad`（**仅** group 对齐）、`codebook_share`（`group`\|`channel`）、`hadamard`（v2：`mode=blocked`、`blocks=[{start,size},…]`、`applied`、`seed`）、`offsets`（`packed_indices`、`codebook` 必填；legacy `input_scale*` / `norms` 可选）。

**张量 `kind == "raw"`**：`dtype` `f16`\|`f32`、`shape`、`offsets.data`。

**`weight.bin`**：无文件头；按 `offsets` 的 `[start, length]` 切片。索引：bits 1–4 为 **LSB-first** 位打包；bits 8 为每索引 1 字节 `uint8`。码本：fp16，C-order；`group` → `(G, Kc)`，`channel` → `(G, N, Kc)`，`Kc = 2^bits`。

**Hadamard / HDM（v2）**：权重存旋转域；`HdmLinear` 先 `W_rot @ x` 再对输出维做 blocked **unrotate**（`S@H`），等价原域 `W@x`。禁止依赖全局 pad→crop。

**tokenizer 侧车**（若存在）：`tokenizer.json`、`tokenizer.model`、`tokenizer_config.json`、`special_tokens_map.json`、`vocab.json`、`merges.txt`。

### 4.2 运行时

- `TensorView`：零拷贝视图；权重优先 mmap。
- KV cache：按层 `[seq, num_kv_heads, head_dim]`（具体 dtype 在实现中固定并单测）。
- 图内中间激活由 `BufferPool` 复用，避免热路径频繁分配。

## 5. 异常（行为要求）

| 场景 | 期望 |
|------|------|
| 缺 `config.json` / `weight.bin` | `Format` |
| `format` ≠ `aria-quant-bundle` | `Format` |
| offset 越界 / codebook 长度非法 | `Format` 或 `Quant` |
| 不支持 bits | `Quant` |
| 家族未实现 | `UnsupportedFamily` |
| matmul 维度错误 | `ShapeMismatch` |
| 云端超时 / 非 2xx | `Cloud`（进程不崩溃） |
| 非法 API 参数 | `InvalidParam` |
| 阶段未开通的路由（如阶段 A 调 ASR） | `Unsupported` |

## 6. 验收标准

### 6.1 Harness

1. 根目录存在 `AGENTS.md`（≤100 行）与本 `requirements.md`。
2. **本文件经人工逐项审核通过**后，方可生成 `task.md` 并编码。
3. 新增功能 / Bug 修复含单测；合入前 `cargo test` 全绿。

### 6.2 阶段 A（MVP）— 合入门禁

1. Cargo workspace 五 crate 可 `cargo build` / `cargo test`（x86_64 scalar）。
2. 加载 **`gemma/gemma-4-e2b-it` tiny q4** Aria bundle（由 `model` 仓 `quantize.py --tiny --bits 4` 产出，产物不入本仓 Git；测试可用生成夹具或路径环境变量）。
3. 反量化与 Python `dequantize` 参考在约定误差界内（RMSE 上界写入单测常量，与 `model` 高斯/tiny 实践同量级）。
4. Prefill + decode 产出非空 token 序列（greedy 可复现）。
5. `POST /v1/chat/completions` 非流式与 SSE 均可测通。
6. Hybrid：低置信 / PreferCloud 时 `CloudHandoff`（带 `reason`）；Pareto 模式与粘性单测；mock 云端成功与失败路径覆盖。
7. §1.1 注册表在代码中完整列出；非阶段 A 家族返回明确 `UnsupportedFamily` / 门控，不得 panic。

### 6.3 阶段 B / C（后续验收，写入 Spec 以免范围漂移）

- **B**：每个 text/MoE 家族具备 loader/graph 钩子并通过该类 tiny 或全量文本生成测试。
- **C**：VL/VLA、ASR、embeddings/RAG、tool_calls 的 OpenAI 面；NEON vs scalar 对拍；`hybrid_execution=device` 禁止云卸载、`=cloud` 强制云端。

## 7. 目录与依赖约定

```
engine/
  AGENTS.md
  requirements.md
  task.md                 # 审核通过后生成
  README.md
  Cargo.toml              # workspace
  openai/                 # aria-openai
  hybrid/                 # aria-hybrid
  inference/              # aria-inference
  graph/                  # aria-graph
  kernel/                 # aria-kernel
  bench/                  # Python 引擎对标评测（report-only）
```

- HTTP：**axum**。
- 混合云：`~/.ariacompute/config.yml` 的 `cloud_url` + `cloud_api_key`；阶段 A 测试用 mock。
- CLI 下载：Dashboard / HF / ModelScope 探针择优；无公开 S3 客户端。
- 权重与多 GB 产物 **不入 Git**。
- 评测：`bench/` 为 **Python ≥3.10、标准库为主**（对齐 `model` 的 `audit_cli` 风格）；不解析 GGUF。本增量与 **model** 协同 blocked Hadamard（`format_version=2`）。

## 8. 引擎对标评测（`bench/`）

> 对齐 `model` 仓 `audit_cli`：**report-only**（默认不因阈值 / 缺后端失败 CI）；产物 **JSON + Markdown**（无 HTML）。
> 人工锁定（2026-08-03）：交付 JSON+MD；后端 aria + llama.cpp + ollama + vllm；指标性能+质量；覆盖 §1.1 全部家族；实现语言 `bench/` Python。

### 8.1 目标

在统一 **OpenAI 兼容** `POST /v1/chat/completions` 面上，将 **aria-engine** 与主流推理后端对比，覆盖 §1.1 全部家族行，生成可归档评测报告。

### 8.2 后端

| id | 典型服务 | 约定 |
|----|----------|------|
| `aria` | `aria-engine serve` | Aria bundle；`serve <model>` 为路径或 `~/.ariacompute/models/<model>` |
| `llamacpp` | `llama-server` OpenAI 兼容 | 仅 HTTP 调用；本仓不解析 GGUF |
| `ollama` | Ollama `/v1` | 同上 |
| `vllm` | vLLM OpenAI 兼容 | 同上 |

- CLI：`--backend <id>=<base_url>`（可重复）；缺省 / 不可达 → 该后端 `status: skipped`（`ci_fail: false`）。
- 每家族在各后端的 `model` 字段默认取 §1.1 `base_model`；可用 `--model-id <family_path>=<id>` 或配置文件覆盖。
- **禁止**在 `bench/` 内启动或打包第三方引擎二进制；文档说明外部启动方式即可。

### 8.3 家族覆盖

- 注册表与 §1.1 / `model/tests/test_families.py` `EXPECTED` 一致（27 行）。
- `kind`：`text` | `vl` | `vla`（与架构类对应；VL/VLA 默认用文本 chat 探针，不支持则 `skipped` + reason）。
- `--family` 可过滤；默认跑全表。

### 8.4 指标

**性能（每 backend × family × prompt，聚合后写入报告）：**

- `latency_ms`：端到端（非流式）p50 / p95 / mean  
- `ttft_ms`：若后端支持 SSE 则测首 token；否则 `null` + note  
- `tokens_per_sec`：`completion_tokens / (latency_s)`（无 usage 则按字符启发式并标注）  
- `warmup` / `runs`：可配置（默认 warmup=1, runs=3）

**质量：**

- 参考后端：`--ref-backend`（默认优先 `llamacpp`，否则第一个非 `aria` 可用后端）  
- `token_overlap`：与 `model.common.gen_compare._token_overlap` 同语义（空白分词 Jaccard）  
- `exact_match`：规范化后字符串全等  
- 无参考后端时质量段 `skipped`

### 8.5 CLI 与产物

```bash
python -m bench run \
  --backend aria=http://127.0.0.1:8080 \
  --backend llamacpp=http://127.0.0.1:8081 \
  --backend ollama=http://127.0.0.1:11434 \
  --backend vllm=http://127.0.0.1:8000 \
  --max-tokens 64 --warmup 1 --runs 3 \
  --report ./out/bench_report.json
```

- 同目录写 `bench_report.md`（或 `--report-md`）。
- JSON 顶层：`mode: "engine_bench"`、`ci_fail: false`、`families`、`backends`、`results[]`、`summary`。
- 缺依赖 / 缺服务 / 超时：条目 `skipped` 或 `error`，进程默认 **exit 0**（配置错误 exit 2，对齐 `audit_cli`）。

### 8.6 验收

1. `python -m unittest discover -s bench/tests -t .` 全绿（mock HTTP，无外部引擎）。  
2. 注册表长度与 §1.1 一致。  
3. 报告同时产出 `.json` 与 `.md`。  
4. 四后端适配器存在；未配置 URL 时 skip 而非崩溃。

## 9. 审核检查表

- [x] §1.1 与 `model/requirements.md` §1.1 家族列表一致可接受
- [x] 仅 `aria-quant-bundle`、禁止 GGUF 可接受
- [x] 阶段 A 黄金路径 = `gemma-4-e2b-it` tiny q4 可接受
- [x] 全家族在 Spec 内、E2E 分 A/B/C 可接受
- [x] OpenAI：阶段 A 仅 chat(+SSE)/models；ASR/RAG/Tool 属阶段 C 可接受
- [x] Hybrid：阶段 A mock + config/`CloudClient::new` 可接受
- [x] CLI：`~/.ariacompute` + auth/download/list/clean/serve；无 `ARIA_HYBRID_*`；三源探针下载可接受
- [x] Kernel：NEON 主路径 + x86 scalar CI 可接受
- [x] 反量化 = rotated-space + 融合 HDM、不强制加载期逆 H 可接受
- [x] 五 crate 命名与目录映射可接受
- [x] 非目标（不动 model、无 Metal/计费、无 Cactus FFI）可接受
- [x] 验收门禁（`cargo test`、单测覆盖正常/异常）可接受
- [x] §8 引擎对标评测（JSON+MD；aria/llamacpp/ollama/vllm；性能+质量；全家族；`bench/` Python）可接受
- [ ] §3.7 SDK bindings（C ABI + 八语言；OpenAI FFI 面；host/device-farm 测；release.yml 多 registry 发布 fail-pass）可接受

> **人工审核状态**：2026-08-02 **已通过（approved）**。§8 增补经 2026-08-03 用户锁定范围 **已通过**。§3.7 SDK bindings 按 2026-08-15 计划实施。可据本 Spec 生成 / 执行 `task.md`。

## 10. 参考

- Cactus：<https://github.com/cactus-compute/cactus>；Bindings：<https://docs.cactuscompute.com/latest/#bindings>
- `model` 契约：`model/common/bundle.py`、`pack.py`、`quant.py`、`hadamard.py`
- `model` Spec：`model/requirements.md`；质量审计：`model/common/audit_cli.py`
- 调研笔记：桌面 `cactus_compute_research.md`；PPT《技术架构-大模型》
