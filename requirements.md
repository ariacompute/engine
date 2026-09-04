# requirements.md — aria engine（Rust）

> 本文件为 `engine` 仓库 **OpenAI 兼容 API / 全家族推理 / 零拷贝计算图 / ARM NEON 内核 / Hybrid 路由** 的功能边界、API、数据布局、异常与验收标准。**须经人工逐项审核**，审核通过后方可据其生成 / 执行 `task.md`。
>
> 架构参考：Cactus Engine / Graph / Kernels / Hybrid；Aria《技术架构-大模型》Engine / Graph / Kernels；权重契约对齐 `model` 仓 `aria-quant-bundle` v1。

## 1. 目标与范围

用 **Rust** 实现端侧推理引擎，消费 `model` 仓导出的 **Aria model bundle**，提供本地推理与 OpenAI 兼容 HTTP。端云 / Mixture-of-Models 路由由独立 **`router` 仓**（`aria-router`）承担；本仓可把本地 serve 注册为该网关的一个 provider。

- **四层产品面**：`openai` / `inference` / `graph` / `kernel`（Cargo：`aria-openai` / `ariacompute-inference` / `ariacompute-graph` / `ariacompute-kernel`）。**禁止**在本仓再做 Local/Cloud handoff。
- **权重格式**：仅 `aria-quant-bundle`（`format_version: 1|2`）— `config.json` + `weight.bin` + tokenizer 侧车；**禁止**解析 / 导出 GGUF 与 Cactus 专有权重格式。
- **位宽**：消费 `q1`–`q4` / `q8` / `q1.5` / `q2.54` / `q3.26`（与 `model` 产物一致）。
- **模型覆盖**：§1.1 全部家族（与 `model/requirements.md` §1.1 对齐）。
- **本机算力**：`compute`=`auto`\|`cpu`\|`cuda`。CPU = `SimdMode::{Scalar, Neon, Avx2}`（`aarch64` Neon；x86_64 AVX2+FMA 若可用否则 scalar + 多线程 `linear`）。可选 **CUDA**（NVIDIA + cuBLAS SGEMM）；`auto` 探测到可用 CUDA 则用之，否则 CPU。`compute` **不是** 路由开关。
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
| `lfm/lfm2.5-2.6b` | `LiquidAI/LFM2.5-2.6B` | text LLM | 阶段 B |
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
| **A（MVP）** | 骨架 + 黄金路径 | workspace、四 crate、scalar kernel、graph 调度、bundle 加载、tiny 文本 LLM E2E、OpenAI chat(+SSE) |
| **B** | 全文本家族 | §1.1 全部 text/MoE 家族可加载并文本生成（可用 tiny fixture 按架构类验收后再接全量权重） |
| **C** | 多模态 / Agent 面 | VL / VLA、ASR、embeddings/RAG、tool_calls；NEON 精度对拍；on-device-only 锁 |

### 1.2 交付风格

完整产品 Spec（本文件覆盖 openai / inference / graph / kernel）；**验收按阶段 A→B→C** 切开。未达阶段的能力须在代码中以明确 `Unsupported` / 功能门控失败，禁止静默空实现冒充完成。路由能力见 **router 仓** `requirements.md`。

## 2. 功能边界

| # | 特性 | 实现深度 |
|---|------|----------|
| 1 | **kernel** | matmul、attention、RMSNorm、RoPE、Softmax、SwiGLU、码本查表反量化、FWHT；`SimdMode::{Scalar, Neon, Avx2}` + `ComputeBackend::{Cpu, Cuda}`；`linear`/`attention` 经 backend 分发；CUDA 与 scalar 相对误差有界；无 GPU 时 `--compute cuda` 硬失败（禁止静默 CPU） |
| 2 | **graph** | Layer → Op → Tensor IR；`BufferPool`；mmap / `external` 零拷贝输入；融合节点 **HDM**（Hadamard + Dequant + MatMul）；阶段 A：LLM decode 所需 ops 可调度；图序列化可选（阶段 B+） |
| 3 | **inference** | `load_bundle`（mmap `weight.bin`）；KV cache；greedy / 基础 sampling；tokenizer；§1.1 家族注册与图构建钩子；阶段 A：`gemma-4-e2b-it` **tiny q4** 黄金路径；阶段 B/C：见上表 |
| 4 | **openai** | `POST /v1/chat/completions`（含 SSE streaming）、`GET /v1/models`；阶段 C：`/v1/audio/transcriptions`、`/v1/embeddings`、tool_calls / RAG 编排；CLI：`setup` / `download` / `list` / `check` / `clean` / `upgrade` / `serve`。chat **仅本地 decode**。可选 `router` URL：listen 后向 `aria-router` 管理面 upsert 本进程为 provider；失败则退出。 |
| 5 | **router 接入** | `engine.yml`：`router` + `router_api_key`（`sk-aria_`）注册；可选 `serve_site`/`serve_api_key`（`bfvk`，仅存储）。CLI 分段 Local vs OAuth。`--router` / `--router-api-key` 覆盖本进程。**禁止** `cloud_handoff` / `route_hybrid`。 |
| 6 | **反量化语义** | 与 Python `dequantize` 一致的 **rotated-space** 码本重建。embedding **必须**原域行 gather（加载期整表 unrotate 或等价）；线性层可用融合 HDM **或** 原域 `linear`。禁止对旋转域 `W[token]` 做 lookup |
| 7 | **HTTP** | **axum** 实现本地 serve |

### 2.1 非目标

- 解析或导出 **GGUF**；Cactus `CACT` / `.graph` 等专有二进制格式
- 通过 FFI 嵌入 Cactus C++（本仓为 **Rust 原生**重写，仅借鉴架构）
- Metal / Vulkan / ANE / NPU（roadmap，不进本 Spec 验收）。**CUDA 进入本增量**（cuBLAS f32 GEMM + attention 分发）；不引入 PyTorch/candle 重型框架；INT4 tensor core / 多卡不在本增量
- 生产计费、云端密钥校验服务
- 剪枝 / 蒸馏 / 对称 int8 旁路量化

## 3. API 边界

### 3.1 `ariacompute-kernel`

- `SimdMode::{Scalar, Neon, Avx2}`；测试可强制 Scalar。`ComputePref::{Auto, Cpu, Cuda}` → `ComputeBackend::{Cpu, Cuda}`。
- 算子（至少）：`matmul`、`linear`（多线程 / AVX2）、`attention`（含 causal prefill）、`rms_norm`、`rope`、`softmax`、`swiglu`、`dequant_lookup` / `dequant_gemm`、`fwht`；可选运行时 cuBLAS `linear`。
- 正常 + 维度不匹配异常路径单测；无 GPU 的 CI 不开 CUDA、不因缺卡失败。

### 3.2 `ariacompute-graph`

- `Graph` / `Node` / `Op` / `TensorView` / `BufferPool` / `execute`。
- `TensorView`：dtype、shape、strides、底层字节为 mmap 切片或外部借用（零拷贝）。
- 支持将权重 blob 以 `external` 引用挂入图，禁止无谓 `memcpy`。

### 3.3 `ariacompute-inference`

- `load_bundle(path) -> Result<Bundle, EngineError>`：校验 `format == "aria-quant-bundle"` 且 `format_version` ∈ `{1, 2}`。
- `Session` / `SessionBuilder` / `GenerateOpts` / `generate`（prefill + decode）。
- `Family` 注册表：覆盖 §1.1 每一行；未知家族 → `UnsupportedFamily`。
- 阶段 A CLI / 库入口可加载 tiny 目录并产出 token 序列。

### 3.3.1 全家族真实 Bundle 对齐（与 model 协同）

> 背景：Session 当前共用一条 LLaMA 式 decoder；tiny fixture / 形状门控通过 ≠ 真实 q4 数值与图正确。Gemma-4 **按层 FFN 宽度**与 **缺 k/v 可 materialize** 已落地；下列按优先级推进，未达项须 `Unsupported` / 明确门控，禁止静默空实现。

| 优先级 | 主题 | 要求 |
|--------|------|------|
| P0 | **Session HDM** | codebook 线性层走融合 HDM **或** 先 unrotate 再 matmul；禁止对 rotated-space `W` 直接当原域 `linear`。embedding / PLE 表禁止旋转域行 gather：加载期按 bundle `hadamard.blocks` 整表 unrotate 后再 gather |
| P0 | **RoPE 布局** | 与 HF `rotate_half`（及家族 partial factor）对齐；单测对照 fixture |
| P0 | **LFM conv hybrid** | LFM2/2.5：`layer_types` / conv 层不要求 `q_proj`；实现 short-conv + cache；消费 `conv.*` 张量名 |
| P0 | **MoE** | `lfm2-8b-a1b`、Inkling：真实 router + expert FFN；`text_moe_decoder_stub` 不得冒充生成；Inkling ArchClass 不得标纯 TextDense |
| P0 | **Qwen3.5 / Bonsai** | linear_attention / Gated DeltaNet 层：实现或对未实现层返回 `Unsupported`；禁止当全 SDPA dense。消费 HF Qwen3.5 拆分投影（`in_proj_qkv`+`in_proj_z`、`in_proj_b`+`in_proj_a`），兼容 Qwen3-Next 融合 `in_proj_qkvz` / `in_proj_ba`。全注意力 `attn_output_gate`：`q_proj` 为每头 `[q \| gate]`（out=`2*heads*head_dim`），`o_proj` 仍为 `heads*head_dim`；attn 后 `sigmoid(gate)` 再投影。**RoPE**：HF `rope_parameters.partial_rotary_factor=0.25`（只转前 `head_dim*0.25`，inv_freq 以该 rotary_dim 为分母，**不是** Gemma-4 p-RoPE）+ `rope_theta=1e7`；hub q4 缺字段时引擎按此补齐。禁止对 Qwen3.5 全头 `rotate_half` / Llama θ=1e4（Hello greedy 会跑满 `max_tokens` 且 `content` 为空）。**RMSNorm**：HF `Qwen3_5RMSNorm` 为 Gemma 式 `x*rrms*(1+w)`（weight 零初始化），含 input/post-attn/q_norm/k_norm/final `norm`；禁止当 Qwen3/Llama `*w`（隐状态被压扁后 Hello greedy `content` 为空）。DeltaNet 输出 `Qwen3_5RMSNormGated` 仍为 ones-init `*w`，须加载 `linear_attn.norm.weight`。 |
| P1 | **Gemma 正确性** | GeGLU（`gelu_pytorch_tanh`）vs SwiGLU；RMSNorm `*(1+w)`（**Gemma-4 / Gemma-3n** 为 `*w`，ones 初始化 / `scale_plus_one=False`）；四 norm；`ffn_norm` 优先 `pre_feedforward_layernorm`。**Gemma-3 文本（270m/1b，不含 3n）**：HF `_sliding_window_pattern=6`（5×sliding+1×full）、`sliding_window=512`、双 RoPE（sliding θ=1e4 / full θ=1e6，**不是** Gemma-4 p-RoPE）；hub q326/q4 缺 `layer_types`/`hidden_act` 时引擎按此补齐。禁止把 Gemma-3 当全 full + 单一 θ（Hello greedy 会塌成印地语循环乱码）。**Gemma-3n E2B/E4B** 见 **§3.3.3**（4×sliding+1×full、双 RoPE 全头、attn scale `1.0`、logit softcap 30、AltUp+Laurel、PLE 加到非 active 流、前 10 层 gaussian top-k）。Gemma-4 文本 decoder 图见 **§3.3.2**（KV-share、滑动窗口、双/部分 RoPE、PLE、**`layer_scalar`**）。**真实 E2B/E4B（hidden≥1024）必须加载码本 PLE**（`embed_tokens_per_layer` + projection + 每层 gate/proj/norm）；禁止 `ple=None` 静默 no-op。词表/PLE 2D 与其它线性层同一解包：LSB unpack → 旋转域 dequant → `hadamard.blocks` 逆 blocked FWHT → 行 gather（shape `[vocab, hidden]` / `[vocab, layers*d]`）。 |
| P1 | **QK-Norm** | Qwen3 / Gemma / LFM attn：加载并应用 `q_norm`/`k_norm` |
| P1 | **VL/VLA** | 消费 bundle 内 vision/action 张量，或硬 `Unsupported`；禁止 RGB mean-pool / 假 action 冒充完成 |
| P2 | **注册表对齐** | §1.1 与 `model` 同步：补登记 `lfm/lfm2.5-2.6b`（或双方删除） |
| P2 | **Bundle `model` 扩展字段** | 消费 model 仓写入的 `head_dim` / `layer_types` / `num_kv_shared_layers` / `hidden_act` / nested RoPE 等（见 model Spec）。**Gemma-4** 另认 `sliding_window` / `global_head_dim` / `partial_rotary_factor`；**Gemma-3 文本**另认 5+1 `layer_types` / `sliding_window=512`；**Gemma-3n** 另认 4+1 `layer_types` / `sliding_window=512` / `head_dim=256` / `num_kv_shared_layers`（缺省按 `num_layers−20`：E2B 10 / E4B 15；**不是** Gemma-4 `global_head_dim=512` 或 p-RoPE）。bundle 缺省时按 HF 架构补齐（hub 旧 q4/q326 仅含基础字段亦可加载），显式写入值优先；Gemma-4 补齐后仍不完整 → `Unsupported`。推荐用当前 `config_from_hf` 重量化并重新上传 hub。 |

### 3.3.2 Gemma-4 文本 decoder（hub q4）

Hub 已发布 `gemma-4-e2b-it_q4` 为消费契约：引擎须按现有张量名加载，**禁止**因缺 `config.json` 扩展字段而拒绝（几何补齐见上表 P2）。文本路径与 HF `Gemma4DecoderLayer` 对齐。

**每层顺序**（prefill 与 decode 相同）：

1. pre-attn RMSNorm → QK-norm → RoPE（sliding θ=1e4 / full p-RoPE θ=1e6）→ SDPA（滑动窗口默认 512 或 full causal；attn scale `1.0`）→ 残差。KV-share 层复用 producer cache，禁止 clone `wk`/`wv`。
2. post-attn RMSNorm（若存在）。
3. pre-FFN RMSNorm（`pre_feedforward_layernorm`）→ GeGLU → 残差；post-FFN RMSNorm（若存在）。
4. PLE（真实 E2B/E4B 必有，见 P1）。
5. **`hidden *= layer_scalar`**。

**`layer_scalar`（HF 同名 / JAX `skip_scale`）**：

| 项 | 约定 |
|----|------|
| 张量名 | 无 `.weight` 后缀：`blk.{i}.layer_scalar`、`model.language_model.layers.{i}.layer_scalar` 等 |
| 布局 | `kind=raw`，shape `[]` 或 `[1]`；hub E2B q4 为每层一个（35 层） |
| 加载 | 取有限标量；**缺省 1.0**（tiny / 无该张量的 fixture） |
| 应用 | 该层 attn+FFN+PLE 全部完成后、进入下一层之前 |
| 禁止 | 因无 `.weight` 忽略该张量。发布 checkpoint 的值 **不是** 1.0；省略等价恒等残差，greedy Hello 会变成多语言乱码 |

embed 缩放 `sqrt(hidden)`；`final_logit_softcapping` 默认 30。chat 模板为 Gemma-4 `<|turn>`；Hello 的 `prompt_tokens` 为 **10**。

### 3.3.3 Gemma-3n 文本 decoder（hub q4）

Hub `gemma-3n-e2b-it_q4` / `gemma-3n-e4b-it_q4` 为消费契约。Gemma-3n **不是** Gemma-3 文本（5+1 + `*(1+w)`），也 **不是** Gemma-4 p-RoPE。文本路径与 HF `Gemma3nTextDecoderLayer` 对齐。

| 项 | 约定 |
|----|------|
| 层型 | 重复 4×sliding + 1×full（full 在 4,9,…,n−1）；`sliding_window=512` |
| RoPE | sliding θ=1e4 / full θ=1e6，**全头**（禁止 p-RoPE / `partial_rotary_factor`） |
| RMSNorm | ones-init `*w`（`scale_plus_one=False`）；禁止 Gemma-3 `*(1+w)` |
| Attn | QK-norm + V-norm（无 scale 时等价 ones）；scale **`1.0`**（不是 `1/sqrt(head_dim)`） |
| 其它 | embed `sqrt(H)`；`final_logit_softcapping=30`；tied embed；KV-share 为最后 `num_layers−20` 层（E2B 30→10，E4B 35→15；禁止把 E2B 填成 15）；`head_dim=256` |
| AltUp | 4 条残差流；`router_input_scale=1/hidden`（**不是** `1/√H`，否则 tanh 饱和、greedy 会吐 `<bos>`/`<pad>`）；加载 `altup_projections` / `altup_unembed_projections` 与每层 `altup.*`；真实 E2B（hidden≥1024）缺则硬失败 |
| Laurel | 每层 `linear_left`/`linear_right`/`post_laurel_norm`；与 attn 残差 `(attn_gated + laurel)/√2` |
| PLE | 与 Gemma-4 同名张量；**加到 AltUp 非 active 流**（`corrected[1:] += delta`），禁止只加到 stream 0 |
| FFN | 前 10 层 `activation_sparsity=0.95` 的 gaussian top-k（`relu(x − (μ + σ Φ^{-1}(0.95)))`）再 GeGLU |
| Chat | Gemma-3 `<start_of_turn>`（不是 Gemma-4 `<\|turn>`）；Hello `prompt_tokens=10` |

禁止把 Gemma-3n 当普通 Gemma-3 decoder（Hello greedy 会变成多语言乱码）。

### 3.4 `aria-openai`

- `serve(bind, model_dir)`（axum）。可选 `router` URL 供注册。
- `POST /v1/chat/completions`：非流式 JSON + `stream: true` 时 SSE。**始终本地 decode**。
- `GET /v1/models`：本地 bundle 目录名。
- 请求/响应字段对齐 OpenAI Chat Completions 常用子集（`messages`、`temperature`、`max_tokens`、`stream`）。`max_tokens` 可选：设置则本地 decode 原样使用；未设置时本地不填默认 16（decode 至 stop 或剩余 context）。
- **删除** `GET /v1/engine/routes`（路由观测在 `aria-router`）。
- **CLI（`aria-engine`）**
  - 缓存根：`~/.ariacompute/`（`engine.yml` + `models/<model>/`）。读配置优先 `engine.yml`，缺则回退旧 `config.yml`（不自动改名）。写只写 `engine.yml`。
  - 子命令：`setup [--status|--clear]`、`download <model>`、`list`、`check [model]`、`clean [model]`、`upgrade [version]`、`serve <model> [--bind] [--compute] [--profile] [--router] [--router-api-key]`；`-h` / `-v`。无 `auth` 别名。
  - `list`：只扫描 `~/.ariacompute/models/` 本地缓存，标记 `downloaded` / `incomplete`。**不**请求 Dashboard catalog。
  - `check [model]`：对照**本区**公开 hub（与 `download` 相同：`.com`→Hugging Face，`.cn`→ModelScope）校验本地 bundle **文件数目、文件名、SHA-256**。指定 model（缓存名或现存路径）则查一项；省略则扫描 `~/.ariacompute/models/` 下全部目录。Hub 清单取 `{sdk}/{bundle}/` 下普通文件（默认 `sdk=v1.0`），跳过 `.gitattributes` / `.gitignore` / 点文件。大文件 SHA-256 来自 hub 元数据（HF `lfs.oid` / ModelScope `Sha256`），**禁止**为校验再拉取 `weight.bin`。逐文件打印 `OK` / `MISSING` / `EXTRA` / `MISMATCH`；任一失败进程 exit 1。不访问 Dashboard、不对区 hub。
  - `upgrade [version]`：按 `upgrade_url`（组织根）拼 `{upgrade_url}/engine`，调 GitHub/Gitee Releases API；默认最新**正式** Release（忽略 prerelease/draft），可选 `0.7.2` / `v0.7.2`；下载本机平台 `aria-engine_*` + `libaria-engine_ffi_*`，原地原子替换当前 CLI，并将 FFI 装入 `~/.ariacompute/lib/`（提示 `ARIA_FFI_LIB`）。未配置 `upgrade_url` 时报错并提示先 `setup`；下载/解压失败不得损坏现有 CLI。
  - `serve <model>`：若为现存路径则用之，否则 `~/.ariacompute/models/<model>`；CLI 旗标仅覆盖本进程，不回写 config。`--compute auto|cpu|cuda` 覆盖本机算力（默认 config / `auto`）。`--profile` 启用加载/生成分段计时，经 `GET /v1/engine/profile` 读出。`--router <url>` 覆盖本进程 router 管理面地址（默认 config / 空）。`--router-api-key` 覆盖 `router_api_key`（Dashboard 签发；注册时 `Authorization: Bearer`）。listen 成功且 URL 非空时 `PUT {router}/v1/router/providers` 注册本地模型；失败 **退出**（禁止假装已接入）。
  - **禁止** `ARIA_HYBRID_*` 环境变量；仅保留编译期 `ARIA_ENGINE_VERSION`。
  - **下载源**：`aria-engine download` **仅**本区公开 hub（`.com`→Hugging Face，`.cn`→ModelScope）。禁止探测/回退对区 hub。**禁止**引擎直连公开 S3/COS registry URL。模型 `download` 与 CLI `upgrade` 宿主（GitHub/Gitee）分离。
  - 每次 `download` 对本区 hub 做连通性探针（用于日志速率），失败则报错退出。不持久化强制源。
  - HF/MS 布局对齐 `serve/scripts/upload-model-hub.sh`：`{sdk}/{bundle}/{file}`，默认 `sdk=v1.0`。
  - Bundle 名解析：`*_q4`→int4、`*_q8`→int8、`*_q326`/`*_q3.26`→int326；可选 `_channel` / `_group`（如 `*_q326_channel`）仍映射同一 quant。`serve` / `download` 已存在检查亦走该别名。否则整名 + 默认 int4。`infer_family_path` 须剥掉上述后缀再对照 §1.1；禁止把 `qwen3-0.6b_q326_channel` 回落成 `gemma/gemma-4-e2b-it` 并误要 PLE。

### 3.4.1 Config（`~/.ariacompute/engine.yml`）

| 字段 | 含义 |
|------|------|
| `router` | 可选 `aria-router` 管理面 URL（空 = 不接入） |
| `router_api_key` | 可选；**Local** Dashboard Keys 签发的 `sk-aria_…`；注册时 Bearer（空则不带） |
| `serve_site` | 可选；`intl` / `cn` 或站点 URL（OAuth / Aria Compute） |
| `serve_api_key` | 可选；OAuth `bfvk-…`（**不得**写入 `router_api_key`）；本增量仅持久化 |
| `site_url` | 站点（`.com` 或 `.cn`），用于 hub 分区 |
| `upgrade_url` | 组织根（`.com`→`https://github.com/ariacompute`；`.cn`→`https://gitee.com/ariacompute`）；`upgrade` 拼 `/engine` |
| `compute` | `auto` \| `cpu` \| `cuda`（默认 `auto`） |
| `hf_token` | Hugging Face hub token（可选，默认空；`.com` 需授权文件） |
| `modelscope_api_token` | ModelScope hub token（可选，默认空；`.cn` 需授权文件） |

**删除**：`cloud_api_key`、`cloud_url`、`hybrid_mode`、`hybrid_execution`、`hybrid_semantic`、`hybrid_semantic_timeout_ms`、`hybrid_semantic_cache_size`。

- `setup`：**分段** `[1/2] Local (router registration)`（router URL + `sk-aria_`）与 `[2/2] OAuth (Aria Compute)`（`serve_site` + `bfvk`）；前缀互斥校验；hub/compute 另段。`--status` 分组脱敏。`--clear` 删除 `engine.yml`。无 `auth` 子命令。
- `download` 公开 hub 与 site 同区（`.com`→HF，`.cn`→ModelScope）。

### 3.4.2 compute（仅本地 GEMM）

`compute`（`auto` \| `cpu` \| `cuda`）只作用于 **Local decode** 的 GEMM。**不是** 路由开关。路由在 `aria-router`。

### 3.5 路由（已上移）

`aria-hybrid` crate **删除**。原 Local/Cloud handoff、`route_hybrid`、Pareto、`CloudClient`、`GET /v1/engine/routes` 均不再属于本仓。Mixture-of-Models 见 **router** 仓 semantic/agent 网关。engine `serve` 仅本地；可选向 router 注册。

### 3.6 错误类型

统一 `EngineError`（或等价枚举，各 crate `thiserror` / `From` 贯通）：

| 变体 | 含义 |
|------|------|
| `Io` | 文件 / 网络 IO |
| `Format` | bundle 损坏、缺字段、offset 越界、`format` 不匹配 |
| `ShapeMismatch` | 算子维度不匹配 |
| `Quant` | 不支持的 bits / 码本布局 / pack 错误 |
| `UnsupportedFamily` | 未注册或未实现的家族 |
| `Upstream` | 可选 router 注册失败 |
| `InvalidParam` | 参数越界（max_tokens=0、非法 `compute` 等） |
| `Unsupported` | 未实现算子 / 阶段未开通的 API |

禁止 `panic` 作为预期错误路径；禁止吞掉错误。

### 3.7 SDK / Bindings（C ABI）

跨语言唯一契约为 **C ABI**（`ariacompute-ffi`：`cdylib` + `staticlib`，头文件 `ffi/include/aria.h`）。禁止嵌入 Cactus C++（§2.1）；本仓自有 ABI 允许。

**入口（OpenAI 面 parity）：**

| C API | 语义 |
|------|------|
| `aria_model_init(path)` | 加载 Aria bundle → opaque handle。未显式 `.family()` 时按目录名 `infer_family_path`（与 `serve` 相同）；禁止把 `gemma-3-1b-it_q326` 回落成 `gemma/gemma-4-e2b-it` 并误要 PLE |
| `aria_complete(…, messages_json, options_json, tools_json, out, …)` | chat；可选 tools；JSON 出参 |
| `aria_complete_stream(…, callback)` | 流式 token/chunk 回调 |
| `aria_embed(…, input_json, out)` | embeddings |
| `aria_transcribe(…, pcm, len, options_json, out)` | ASR |
| `aria_model_destroy` / `aria_last_error` | 生命周期 / 错误 |

**语言包：** Python、Go、Rust（`ariacompute-engine`）、Swift、Kotlin、Flutter、React Native（npm `@ariacompute/engine-rn`）、TypeScript（npm `@ariacompute/engine-ts`）。布局：`ffi/` + `bindings/<lang>/` + `bindings/testdata/`。

**按模型名下载：** 与 `aria-engine download` 相同，仅本区公开 hub（`.com`→Hugging Face，`.cn`→ModelScope）。**不**请求 Dashboard zip meta。Hub 凭证字段与 `aria-engine setup` 相同：`hf_token`（`.com`）/ `modelscope_api_token`（`.cn`）。调用时可显式传入（含 `Engine.setup` 实例内存）；未传则读 `~/.ariacompute/engine.yml`（缺则回退旧 `config.yml`）。**不**读环境变量 `HF_TOKEN` / `MODELSCOPE_API_TOKEN`。Dashboard `sk-`/`bfvk-` token 不作为 hub Bearer。公开模型无需 token。缓存 `~/.ariacompute/models/{model}`；已有有效 bundle 则跳过下载。

**实例 `setup`：** 八语言 SDK 在 Engine 实例上提供 `setup` / `setup_status` / `setup_clear`。字段与 §3.4.1 相同（`router`、`router_api_key`、`site_url`、`upgrade_url`、`compute`、`hf_token`、`modelscope_api_token`）。仅内存、部分合并、非法枚举报错且不改状态。**禁止**写入 `engine.yml`（CLI `aria-engine setup` 仍写该文件）。实例字段优先于 yml；空字段下载仍可读 yml。支持空构造 → `setup` → `open`。`setup_clear` 只重置该实例。无 `auth` 别名。

**libaria-engine_ffi：** SDK 加载模型前须保证本机动态库可用。解析顺序：`ARIA_FFI_LIB`（若已设置则用之）→ 语言包捆绑路径 → `~/.ariacompute/lib/`（与 `aria-engine upgrade` 相同目录）。若均不存在，从本区 Releases 下载最新正式版 `libaria-engine_ffi_{ver}_{os}.tar.gz`（engine.yml `upgrade_url` 优先；否则 `.com`→`https://github.com/ariacompute`，`.cn`→`https://gitee.com/ariacompute`），解压到 `~/.ariacompute/lib/`。已缓存则跳过。失败须明确报错，禁止静默。Rust 原生 `Engine::open` 不 dlopen 该库，但仍走同一安装路径以便其它绑定复用。`aria-engine download` / `serve` 为原生二进制，不经过此路径。

**测试：** 共享 `cases.json`（lifecycle / chat / stream / tools / embed / ASR）；`cargo test -p ariacompute-ffi`；`./scripts/run-binding-tests.sh`。Flutter/RN：iOS+Android device-farm/emulator CI（`.github/workflows/bindings-mobile.yml`）。

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

**张量 `kind == "raw"`**：`dtype` `f16`\|`f32`、`shape`、`offsets.data`。含 1D 范数权重，以及 **0-d / `[1]` 标量**（Gemma-4 `layers.{i}.layer_scalar`，无 `.weight` 后缀；见 §3.3.2）。

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

1. Cargo workspace 可 `cargo build` / `cargo test`（x86_64 scalar）。
2. 加载 **`gemma/gemma-4-e2b-it` tiny q4** Aria bundle（由 `model` 仓 `quantize.py --tiny --bits 4` 产出，产物不入本仓 Git；测试可用生成夹具或路径环境变量）。
3. 反量化与 Python `dequantize` 参考在约定误差界内（RMSE 上界写入单测常量，与 `model` 高斯/tiny 实践同量级）。
4. Prefill + decode 产出非空 token 序列（greedy 可复现）。
5. `POST /v1/chat/completions` 非流式与 SSE 均可测通。
6. 端云 / Mixture-of-Models 路由在 **router** 仓；本仓 chat 仅本地。可选 `router` URL：`serve` 向网关 `PUT /v1/router/providers` 注册，失败退出。
7. §1.1 注册表在代码中完整列出；非阶段 A 家族返回明确 `UnsupportedFamily` / 门控，不得 panic。

### 6.3 阶段 B / C（后续验收，写入 Spec 以免范围漂移）

- **B**：每个 text/MoE 家族具备 loader/graph 钩子并通过该类 tiny 或全量文本生成测试。
- **C**：VL/VLA、ASR、embeddings/RAG、tool_calls 的 OpenAI 面；NEON vs scalar 对拍。
- **真实 codebook（与 §3.3.1）**：至少一条非 tiny Gemma-4 / Qwen3 / LFM 路径在 HDM 接通后，层 RMSE 或短生成与 Python `reconstruct_weight` / 参考前向有界；未实现架构族须硬失败而非乱输出。
- **Gemma-4 hub q4 Hello（§3.3.2）**：`gemma-4-e2b-it_q4`、`temperature=0`、chat「Hello」、`prompt_tokens=10` 时续写须为英文问候（如 `Hello! How can I help you today?`），禁止多语言乱码。须加载并应用每层 `layer_scalar`；仅 unpack embed/PLE 不足以通过。
- **Gemma-3n hub q4 Hello（§3.3.3）**：`gemma-3n-e2b-it_q4`、`temperature=0`、chat「Hello」、`prompt_tokens=10` 时续写须为可读英文（问候/帮助），禁止多语言乱码。须走 AltUp+Laurel+3n PLE，禁止当 Gemma-3 `*(1+w)` / 单一 RoPE。

## 7. 目录与依赖约定

```
engine/
  AGENTS.md
  requirements.md
  task.md                 # 审核通过后生成
  README.md
  Cargo.toml              # workspace
  openai/                 # aria-openai
  inference/              # ariacompute-inference
  graph/                  # ariacompute-graph
  kernel/                 # ariacompute-kernel
  bench/                  # Python 引擎对标评测（report-only）
```

- HTTP：**axum**。
- 可选 `router` URL：`~/.ariacompute/engine.yml`；阶段 A 测试用 mock upsert。
- CLI 下载：仅本区 hub（`.com`→HF，`.cn`→ModelScope）；无 Dashboard/对区回退；无公开 S3 客户端；`DownloadSource` 仅 HuggingFace | ModelScope。
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

### 8.7 本机热点报告（`aria-engine --profile`）

Report-only JSON（`GET /v1/engine/profile` 或 `scripts/profile_qwen3_serve.py`）：

- **load**：`mmap_ms`、`dequant_ms`、`unrotate_ms`、`materialize_ms`、`cuda_upload_ms`（无 CUDA 则 0 / omit）
- **generate**：`prefill_ms`、`decode_ms`、`gemm_attn_ms`、`gemm_ffn_ms`、`gemm_lm_head_ms`
- 顶层：`compute`（cpu/cuda）、`simd` / device 名、`ci_fail: false`
- 无 NVIDIA / 未链上 cuBLAS：`compute=cpu`；显式 `--compute cuda` 失败不得静默降级

## 9. 审核检查表

- [x] §1.1 与 `model/requirements.md` §1.1 家族列表一致可接受
- [x] 仅 `aria-quant-bundle`、禁止 GGUF 可接受
- [x] 阶段 A 黄金路径 = `gemma-4-e2b-it` tiny q4 可接受
- [x] 全家族在 Spec 内、E2E 分 A/B/C 可接受
- [x] OpenAI：阶段 A 仅 chat(+SSE)/models；ASR/RAG/Tool 属阶段 C 可接受
- [x] Hybrid：阶段 A mock + config/`CloudClient::new` 可接受
- [x] CLI：`~/.ariacompute` + auth/download/list/check/clean/upgrade/serve；无 `ARIA_HYBRID_*`；本区 hub（HF/MS）探针下载
- [x] Kernel：NEON/AVX2 CPU + 可选 CUDA；`compute=auto` 与 hybrid 正交可接受
- [x] 反量化 = rotated-space；embedding 原域 gather；线性层 HDM 或原域 linear 可接受
- [x] 五 crate 命名与目录映射可接受
- [x] 非目标（不动 model、无 Metal/计费、无 Cactus FFI）可接受
- [x] 验收门禁（`cargo test`、单测覆盖正常/异常）可接受
- [x] §8 引擎对标评测（JSON+MD；aria/llamacpp/ollama/vllm；性能+质量；全家族；`bench/` Python）可接受
- [x] §8.7 本机热点 profile（load/generate 分段；`--compute`；无 GPU skip）可接受
- [x] §3.7 SDK bindings（C ABI + 八语言；OpenAI FFI 面；host/device-farm 测；release.yml 多 registry 发布 fail-pass）可接受
- [x] §3.7 PyPI 发布方案 B（cibuildwheel 多平台 wheel + 内嵌动态库 + `_load_lib` 回退 + `publish-pypi.yml`）可接受
- [x] §3.5 P2 规则+语义两层混合路由（`route_hybrid`；语义层失败静默回退；健康回退链；`/v1/engine/routes` 观测；无新增环境变量）可接受
- [x] §3.3.2 Gemma-4 `layer_scalar`（HF/JAX `skip_scale`；raw 0-d/`[1]`；层末乘残差；缺省 1.0）与 hub `gemma-4-e2b-it_q4` Hello 验收可接受

> **人工审核状态**：2026-08-02 **已通过（approved）**。§8 增补经 2026-08-03 用户锁定范围 **已通过**。§3.7 SDK bindings 按 2026-08-15 计划实施。§3.7 PyPI 发布方案 B 经 2026-08-19 用户确认 **已通过（approved）**。§3.5 P2 两层混合路由经 2026-08-21 用户确认 **已通过（approved）**。可据本 Spec 生成 / 执行 `task.md`。

## 10. 参考

- Cactus：<https://github.com/cactus-compute/cactus>；Bindings：<https://docs.cactuscompute.com/latest/#bindings>
- `model` 契约：`model/common/bundle.py`、`pack.py`、`quant.py`、`hadamard.py`
- `model` Spec：`model/requirements.md`；质量审计：`model/common/audit_cli.py`
- 调研笔记：桌面 `cactus_compute_research.md`；PPT《技术架构-大模型》
