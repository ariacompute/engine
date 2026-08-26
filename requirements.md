# requirements.md — aria engine（Rust）

> 本文件为 `engine` 仓库 **OpenAI 兼容 API / 全家族推理 / 零拷贝计算图 / ARM NEON 内核 / Hybrid 路由** 的功能边界、API、数据布局、异常与验收标准。**须经人工逐项审核**，审核通过后方可据其生成 / 执行 `task.md`。
>
> 架构参考：Cactus Engine / Graph / Kernels / Hybrid；Aria《技术架构-大模型》Engine / Graph / Kernels；权重契约对齐 `model` 仓 `aria-quant-bundle` v1。

## 1. 目标与范围

用 **Rust** 实现端侧推理引擎，消费 `model` 仓导出的 **Aria model bundle**，提供本地推理与 OpenAI 兼容 HTTP，并在置信度不足时路由至云端。

- **五层产品面**：`openai` / `inference` / `graph` / `kernel` / `hybrid`（与仓库目录 1:1，Cargo workspace crate：`aria-openai` / `ariacompute-inference` / `ariacompute-graph` / `ariacompute-kernel` / `aria-hybrid`）。
- **权重格式**：仅 `aria-quant-bundle`（`format_version: 1|2`）— `config.json` + `weight.bin` + tokenizer 侧车；**禁止**解析 / 导出 GGUF 与 Cactus 专有权重格式。
- **位宽**：消费 `q1`–`q4` / `q8` / `q1.5` / `q2.54` / `q3.26`（与 `model` 产物一致）。
- **模型覆盖**：§1.1 全部家族（与 `model/requirements.md` §1.1 对齐）。
- **本机算力**（与 hybrid 路由正交）：`compute`=`auto`\|`cpu`\|`cuda`。CPU = `SimdMode::{Scalar, Neon, Avx2}`（`aarch64` Neon；x86_64 AVX2+FMA 若可用否则 scalar + 多线程 `linear`）。可选 **CUDA**（NVIDIA + cuBLAS SGEMM）；`auto` 探测到可用 CUDA 则用之，否则 CPU。`hybrid_execution=device` **不是** GPU 开关，仅禁止云 handoff。
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
| **A（MVP）** | 骨架 + 黄金路径 | workspace、五 crate、scalar kernel、graph 调度、bundle 加载、tiny 文本 LLM E2E、OpenAI chat(+SSE)、hybrid mock |
| **B** | 全文本家族 | §1.1 全部 text/MoE 家族可加载并文本生成（可用 tiny fixture 按架构类验收后再接全量权重） |
| **C** | 多模态 / Agent 面 | VL / VLA、ASR、embeddings/RAG、tool_calls；NEON 精度对拍；on-device-only 锁 |

### 1.2 交付风格

完整产品 Spec（本文件覆盖 openai / inference / graph / kernel / hybrid 全能力）；**验收按阶段 A→B→C** 切开。未达阶段的能力须在代码中以明确 `Unsupported` / 功能门控失败，禁止静默空实现冒充完成。

## 2. 功能边界

| # | 特性 | 实现深度 |
|---|------|----------|
| 1 | **kernel** | matmul、attention、RMSNorm、RoPE、Softmax、SwiGLU、码本查表反量化、FWHT；`SimdMode::{Scalar, Neon, Avx2}` + `ComputeBackend::{Cpu, Cuda}`；`linear`/`attention` 经 backend 分发；CUDA 与 scalar 相对误差有界；无 GPU 时 `--compute cuda` 硬失败（禁止静默 CPU） |
| 2 | **graph** | Layer → Op → Tensor IR；`BufferPool`；mmap / `external` 零拷贝输入；融合节点 **HDM**（Hadamard + Dequant + MatMul）；阶段 A：LLM decode 所需 ops 可调度；图序列化可选（阶段 B+） |
| 3 | **inference** | `load_bundle`（mmap `weight.bin`）；KV cache；greedy / 基础 sampling；tokenizer；§1.1 家族注册与图构建钩子；阶段 A：`gemma-4-e2b-it` **tiny q4** 黄金路径；阶段 B/C：见上表 |
| 4 | **openai** | `POST /v1/chat/completions`（含 SSE streaming）、`GET /v1/models`；阶段 C：`/v1/audio/transcriptions`、`/v1/embeddings`、tool_calls / RAG 编排；CLI：`auth` / `download` / `list` / `clean` / `upgrade` / `serve` |
| 5 | **hybrid** | 信号→投影→决策：`RouteSignal` → `ProjectionBand` → `RouteDecision{action, reason, policy_version, fallback}`；Pareto 模式 `Cost`/`Balance`/`Intelligence`；会话粘性与失败升级；`RouteOutcome` 内存落盘；云端 OpenAI 兼容 POST；凭证与模式来自 `~/.ariacompute/config.yml`（`CloudClient::new`）；`execution`=`hybrid`\|`device`\|`cloud`；阶段 A：mock + 单测；`device` / privacy 强制 Local，`cloud` 强制 Handoff；**P2**：规则路由层（快路径）+ 语义路由层（慢路径）两层混合路由 `route_hybrid`，语义失败静默回退规则层 |
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

- `serve(bind, model_dir, hybrid_opts)`（axum）。
- `POST /v1/chat/completions`：非流式 JSON + `stream: true` 时 SSE。
- `GET /v1/models`：按 `hybrid_execution` 列出模型 id——`cloud` 仅 `ariacompute/ariamodel`；`device` 为本地 bundle 目录名；`hybrid` 为本地目录名，若已配置 cloud 凭证则另附 `ariacompute/ariamodel`（不再用内部家族 path 如 `gemma/gemma-4-e2b-it` 冒充云端 id）。
- 请求/响应字段对齐 OpenAI Chat Completions 常用子集（`messages`、`temperature`、`max_tokens`、`stream`）。`max_tokens` 可选：设置则本地 decode / 云 handoff 原样使用；未设置时本地不填默认 16（decode 至 stop 或剩余 context），云端省略该字段。
- `GET /v1/engine/routes`（P2 观测端点，只读）：返回最近 N 条路由 `RouteOutcome`（含 `layer` / `confidence` / `semantic_consulted`）+ Local/Cloud 健康分快照 + `policy_version` + **生效** `execution` / `mode`（非 hybrid 为 `unused`）/ `compute` / `cloud_available` / `semantic.enabled`（配置开关）/ `semantic.applicable`（真正会咨询）；`?n=` 可选（默认 20，上限 100）。
- **CLI（`aria-engine`）**
  - 缓存根：`~/.ariacompute/`（`config.yml` + `models/<model>/`）。
  - 子命令：`auth [--status|--clear]`、`download <model>`、`list`、`clean [model]`、`upgrade [version]`、`serve <model> [--bind] [--hybrid-mode] [--hybrid-execution] [--hybrid-semantic on|off] [--compute] [--profile]`；`-h` / `-v`。
  - `list`：`GET {site_url}/api/dashboard/models`（Bearer `cloud_api_key`）展开为可下载 bundle（`_q4`/`_q8`/`_q326`），对照本地缓存标记 `downloaded` / `not downloaded` / `incomplete`；另附 catalog 外本地项。catalog `*_q326` 与本地 `*_q326_channel` / `*_q326_group`（及 `*_q3.26*`）视为同一 int326 缓存，禁止把已下载的 channel 配方标成 `not downloaded`。
  - `upgrade [version]`：按 `upgrade_url`（组织根）拼 `{upgrade_url}/engine`，调 GitHub/Gitee Releases API；默认最新**正式** Release（忽略 prerelease/draft），可选 `0.7.2` / `v0.7.2`；下载本机平台 `engine_*` + `libaria_ffi_*`，原地原子替换当前 CLI，并将 FFI 装入 `~/.ariacompute/lib/`（提示 `ARIA_FFI_LIB`）。未配置 `upgrade_url` 时报错并提示先 `auth`；下载/解压失败不得损坏现有 CLI。
  - `serve <model>`：若为现存路径则用之，否则 `~/.ariacompute/models/<model>`；CLI 旗标仅覆盖本进程，不回写 config。`--compute auto|cpu|cuda` 覆盖本机算力（默认 config / `auto`）。`--profile` 启用加载/生成分段计时，经 `GET /v1/engine/profile` 读出。`--hybrid-semantic on|off` 覆盖语义路由层开关（默认 config / `on`）。
  - **禁止** `ARIA_HYBRID_*` 环境变量；仅保留编译期 `ARIA_ENGINE_VERSION`。
  - **下载源**：`aria-engine download` **仅**本区公开 hub（`.com`→Hugging Face，`.cn`→ModelScope）。禁止探测/回退对区 hub，禁止 Dashboard 与 hub 按速率择优或拉取失败互退。Dashboard 仅用于 `list` 目录与 `auth` 区探测。**禁止**引擎直连公开 S3/COS registry URL。模型 `download` 与 CLI `upgrade` 宿主（GitHub/Gitee）分离。
  - 每次 `download` 对本区 hub 做连通性探针（用于日志速率），失败则报错退出。不持久化强制源。
  - HF/MS 布局对齐 `serve/scripts/upload-model-hub.sh`：`{sdk}/{bundle}/{file}`，默认 `sdk=v1.0`。
  - Bundle 名解析：`*_q4`→int4、`*_q8`→int8、`*_q326`/`*_q3.26`→int326；可选 `_channel` / `_group`（如 `*_q326_channel`）仍映射同一 quant。`serve` / `download` 已存在检查亦走该别名。否则整名 + 默认 int4。`infer_family_path` 须剥掉上述后缀再对照 §1.1；禁止把 `qwen3-0.6b_q326_channel` 回落成 `gemma/gemma-4-e2b-it` 并误要 PLE。

### 3.4.1 Config（`~/.ariacompute/config.yml`）

| 字段 | 含义 |
|------|------|
| `cloud_api_key` | Dashboard / hybrid 同一 API key |
| `cloud_url` | Gateway base（`.com` 或 `.cn`） |
| `site_url` | Dashboard site（`.com` 或 `.cn`） |
| `upgrade_url` | 组织根（`.com`→`https://github.com/ariacompute`；`.cn`→`https://gitee.com/ariacompute`）；`upgrade` 拼 `/engine` |
| `hybrid_mode` | `cost` \| `balance` \| `intelligence` |
| `hybrid_execution` | `hybrid` \| `device` \| `cloud` |
| `hybrid_semantic` | `true` \| `false`（默认 `true`；语义路由层总开关，无云凭证时自动短路等价纯规则） |
| `hybrid_semantic_timeout_ms` | 语义路由单次调用超时（默认 `800`） |
| `hybrid_semantic_cache_size` | 语义决策缓存容量上限（默认 `512`；TTL 60s） |
| `compute` | `auto` \| `cpu` \| `cuda`（默认 `auto`；与 `hybrid_execution` 正交） |
| `hf_token` | Hugging Face hub token（可选，默认空；`.com` 需授权文件） |
| `modelscope_api_token` | ModelScope hub token（可选，默认空；`.cn` 需授权文件） |

- `auth`：提示 API key → **用 key 探测** Dashboard（`GET /api/dashboard/models`）判定 `.com` / `.cn`，写入匹配的 `cloud_url`/`site_url`/`upgrade_url`（同 TLD）；两端均失败时回退 locale + 连通性。按区**只提示一项** hub token（`.com` → `hf_token`，`.cn` → `modelscope_api_token`；回车跳过；已有值则保留；另一字段不动）。`list`/`download` 启动时若 URL 不一致或 key 被当前 site 拒绝会自动纠偏并回写 config（含刷新 `upgrade_url`）。
- Gateway / site / upgrade 组织根始终成对（同为 `.com` 或同为 `.cn`）；`download` 公开 hub 与 site 同区且不交叉（`.com`→HF，`.cn`→ModelScope），不与 Dashboard 竞速或互退。

### 3.4.2 四轴合成（compute ⊥ execution）

两层正交，互不覆盖：

- **compute**（`auto` \| `cpu` \| `cuda`）：只作用于 **Local decode** 的 GEMM。**不是** hybrid 开关。`execution=cloud` 时仍加载本地 bundle（隐私硬约束会回本地），成功 handoff 不走 CUDA。
- **execution**（`device` \| `hybrid` \| `cloud`）：允许的后端。`device` = 永不离机；`cloud` = 始终 handoff（不可用则报错，不静默本地）；`hybrid` = 按 mode/semantic 二选一。
- **mode**（`cost` \| `balance` \| `intelligence`）：**仅 `execution=hybrid` 有效**。调节复杂度 cutoff，以及 Chat 策略（Cost 问语义层；Balance/Intelligence 知识类 Chat 直接云）。
- **semantic**：**仅 `execution=hybrid` 且 `cloud_available` 且开关 on 才生效**。只处理规则 `need_semantic` 的请求（Agent / LongContext / 复杂度邻域 / Cost 的 Chat）。问候、硬约束、Balance Chat 直云都不问。

合成后的有效路由（`cloud_available=true`，semantic on）：

| execution | mode | Hello | Introduce/Chat | Agent |
|-----------|------|-------|----------------|-------|
| `device` | 忽略 | Local | Local | Local |
| `cloud` | 忽略 | CloudHandoff | CloudHandoff | CloudHandoff |
| `hybrid` | `cost` | Local | 语义层 | 语义层 |
| `hybrid` | `balance` \| `intelligence` | Local | 规则直云 `rule:chat_prefer_cloud` | 语义层 |

`cloud_available=false` 且 hybrid：全部留本地（含 Chat）；semantic 短路。serve listen 与 `GET /v1/engine/routes` 打印**生效**策略：`semantic=on\|off\|n/a`（配置开但非 hybrid / 无凭证时为 `n/a`，避免误导）。

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
- **P2（规则路由层 + 语义路由层，`route_hybrid`）**
  - **规则路由层（快路径，`rules.rs`）**：`RuleEngine` 确定性规则链，零 LLM 调用；输入 `RequestKind`（由 prompt 分类：`Inline` / `Chat` / `Agent` / `LongContext` / `Media`）+ `RouteSignal` + Pareto 阈值；输出 `RuleDecision{action: Option<RouteAction>, confidence, reason, need_semantic}`。硬约束（`execution` / privacy / `force_cloud` / modality / context_overflow / failures ≥ upgrade）→ 高置信直接决策，**不走语义层**。Chat 策略由 `chat_policy(mode)` 给出（§3.4.2）：Cost → `need_semantic`（`rule:chat_task`）；Balance/Intelligence → 云可用时直接 `CloudHandoff`（`rule:chat_prefer_cloud`）。`Agent` / `LongContext` / 复杂度邻域 → `need_semantic`。仅 Inline 问候（`hi` / `Hello` / `ok` / `谢谢`，无任务词、<80 字）直接 Local。云不可用时 PreferCloud Chat → `rule:chat_prefer_cloud_but_unavailable` 留本地。
  - **Chat / Agent 分类词**（规则只判种类，是否咨询语义层见 §3.4.2；短祈使句也须命中，禁止漏成 Inline）：知识/讲解（introduce / explain / `?` / 介绍）；推理/多步（prove / 证明 / 推导）；代码生成或审查（`write a` / implement / 写一个 / 实现）；数学/STEM（solve / 计算 / 解方程）；翻译/改写/摘要（translate / summarize / 翻译 / 总结）；创作（poem / 诗 / 写一封 / 角色扮演）；对比/建议（recommend / 哪个好 / 选型）；约束格式（as json / schema / 只输出）；专业咨询词（legal advice / 投资建议）——若 `privacy_sensitive` 则硬留本地。≥80 字普通对话同样 `Chat`。
  - **语义路由层（慢路径，`semantic.rs`）**：`SemanticClient` 抽象（生产 `CloudSemanticClient` 复用 `CloudClient::chat`，system prompt 强约束输出严格 JSON `{"action":"local"|"cloud","confidence":0..1,"intent":"…","reason":"…"}`，请求带 `enable_thinking=false` 以免 thinking 网关撑破 800ms；测试注入 fake）；`SemanticRouter` 负责决策缓存（归一化 prompt 哈希 key + TTL 60s + 容量上限淘汰）、同 key 单飞去重、`tokio::time::timeout` 包裹。未启用 / 无云凭证 / 超时 / 非 2xx / JSON 非法或字段越界 → 返回 `None` **静默回退规则层**（禁止报错、禁止 panic、禁止吞错后产生误导性决策）。云端 chat 响应 `model` 写为 `ariacompute/ariamodel`（禁止沿用本地 bundle 名）。
  - **健康度回退链（`health.rs`）**：`HealthTracker` 维护 Local/Cloud 双后端分数（初值 1.0；成功 +0.05 封顶 1.0；失败 −0.20；超时 −0.10；底线 0.0；`healthy` = 分数 ≥ 0.5）；`snapshot()` 供观测端点。仅软决策可翻转：选定后端不健康且备选健康时翻转，Cloud 翻转另须 `cloud_available`；硬约束（`ProjectionBand::MustLocal`）永不翻转。
  - `Router::route_hybrid(&self, signal, prompt, semantic, health).await -> RouteDecision`：规则快路径 →（`need_semantic`）语义慢路径（采纳阈值 `confidence ≥ 0.6` 且不与硬约束冲突）→ 健康翻转 → 复用既有 `apply_stickiness` 粘性。**`route()` / `route_confidence()` 及全部 P0/P1 行为零变化**（既有单测原样通过）。
  - `RouteDecision` 增 `layer`（`rules` \| `semantic`）/ `confidence` / `semantic_consulted`；`RouteOutcome` 增同三字段 + `semantic_latency_ms`；一律 `#[serde(default)]`，既有 serde roundtrip 与对外 JSON 兼容。语义层采纳的决策 `policy_version` 追加 `+semantic`。
  - 观测：复用 `OutcomeStore.recent(n)` + `HealthTracker::snapshot()` + 生效轴（§3.4.2），经 `GET /v1/engine/routes` 暴露（§3.4）。
  - 配置：§3.4.1 三字段 + CLI `--hybrid-semantic on|off`（仅覆盖本进程）；**禁止**新增任何环境变量。
  - 单测：规则链各规则命中 / 邻域触发语义 / 默认回退；语义 JSON 合法·非法·缺字段·越界 / 缓存命中·TTL 过期·容量淘汰·单飞 / 超时与未启用降级；健康分增减·阈值·翻转·MustLocal 不翻转；`route_hybrid` 快路径命中 / 语义采纳 / 语义失败回退 / 健康翻转；`route()` 回归零变化。
- `Router::new() -> Result<Self, EngineError>`；`route(&self, &RouteSignal) -> RouteDecision`（兼容 `route_confidence(f32)`）。
- `CloudClient::new(base_url, api_key)`：OpenAI 兼容 HTTP；超时默认 **60s**（`DEFAULT_CLOUD_CHAT_TIMEOUT_MS`；完整 ariamodel 含 thinking 可 >25s）与非 2xx → `EngineError::Cloud`；handoff 请求 `model` 固定为 `ariacompute/ariamodel`（`CLOUD_GATEWAY_MODEL`）。客户端设置了 `max_tokens` 则原样转发；未设置则省略该字段（禁止填入默认 16：gateway 把 reasoning 计入该预算，会被截成 `finish_reason=length` 且 `content=""`）。**无** `from_env` / `ARIA_HYBRID_*`。
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

跨语言唯一契约为 **C ABI**（`ariacompute-ffi`：`cdylib` + `staticlib`，头文件 `ffi/include/aria.h`）。禁止嵌入 Cactus C++（§2.1）；本仓自有 ABI 允许。

**入口（OpenAI 面 parity）：**

| C API | 语义 |
|------|------|
| `aria_model_init(path)` | 加载 Aria bundle → opaque handle |
| `aria_complete(…, messages_json, options_json, tools_json, out, …)` | chat；可选 tools；JSON 出参 |
| `aria_complete_stream(…, callback)` | 流式 token/chunk 回调 |
| `aria_embed(…, input_json, out)` | embeddings |
| `aria_transcribe(…, pcm, len, options_json, out)` | ASR |
| `aria_model_destroy` / `aria_last_error` | 生命周期 / 错误 |

**语言包：** Python、Go、Rust（`ariacompute-engine`）、Swift、Kotlin、Flutter、React Native（npm `@ariacompute/engine-rn`）、TypeScript（npm `@ariacompute/engine-ts`）。布局：`ffi/` + `bindings/<lang>/` + `bindings/testdata/`。

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

1. Cargo workspace 五 crate 可 `cargo build` / `cargo test`（x86_64 scalar）。
2. 加载 **`gemma/gemma-4-e2b-it` tiny q4** Aria bundle（由 `model` 仓 `quantize.py --tiny --bits 4` 产出，产物不入本仓 Git；测试可用生成夹具或路径环境变量）。
3. 反量化与 Python `dequantize` 参考在约定误差界内（RMSE 上界写入单测常量，与 `model` 高斯/tiny 实践同量级）。
4. Prefill + decode 产出非空 token 序列（greedy 可复现）。
5. `POST /v1/chat/completions` 非流式与 SSE 均可测通。
6. Hybrid：低置信 / PreferCloud 时 `CloudHandoff`（带 `reason`）；Pareto 模式与粘性单测；mock 云端成功与失败路径覆盖。
7. §1.1 注册表在代码中完整列出；非阶段 A 家族返回明确 `UnsupportedFamily` / 门控，不得 panic。
8. Hybrid P2：`route_hybrid` 两层路由单测全绿（规则快路径命中、语义 mock 采纳 / 失败静默回退、健康翻转、硬约束不翻转）；`GET /v1/engine/routes` 返回最近决策（含 `layer`）与健康分快照；无云凭证时行为与纯规则路径完全一致。

### 6.3 阶段 B / C（后续验收，写入 Spec 以免范围漂移）

- **B**：每个 text/MoE 家族具备 loader/graph 钩子并通过该类 tiny 或全量文本生成测试。
- **C**：VL/VLA、ASR、embeddings/RAG、tool_calls 的 OpenAI 面；NEON vs scalar 对拍；`hybrid_execution=device` 禁止云卸载、`=cloud` 强制云端。
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
  hybrid/                 # aria-hybrid
  inference/              # ariacompute-inference
  graph/                  # ariacompute-graph
  kernel/                 # ariacompute-kernel
  bench/                  # Python 引擎对标评测（report-only）
```

- HTTP：**axum**。
- 混合云：`~/.ariacompute/config.yml` 的 `cloud_url` + `cloud_api_key`；阶段 A 测试用 mock。
- CLI 下载：仅本区 hub（`.com`→HF，`.cn`→ModelScope）；无 Dashboard/对区回退；无公开 S3 客户端。
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
- [x] CLI：`~/.ariacompute` + auth/download/list/clean/upgrade/serve；无 `ARIA_HYBRID_*`；三源探针下载可接受
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
