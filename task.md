# task.md — aria engine 实施清单

依据已审核通过的 [`requirements.md`](requirements.md)。完成后勾选。

## 阶段 A（MVP）

### T0 — Workspace 脚手架
- [x] 根 `Cargo.toml` workspace；五 crate：`aria-kernel` / `aria-graph` / `aria-inference` / `aria-hybrid` / `aria-openai`
- [x] 共享 `EngineError`（`aria-kernel`）
- [x] `cargo build` / `cargo test` 可运行
- [x] 更新 `README.md` 指向 AGENTS / requirements / 常用命令

### T1 — `aria-kernel`（scalar + NEON 入口）
- [x] `SimdMode::{Scalar, Neon}`
- [x] `matmul` / `rms_norm` / `softmax` / `rope` / `attention` / `swiglu` / `fwht` / `dequant_lookup`
- [x] 正常 + `ShapeMismatch` 单测；测试强制 Scalar

### T2 — `aria-graph`
- [x] `TensorView` / `BufferPool` / `Graph` / `Op` / `execute`
- [x] external / mmap 视图（零拷贝借用）
- [x] HDM 融合 op 调度（`HdmLinear` → linear）
- [x] 单测：dispatch + 维度错误

### T3 — `aria-inference`（bundle + session + 家族）
- [x] `load_bundle`：`aria-quant-bundle` v1|v2；mmap `weight.bin`；codebook/raw
- [x] LSB unpack 1–4 / u8 for 8；`dequantize` rotated-space
- [x] §1.1 `Family` 注册表完整；非阶段 A → `UnsupportedFamily`
- [x] 测试夹具 tiny q4 bundle（`fixture::write_tiny_q4_bundle`）
- [x] `Session`：prefill + decode，greedy 产出非空 tokens
- [x] 反量化 RMSE 有界单测

### T22 — Blocked Hadamard（与 model 协议 B）
- [x] kernel：`pow2_tile` + portable signs + blocked FWHT forward/inverse
- [x] bundle：`format_version` 1|2；fixture v2 `mode=blocked`
- [x] graph：`HdmLinear` = Gemm(W_rot) 后对 out 维 blocked unrotate
- [x] 单测：kernel golden/edge、HDM batch、bundle v1|v2|reject-v3；README

### T4 — `aria-hybrid`
- [x] `Router` + 置信度阈值 → `Local` / `CloudHandoff`
- [x] `CloudClient`（reqwest）+ `ARIA_HYBRID_CLOUD_API_KEY`
- [x] mock：成功 / 超时或非 2xx → `Cloud`

### T21 — `aria-hybrid` P0/P1（信号路由）
- [x] P0：`RouteDecision`（reason / policy_version / fallback）+ `ParetoMode` + 硬约束 + 会话粘性 + `RouteOutcome`/`OutcomeStore`
- [x] P1：`RouteSignal` → `ProjectionBand` → 决策；单测覆盖模式/粘性/投影/Outcome
- [x] `aria-openai` 接线 `route(&RouteSignal)`；`FORCE_CLOUD` / `ARIA_HYBRID_EXECUTION` 集成测仍绿
- [x] 补测：边界阈值、force/modality 无云、会话隔离、serde、Outcome HTTP、Pareto 模式、云不可用降级

### T5 — `aria-openai`
- [x] axum：`GET /v1/models`、`POST /v1/chat/completions`（JSON + SSE）
- [x] 接 hybrid 路由；阶段 A 拒 ASR/embeddings → `Unsupported`
- [x] 集成测试（tower oneshot）

### T6 — 阶段 A 验收
- [x] `cargo test` 全绿（x86_64）
- [x] AGENTS「进行中」指向本文件；阶段 A 项勾选完成

## 阶段 B / C

### T10 — 阶段 B 全文本家族
- [x] `ArchClass` + `graph_hook`；`require_stage_b` / `require_runnable`
- [x] 各 text/MoE 代表路径 tiny 生成测试（Gemma/Qwen/LFM/MoE/Nanbeige/Bonsai/Inkling）
- [x] Session 按家族挂载共享 text decoder（MoE 用 stub hook）

### T11 — 阶段 C 多模态 / Agent
- [x] VL `vision_prefix` / VLA `predict_action` + 单测
- [x] OpenAI：`/v1/audio/transcriptions`、`/v1/embeddings`、chat `tool_calls`、`rag_snippets`
- [x] NEON vs scalar matmul 对拍（`matmul_blocked` / `SimdMode::Neon`）
- [x] hybrid `ARIA_HYBRID_EXECUTION=device|cloud`（集成测试）

## 阶段 D — 引擎对标评测（requirements §8）

### T20 — `bench/` Python harness
- [x] §1.1 全家族注册表（与 model EXPECTED 锁表）
- [x] 后端适配：`aria` / `llamacpp` / `ollama` / `vllm`（OpenAI chat）
- [x] 性能：latency p50/p95、tok/s、可选 TTFT；质量：token_overlap + exact_match vs ref
- [x] CLI：`python -m bench run` → JSON + MD；缺后端 skip；默认 exit 0
- [x] `unittest` mock；更新 AGENTS / README

## CLI redesign（config + 下载源）

### T30 — Spec / harness
- [x] `requirements.md`：CLI、config.yml、三源探针、无 `ARIA_HYBRID_*` / 无公开 S3
- [x] 本文件 T30–T34 清单

### T31 — `~/.ariacompute` + auth
- [x] `config.rs`：路径、YAML roundtrip
- [x] `gateway_detect.rs`：locale + 连通性 → `cloud_url` / `site_url`
- [x] `auth` / `--status` / `--clear`

### T32 — download / list / clean
- [x] 探针 + 自动选 Dashboard / HF / ModelScope；失败回退
- [x] `list` / `clean [model]`；bundle 校验

### T33 — serve CLI + CloudClient
- [x] `serve <model> [--bind] [--hybrid-mode] [--hybrid-execution]`
- [x] `CloudClient::new`；删除 `from_env` 与 `ARIA_HYBRID_*`

### T34 — Docs / 测试
- [x] engine + serve 文档；`cargo test` / clippy；serve Go 下载 API-key + JSON 测

## SDK bindings（C ABI + 八语言）

### T40 — Spec
- [x] `requirements.md` §3.7；本文件 T40–T48；`AGENTS.md` 目录 / 命令

### T41 — `aria-ffi`
- [x] C ABI：init / complete / stream / embed / transcribe / destroy / last_error
- [x] `ffi/include/aria.h`；`cargo test -p aria-ffi`

### T42 — Language scaffolds
- [x] `bindings/{python,go,rust,swift,kotlin,flutter,react-native,typescript}`

### T43 — Host test matrix
- [x] `bindings/testdata/` + `cases.json`；`scripts/run-binding-tests.sh`

### T44 — Device-farm CI
- [x] `.github/workflows/bindings-mobile.yml`（Flutter/RN iOS+Android）

### T45 — Release publish
- [x] `release.yml` 发布 Maven/CocoaPods/npm/pub.dev/crates.io/PyPI（fail-pass）

### T46 — Engine docs
- [x] README / README_cn Bindings + install

### T47 — Serve UI
- [x] 主页 + 文档 Tab 示例（全站语言）

## CLI upgrade + Serve Download

### T50 — Spec
- [x] `requirements.md` §3.4 / §3.4.1：`upgrade` / `upgrade_url`；本文件 T50–T54；`AGENTS.md` 命令
- [x] serve：`/download` + 站点→GitHub/Gitee 映射（README）

### T51 — Config + auth
- [x] `AriaConfig.upgrade_url`；`GatewayPair::upgrade_url()`；auth / reconcile / `--status`

### T52 — `aria-engine upgrade`
- [x] 解析正式 Release；下载 CLI + `libaria_ffi`；原子安装到 `current_exe` + `~/.ariacompute/lib/`

### T53 — Serve Download page
- [x] `GET /api/download/engine-latest`（按 Host 选源 + 短缓存）
- [x] `/download` 前端 + 13 locale + 导航 / SEO

### T54 — Tests / docs
- [x] Engine 单测 + README；Serve handler 测 + README `/download`

## 全家族真实 Bundle 对齐（§3.3.1）

### T60 — Spec
- [x] `requirements.md` §3.3.1 + §6.3 真实 codebook；本文件 T60–T69；`AGENTS.md` 进行中

### T61 — Session HDM
- [x] Session 线性层走 `HdmLinear`（或等价 unrotate）；单测 rotated vs 原域

### T62 — RoPE `rotate_half`
- [x] 与 HF 布局对齐；partial RoPE 预留；单测

### T63 — Gemma 正确性
- [x] GeGLU + RMSNorm `(1+w)` + 四 norm 别名；`ffn_norm` 优先 `pre_feedforward_layernorm`
- [ ] Gemma-4 KV **cache** 复用 / 滑动·全局 / 双 RoPE / PLE 门控（未做；缺 k/v 仍 clone 先验权重）

### T64 — QK-Norm
- [x] 别名 + op；Qwen3 / Gemma / LFM attn 路径

### T65 — LFM conv hybrid
- [x] `conv.*` 名 + short-conv 层 + per-layer cache；不强制每层 `q_proj`
- [x] `layer_types` 混合 `conv` / `full_attention` 可 generate（单测）

### T66 — MoE
- [x] LFM2-8B-A1B + Inkling：router + top-k expert FFN；`text_moe_decoder`；Inkling ArchClass=`TextMoE`
- [x] 无 `num_experts` 的 MoE 家族仍硬失败（禁止 dense stub 冒充）
- [x] 单测：4-expert / top-2 generate

### T67 — Qwen3.5 / Bonsai
- [x] 家族路径硬 `Unsupported`（禁止当全 dense）；`layer_types` linear_attention/delta 同样门控
- [ ] DeltaNet / linear_attention 真实现

### T68 — VL / VLA
- [x] `vision_prefix` / `predict_action` 硬 `Unsupported`（去掉 RGB mean-pool / 假 action）
- [ ] 消费 bundle 内 vision/action 张量

### T69 — Registry + model 字段
- [x] 登记 `lfm2.5-2.6b`；消费扩展 `model` 元数据（`head_dim` / `layer_types` / `hidden_act` / MoE / `conv_l_cache`）
