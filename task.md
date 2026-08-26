# task.md — aria engine 实施清单

依据已审核通过的 [`requirements.md`](requirements.md)。完成后勾选。

## 阶段 A（MVP）

### T0 — Workspace 脚手架
- [x] 根 `Cargo.toml` workspace；五 crate：`ariacompute-kernel` / `ariacompute-graph` / `ariacompute-inference` / `aria-hybrid` / `aria-openai`
- [x] 共享 `EngineError`（`ariacompute-kernel`）
- [x] `cargo build` / `cargo test` 可运行
- [x] 更新 `README.md` 指向 AGENTS / requirements / 常用命令

### T1 — `ariacompute-kernel`（scalar + NEON 入口）
- [x] `SimdMode::{Scalar, Neon}`
- [x] `matmul` / `rms_norm` / `softmax` / `rope` / `attention` / `swiglu` / `fwht` / `dequant_lookup`
- [x] 正常 + `ShapeMismatch` 单测；测试强制 Scalar

### T2 — `ariacompute-graph`
- [x] `TensorView` / `BufferPool` / `Graph` / `Op` / `execute`
- [x] external / mmap 视图（零拷贝借用）
- [x] HDM 融合 op 调度（`HdmLinear` → linear）
- [x] 单测：dispatch + 维度错误

### T3 — `ariacompute-inference`（bundle + session + 家族）
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
- [x] `download` 仅本区 hub（`.com`→HF，`.cn`→ModelScope）；禁止 Dashboard 竞速/互退与对区 hub 回退
- [x] `list` / `clean [model]`；bundle 校验
- [x] `list` / `serve` / `download` 将 catalog `*_q326` 与本地 `*_q326_channel` 视为同一缓存
- [x] `check [model]`：对照本区 hub 校验文件数目 / 名称 / SHA-256（不重拉 `weight.bin`）

### T33 — serve CLI + CloudClient
- [x] `serve <model> [--bind] [--hybrid-mode] [--hybrid-execution]`
- [x] `CloudClient::new`；删除 `from_env` 与 `ARIA_HYBRID_*`

### T34 — Docs / 测试
- [x] engine + serve 文档；`cargo test` / clippy；serve Go 下载 API-key + JSON 测

## SDK bindings（C ABI + 八语言）

### T40 — Spec
- [x] `requirements.md` §3.7；本文件 T40–T48；`AGENTS.md` 目录 / 命令

### T41 — `ariacompute-ffi`
- [x] C ABI：init / complete / stream / embed / transcribe / destroy / last_error
- [x] `ffi/include/aria.h`；`cargo test -p ariacompute-ffi`

### T42 — Language scaffolds
- [x] `bindings/{python,go,rust,swift,kotlin,flutter,react-native,typescript}`
- [x] 按模型名下载对齐 CLI hub（`.com`→HF，`.cn`→ModelScope；不走 Dashboard）

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
- [x] Gemma-4 KV **cache** 按 `num_kv_shared_layers` + `layer_types` 同类复用（不 clone `wk`/`wv`）
- [x] Gemma-4 双 RoPE（sliding θ=1e4 / full p-RoPE θ=1e6）+ 四 norm + PLE + embed `sqrt(H)` + attn scale `1.0`
- [x] 滑动窗口 mask（Gemma-4 `sliding_window=512`；只裁 attention，不裁共享 KV cache）
- [x] Gemma-4 `layer_scalar`（HF/JAX `skip_scale`）：无 `.weight` 的 raw 标量，层末乘残差；缺省 1.0；hub q4 Hello 依赖此项
- [x] Gemma-3 文本（270m/1b）：5×sliding+1×full、sliding θ=1e4 / full θ=1e6、hub 缺 `layer_types` 时补齐；GeGLU 不依赖 bundle `hidden_act`
- [x] Gemma-3n E2B/E4B：4×sliding+1×full、双 RoPE 全头、RMSNorm `*w`、attn scale 1.0、logit softcap 30、AltUp+Laurel（`router_input_scale=1/H`）、PLE 加到非 active 流、前 10 层 gaussian top-k；KV-share=`num_layers-20`；真实 hidden≥1024 缺 AltUp/PLE 硬失败

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
- [x] `layer_types` 驱动 Gated DeltaNet 循环 + full attention；无 `layer_types` 仍拒绝 dense 冒充
- [x] 单测：linear_attention + full_attention generate
- [x] 加载 HF Qwen3.5 拆分 `in_proj_qkv`/`z`、`in_proj_b`/`a`（兼容融合 qkvz/ba）
- [x] 全注意力 `attn_output_gate`：`q_proj` 融 query+gate，与 `o_proj` 几何对齐后再 `sigmoid`
- [x] 全注意力 **partial RoPE**（`partial_rotary_factor=0.25`，θ=1e7；hub 缺字段补齐）；`strip_assistant_visible` 在 `</think>` 后无回答时保留 think 正文
- [x] Qwen3.5 RMSNorm 走 Gemma 式 `*(1+w)`（零初始化）；DeltaNet 输出 RMSNormGated 仍 `*w` 并加载 `linear_attn.norm`
- [ ] DeltaNet GQA（`n_v_heads != n_k_heads`）/ chunked prefill

### T68 — VL / VLA
- [x] 消费 bundle `mm_projector` / `action_head` 等张量；缺张量仍 `Unsupported`
- [x] 单测：vision_prefix + predict_action 走真实权重
- [ ] 完整 ViT / SigLIP tower

### T69 — Registry + model 字段
- [x] 登记 `lfm2.5-2.6b`；消费扩展 `model` 元数据（`head_dim` / `layer_types` / `hidden_act` / MoE / `conv_l_cache`）

### T70 — 本机热点 profile
- [x] `Session` 加载 / generate 分段计时（默认关；`--profile`）
- [x] `GET /v1/engine/profile`；`scripts/profile_qwen3_serve.py`

### T71 — `compute=auto|cpu|cuda`
- [x] `ComputePref` / `ComputeBackend`；serve `--compute` + config `compute`
- [x] 探测 NVIDIA+cuBLAS；`--compute cuda` 不可用则硬失败
- [x] serve 日志打印 `compute=` / device 或 simd

### T72 — CPU 加载与 GEMM
- [x] 并行 blocked unrotate；tied embed `Arc` 共享
- [x] 多线程 + AVX2/Neon `linear`；Attn+Dense 层 batched prefill

### T73 — CUDA linear / attention 分发
- [x] 运行时 cuBLAS SGEMM（libloading，默认构建可探测）
- [x] 权重 upload；greedy 与 CPU 对拍；无卡 skip

### T74 — 文档
- [x] README / README_cn / AGENTS：`--compute`、`--profile`、H200 `--features` 说明

## PyPI 发布（方案 B：cibuildwheel + auditwheel 多平台 wheel）

### T75 — `aria-engine` PyPI 发布
- [x] `bindings/python/pyproject.toml`：动态 version 注入点 + `[tool.cibuildwheel]` + `package-data` 含 `lib/*`；`setup.py` `BinaryDistribution` 强制平台 wheel tag
- [x] `aria_engine/_load_lib()`：`ARIA_FFI_LIB` env → 包内 `aria_engine/lib/`（按平台 .so/.dylib/.dll）→ 报错提示；`__version__` 暴露
- [x] `scripts/build-python-ffi.sh`：`cargo build --release -p ariacompute-ffi` + 按平台拷贝动态库进 `aria_engine/lib/`
- [x] `.github/workflows/publish-pypi.yml`：linux x86_64/aarch64（容器内 rustup + auditwheel）+ macos x86_64/arm64 + windows x86_64；版本 = tag 去 `v` sed 注入；`twine check` + `twine upload`（`PYPI_TOKEN`）
- [x] 单测 `tests/test_load_lib.py`：env 优先 / 包内回退（三平台文件名）/ 缺失报错
- [x] 文档：`bindings/python/README.md`（`pip install aria-engine` 即用）、AGENTS 命令、`.gitignore` 忽略 `aria_engine/lib/`
- [x] 验证：本机 `cargo build -p ariacompute-ffi` + `python -m build --wheel` 冒烟（wheel 含 .so + 平台 tag）、`unittest` 全绿、`cargo test` 不回归

## Hybrid P2：规则路由层 + 语义路由层（requirements §3.5 P2，2026-08-21 审核通过）

### T76 — 规则路由层 `hybrid/src/rules.rs`
- [x] `RequestKind`（Inline/Chat/Agent/LongContext/Media）分类（`classify`：Agent 词、Chat 语义提示词覆盖知识/推理/代码/数学/翻译摘要/创作/对比/格式/专业咨询、短祈使句不漏 Inline、上下文 ≥0.8 阈值）
- [x] `RuleEngine` 有序规则链：硬约束高置信直决；Balance/Intelligence 的 Chat 直接云 handoff（`rule:chat_prefer_cloud`）；Cost 的 Chat 与 Agent/长上下文/复杂度邻域 → `need_semantic`；仅 Inline 问候本地快路径
- [x] 单测：分类矩阵、硬约束/不可用回退、邻域触发、模式 cutoff、默认回退

### T77 — 语义路由层 `hybrid/src/semantic.rs`
- [x] `SemanticDecision{action,confidence,intent,reason}`；`SemanticClient` 枚举（`CloudSemanticClient` 复用 `CloudClient` + 严格 JSON system prompt；`FakeSemanticClient` 测试注入，box future 无新依赖）
- [x] `SemanticRouter`：归一化 prompt FNV-1a 缓存（TTL 60s/容量 512 LRU-oldest）、同 key 单飞、`tokio::time::timeout`（默认 800ms）；未启用/无凭证/超时/解析失败 → `None` 静默降级
- [x] 单测：JSON 合法/fenced/非法/越界、缓存命中/TTL/容量淘汰、单飞、超时与错误降级、mock 网关解析

### T78 — 健康度回退链 `hybrid/src/health.rs`
- [x] `HealthTracker`：Local/Cloud 双后端评分（成功 +0.05 / 失败 −0.20 / 超时 −0.10，[0,1] 截断，healthy ≥ 0.5）；`snapshot()`
- [x] 单测：封顶/底线/阈值/恢复/快照 serde

### T79 — 决策合成 `Router::route_hybrid`（route.rs）
- [x] 规则快路径 → 语义慢路径（采纳阈值 0.6；MustLocal/execution=cloud 冲突不采纳）→ 健康翻转（软决策；Cloud 翻转须 `cloud_available`；MustLocal 永不翻转）→ 复用粘性
- [x] `RouteLayer::{Rules, Semantic}`；`RouteDecision`/`RouteOutcome` 增 `layer`/`confidence`/`semantic_consulted`/`semantic_latency_ms`（均 `#[serde(default)]`）；语义采纳 `policy_version` 追加 `+semantic`
- [x] 粘性守卫泛化（Local→Cloud 一律需硬升级，覆盖语义软升级；P0/P1 行为不变）
- [x] 单测：快路径跳过语义/采纳/低置信拒绝/失败降级/硬约束冲突/双向健康翻转/MustLocal 不翻转/禁用语义≡`route()`/粘性保持

### T80 — openai 接线与观测
- [x] `AriaConfig` 增 `hybrid_semantic`(true)/`hybrid_semantic_timeout_ms`(800)/`hybrid_semantic_cache_size`(512)（全带 default）；serve `--hybrid-semantic on|off`；auth 默认写回；status 打印
- [x] `AppState` 增 `semantic`/`health`；`build_state_with_hybrid_opts`；chat 改调 `route_hybrid`；执行结果回写 `HealthTracker`（超时/失败分类）
- [x] `GET /v1/engine/routes`：`?n=`（默认 20，上限 100）返回 recent outcomes + health snapshot + semantic 状态
- [x] 四轴合成：`EffectiveRouting` / `chat_policy`；serve 与 `/v1/engine/routes` 打印生效 `execution` / `mode` / `semantic` / `cloud` / `compute`（`semantic.applicable` 与配置 `enabled` 区分）
- [x] 单测：routes 端点字段、fake 语义层采纳 E2E（`semantic-cloud` + outcome layer=semantic）
- [x] `max_tokens` 设置则本地/云端原样使用；未设置则本地至 stop 或剩余 context（不默认 16），云端省略；`CloudClient` 超时 60s
- [x] hub token 走 `aria-engine auth`：按 site 区只提示 `hf_token`（`.com`）或 `modelscope_api_token`（`.cn`）；`download` 不再读 `HF_TOKEN` / `MODELSCOPE_API_TOKEN`
