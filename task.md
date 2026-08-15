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
