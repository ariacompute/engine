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
- [x] `load_bundle`：`aria-quant-bundle` v1；mmap `weight.bin`；codebook/raw
- [x] LSB unpack 1–4 / u8 for 8；`dequantize` rotated-space
- [x] §1.1 `Family` 注册表完整；非阶段 A → `UnsupportedFamily`
- [x] 测试夹具 tiny q4 bundle（`fixture::write_tiny_q4_bundle`）
- [x] `Session`：prefill + decode，greedy 产出非空 tokens
- [x] 反量化 RMSE 有界单测

### T4 — `aria-hybrid`
- [x] `Router` + 置信度阈值 → `Local` / `CloudHandoff`
- [x] `CloudClient`（reqwest）+ `ARIA_HYBRID_CLOUD_API_KEY`
- [x] mock：成功 / 超时或非 2xx → `Cloud`

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
- [x] hybrid `on_device_only` 强制 Local（集成测试）

## 阶段 D — 引擎对标评测（requirements §8）

### T20 — `bench/` Python harness
- [x] §1.1 全家族注册表（与 model EXPECTED 锁表）
- [x] 后端适配：`aria` / `llamacpp` / `ollama` / `vllm`（OpenAI chat）
- [x] 性能：latency p50/p95、tok/s、可选 TTFT；质量：token_overlap + exact_match vs ref
- [x] CLI：`python -m bench run` → JSON + MD；缺后端 skip；默认 exit 0
- [x] `unittest` mock；更新 AGENTS / README
