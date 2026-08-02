# AGENTS.md — aria engine

工程上下文入口。逐层展开：先看「概述/架构/目录」，动手时再看「规范/命令/进行中/注意」。

## 概述
`engine` 仓库 = aria 推理引擎（Rust）：参考 Cactus Engine / Graph / Kernels / Hybrid。
五层：`openai`（OpenAI 兼容 API）→ `hybrid`（端云路由）→ `inference`（全家族推理）
→ `graph`（零拷贝计算图）→ `kernel`（ARM NEON + 可移植 scalar）。
权重格式：**仅** Aria bundle（`aria-quant-bundle` v1：`weight.bin` + `config.json` + tokenizer）。

## 架构
`openai` → `hybrid` → `inference`（Bundle + Session + 家族图）→ `graph` → `kernel`。
位宽消费：`q1`–`q4` / `q8` / 混合 `q1.5` / `q2.54` / `q3.26`（与 `model` 产物对齐）。
反量化：与 Python `dequantize` 一致的 **rotated-space** 重建；推理用融合 HDM
（Hadamard + Dequant + MatMul），不强制整表逆 Hadamard 物化。

## 目录
- `openai/`：`aria-openai` — HTTP（chat / 后续 audio·embeddings·tools）
- `hybrid/`：`aria-hybrid` — 置信度路由、`cloud_handoff`、云端 OpenAI 客户端
- `inference/`：`aria-inference` — Bundle 加载、Prefill/Decode、家族注册表
- `graph/`：`aria-graph` — Op DAG、BufferPool、mmap / external 零拷贝
- `kernel/`：`aria-kernel` — matmul / attention / norm / RoPE / dequant / FWHT
- 根：`AGENTS.md` / `requirements.md` / `task.md` / `README.md` / `Cargo.toml`

## 开发规范
- Rust（edition 2021+）；核心无重型 ML 框架；错误统一 `EngineError`；禁止吞错。
- 新增功能须单测（正常 + 异常）；Bug 修复须含复现用例；合入前 `cargo test` 全绿。
- Harness：半天以上须 `requirements.md`（人审）→ `task.md` → 编码；严禁无 Spec 直接 Coding。
- AGENTS.md ≤100 行；家族清单与 API 细节下沉 `requirements.md`。
- 不动 live `model` / `serve`；不解析 / 导出 GGUF。

## 常用命令
- `cargo test`
- `cargo test -p aria-kernel`
- `cargo build`
- `cargo clippy --all-targets`
- `cargo run -p aria-openai --bin aria-engine -- serve --model <bundle_dir>`

## 进行中需求
Spec 见 `requirements.md`（已审核通过）。`task.md` T0–T6（阶段 A）与 T10/T11（阶段 B/C）均已完成。

## 注意事项
- 黄金路径：tiny Aria q4 → load → dequant → decode → OpenAI chat/SSE/embeddings/ASR/tools。
- 全家族 §1.1：`ArchClass` + `graph_hook`；VL/VLA 见 `multimodal`。
- NEON：`SimdMode::Neon`（blocked matmul，与 scalar 对拍）；混合云：`ARIA_HYBRID_CLOUD_API_KEY` / `on_device_only`。
