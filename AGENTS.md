# AGENTS.md — aria engine

工程上下文入口。逐层展开：先看「概述/架构/目录」，动手时再看「规范/命令/进行中/注意」。

## 概述
`engine` 仓库 = aria 推理引擎（Rust）：参考 Cactus Engine / Graph / Kernels / Hybrid。
五层：`openai`（OpenAI 兼容 API）→ `hybrid`（端云路由）→ `inference`（全家族推理）
→ `graph`（零拷贝计算图）→ `kernel`（ARM NEON + 可移植 scalar）。
权重格式：**仅** Aria bundle（`aria-quant-bundle` **v1|v2**：`weight.bin` + `config.json` + tokenizer）。
评测：`bench/` Python，对标 aria / llama.cpp / ollama / vllm，产出 JSON+MD（report-only）。

## 架构
`openai` → `hybrid` → `inference`（Bundle + Session + 家族图）→ `graph` → `kernel`。
位宽消费：`q1`–`q4` / `q8` / 混合 `q1.5` / `q2.54` / `q3.26`（与 `model` 产物对齐）。
反量化：与 Python `dequantize` 一致的 **rotated-space** 重建；推理用融合 HDM
（blocked Hadamard + Dequant + MatMul），不强制整表逆 Hadamard 物化。v2 = blocked tiles。

## 目录
- `openai/`：`aria-openai` — HTTP（chat / audio·embeddings·tools）
- `hybrid/`：`aria-hybrid` — 信号→投影→决策、Pareto 模式、会话粘性、`cloud_handoff`、Outcome
- `inference/`：`aria-inference` — Bundle 加载、Prefill/Decode、家族注册表
- `graph/`：`aria-graph` — Op DAG、BufferPool、mmap / external 零拷贝
- `kernel/`：`aria-kernel` — matmul / attention / norm / RoPE / dequant / FWHT
- `bench/`：Python 引擎对标评测（§1.1 全家族；性能+质量；JSON+MD）
- 根：`AGENTS.md` / `requirements.md` / `task.md` / `README.md` / `Cargo.toml`

## 开发规范
- Rust（edition 2021+）；核心无重型 ML 框架；错误统一 `EngineError`；禁止吞错。
- 新增功能须单测（正常 + 异常）；Bug 修复须含复现用例；合入前 `cargo test` 全绿。
- Harness：半天以上须 `requirements.md`（人审）→ `task.md` → 编码；严禁无 Spec 直接 Coding。
- AGENTS.md ≤100 行；家族清单与 API 细节下沉 `requirements.md`。
- 不解析 / 导出 GGUF。本增量可协同改 `model`（blocked Hadamard）。

## 常用命令
- `cargo test`
- `cargo run -p aria-openai --bin aria-engine -- serve --model <bundle_dir>`
- `python -m unittest discover -s bench/tests -t .`
- `python -m bench run --backend aria=http://127.0.0.1:8080 --report ./out/bench_report.json`

## 进行中需求
Spec 见 `requirements.md`（含 §8 评测）。`task.md` T0–T6 / T10–T11 / T20 / **T21** 均已完成。

## 注意事项
- 黄金路径：tiny Aria q4 → load → dequant → decode → OpenAI chat/SSE/embeddings/ASR/tools。
- 全家族 §1.1：`ArchClass` + `graph_hook`；VL/VLA 见 `multimodal`。
- 评测对齐 `model` `audit_cli`：缺后端 skip、`ci_fail: false`；不启动第三方引擎进程。
- NEON：`SimdMode::Neon`；混合云：`ARIA_HYBRID_CLOUD_API_KEY` / `ARIA_HYBRID_MODE`（复杂度阈值）/ `ARIA_HYBRID_EXECUTION`。
- 与 **model** 协同 blocked Hadamard（`format_version=2`）。
