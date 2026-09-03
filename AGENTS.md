# AGENTS.md — aria engine

工程上下文入口。逐层展开：先看「概述/架构/目录」，动手时再看「规范/命令/进行中/注意」。

## 概述
`engine` 仓库 = aria 推理引擎（Rust）：参考 Cactus Engine / Graph / Kernels / Hybrid。
四层：`openai`（OpenAI 兼容 API）→ `inference`（全家族推理）
→ `graph`（零拷贝计算图）→ `kernel`（NEON / AVX2 scalar + 可选 CUDA）。
端云 / Mixture-of-Models 路由在独立 **`router` 仓**（`aria-router`）；本仓只做本地推理。
权重格式：**仅** Aria bundle（`aria-quant-bundle` **v1|v2**：`weight.bin` + `config.json` + tokenizer）。
评测：`bench/` Python，对标 aria / llama.cpp / ollama / vllm / sglang，产出 JSON+MD（report-only）。

## 架构
`openai` → `inference`（Bundle + Session + 家族图）→ `graph` → `kernel`。`compute` 仅本地 GEMM。
位宽消费：`q1`–`q4` / `q8` / 混合 `q1.5` / `q2.54` / `q3.26`（与 `model` 产物对齐）。
反量化：rotated-space 码本；embedding 加载期整表 unrotate；线性层原域 `linear` 或融合 HDM。v2 = blocked tiles。

## 目录
- `openai/`：`aria-openai` — HTTP（chat / audio·embeddings·tools）；可选 `--router` 向 `aria-router` 注册本机为 provider
- `inference/`：`ariacompute-inference` — Bundle 加载、Prefill/Decode、家族注册表
- `graph/`：`ariacompute-graph` — Op DAG、BufferPool、mmap / external 零拷贝
- `kernel/`：`ariacompute-kernel` — matmul / attention / norm / RoPE / dequant / FWHT / CUDA GEMM
- `ffi/`：`ariacompute-ffi` — C ABI（cdylib/staticlib）；`bindings/` 八语言 SDK（均支持按模型名从本区公开 hub 自动下载到 `~/.ariacompute/models` 再加载）
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
- `cargo run -p aria-openai --bin aria-engine -- serve <model|bundle_dir>`
- `aria-engine setup` / `download` / `list` / `check` / `clean` / `upgrade`
- `./scripts/run-binding-tests.sh`
- `python -m unittest discover -s bench/tests -t .`
- `python -m bench run --backend aria=http://127.0.0.1:8080 --report ./out/bench_report.json`
- `bash scripts/build-python-ffi.sh` + `cibuildwheel`（PyPI 平台 wheel 构建）

## 进行中需求
Spec 见 `requirements.md`（含 §8.7 profile、**§3.4 `compute=auto`**、§3.3.1、§3.7 PyPI 发布、**§3.5 路由上移 router 仓**）。T50–T80 历史项已落地；本增量删除 hybrid 云面，新增可选 `router` 注册。DeltaNet GQA、完整 ViT 仍待。

## 注意事项
- 黄金路径：tiny Aria q4 → load → dequant → decode → OpenAI chat/SSE/embeddings/ASR/tools。
- 全家族 §1.1：`ArchClass` + `graph_hook`；VL/VLA 见 `multimodal`。
- 评测对齐 `model` `audit_cli`：缺后端 skip、`ci_fail: false`；不启动第三方引擎进程。
- NEON/AVX2 CPU；`compute=auto` 可选 CUDA（仅本地 GEMM）。`~/.ariacompute/engine.yml`：`router`/`router_api_key`（`sk-aria_`）与可选 `serve_site`/`serve_api_key`（`bfvk`）；CLI 分段 Local vs OAuth。`download` 仅本区 hub。`list` 只扫本地缓存。
- 与 **model** 协同 blocked Hadamard（`format_version=2`）。C ABI 变更须更新 `bindings/testdata/cases.json` 与宿主测。
