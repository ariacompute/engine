# task.md — Aria Engine（AFM-D）实施清单

依据 [`requirements.md`](requirements.md)。完成后勾选。

## 阶段 A — 脚手架 + 契约层（已完成）

### T0 — Workspace
- [x] 根 `Cargo.toml`：`core` / `de` / `dd` / `cli` / `ffi` / `bindings/rust`
- [x] `README.md` / `AGENTS.md` / MIT
- [x] `cargo build --workspace` / `cargo test --workspace` 绿

### T1 — `ariacompute-core`
- [x] System One 请求/响应与 `record_from_systemone_question`
- [x] Packing：Laya 文本/ID 序列、`OPTION_DESC_MAX=96`、noul 规范化
- [x] `typed_answer` / softmax；`~/.ariacompute` config + gateway upgrade_url
- [x] 单测：packing / typed / systemone / gateway

### T2 — `ariacompute-de`（Encoder）
- [x] Checkpoint 布局：`rl_agent_config.json` / `model.safetensors` / tokenizer
- [x] Pack + shortlist + `score_record_with_logits`
- [x] candle `DecisionHead` 骨架
- [x] 黄金夹具：`tests/fixtures/encoder_*.json` → `de/tests/golden_parity.rs`

### T3 — `ariacompute-dd`（Decoder）
- [x] `record_to_semif_row`（2–16）
- [x] `probs_from_semif_out` + System One map
- [x] `DecoderCheckpoint` / `dd_config.json` + adapter 探测
- [x] 黄金夹具：`decoder_choice.json` → `dd/tests/golden_parity.rs`

### T4 — CLI + serve + upgrade
- [x] `aria-engine`：setup / download / list / check / clean / upgrade / serve / decide / version
- [x] `ARIA_ENGINE_VERSION`（`build.rs`）
- [x] axum `POST /v1/systemone`；Decoder 默认 `:8011`，Encoder `:8010`
- [x] upgrade 资产名与 `release.yml` 对齐

### T5 — FFI
- [x] `ffi/include/aria.h`：`aria_model_init` / `aria_systemone` / helpers
- [x] cdylib/staticlib `aria_ffi`；无 chat complete/embed/transcribe

### T6 — Bindings + CI publish
- [x] `bindings/{rust,typescript,react-native,python,flutter,kotlin,swift,go}` — `systemone` API
- [x] `scripts/{build-python-ffi,prepare-binding-fixture,run-binding-tests}.sh`
- [x] workflows：`release.yml`、`bindings-host.yml`、`bindings-mobile.yml`
- [x] `publish-{cargo,npm,pub,pypi,maven,cocoapods}.yml`（crates 序：core→de→dd→ffi→engine）

### T7 — 文档
- [x] `requirements.md` / `task.md` 与 AFM-D 产品面对齐（本文件）

## 阶段 B — 权重前向 parity（进行中）

### T10 — Encoder candle 前向
- [ ] 加载 Hub/本地 `afm-de` `model.safetensors`（ModernBERT + head 键名对齐）
- [ ] `EncoderScorer::score_record` 真实前向（不仅 logits 注入）
- [ ] 与 Python `afm_d.de.score` 对比：argmax + max|Δp| 阈值

### T11 — Decoder candle 前向
- [ ] MiniCPM5-2B（pin rev）+ 可选 PEFT merge-at-load
- [ ] SemIf direct 首位置字母 logits
- [ ] 与 Python `DecoderScorer` / `afm_d.dd.score` 对比

### T12 — Serve / FFI E2E（有权重）
- [ ] `aria-engine serve --track encoder|decoder` 对真实 checkpoint 返回 answers
- [ ] FFI / 至少一种宿主 SDK `systemone` 冒烟（init → systemone → destroy）

## 阶段 C — 发布硬化

### T20 — Release 资产
- [ ] Tag 触发 `release.yml`：四 OS CLI + FFI 归档上传
- [ ] `aria-engine upgrade` 自更验证（GitHub 或 Gitee）

### T21 — 注册表发布
- [ ] crates.io `ariacompute-*` 拓扑发布
- [ ] npm / PyPI / pub.dev / Maven / CocoaPods（密钥与 XCFramework 齐备后）

## 明确不做

- [ ] ~~恢复 Aria chat graph/kernel~~（永久非目标）
- [ ] ~~本仓训练 RLCD / LoRA~~（见 `model/afm-d`）
