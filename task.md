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

### T8 — setup 五字段（无 router）
- [x] `apply_hub_token_input` / `preferred_hub`；`clear` 含 legacy `config.yml`
- [x] `cli/src/setup.rs`：交互 + `--status`/`--clear`/`--site-url`/`--upgrade-url`/`--compute`
- [x] download：按 `site_url` 选 HF/ModelScope，**HTTP API**（tree + resolve）+ setup Bearer token（无 hub CLI）
- [x] serve：读 `compute`（`--compute` 覆盖）；无 router 注册
- [x] README / README_cn / requirements 同步五字段表

## 阶段 B — 权重前向 parity（进行中）

### T10 — Encoder candle 前向
- [ ] 加载 Hub/本地 `afm-de` `model.safetensors`（ModernBERT + head 键名对齐）
- [ ] `EncoderScorer::score_record` 真实前向（不仅 logits 注入）
- [ ] 与 Python `afm_d.de.score` 对比：argmax + max|Δp| 阈值

### T11 — Decoder candle 前向
- [x] MiniCPM5-2B（pin rev）+ 可选 PEFT merge-at-load
- [x] SemIf direct 首位置字母 logits
- [ ] 与 Python `DecoderScorer` / `afm_d.dd.score` 对比

### T12 — Serve / FFI E2E（有权重）
- [ ] `aria-engine serve --track encoder|decoder` 对真实 checkpoint 返回 answers
- [ ] FFI / 至少一种宿主 SDK `systemone` 冒烟（init → systemone → destroy）

## 阶段 E — 测试覆盖硬化（T90）

目标：全 workspace 行覆盖 100%（`./scripts/coverage.sh`，排除 2 个纯张量前向文件，理由见
`requirements.md` §4.1），并把门禁固化到 CI。

- [x] 注入 seam：`EncoderForward`（de）/ `LetterScorer`（dd）/ `Judge`（serve），
      `#[doc(hidden)] with_forward` / `from_parts` / `with_encoder`、`with_decoder`
- [x] 模块拆分：`de/src/model/{mod,config,forward}.rs`、`dd/src/{weights,letters}.rs`，
      把不可测的张量图收敛到 `forward.rs`
- [x] CLI 拆分：`app.rs`（子命令分发）+ `main.rs`（tracing + parse + exit）
- [x] 网络 seam：`HubEndpoints` / `RetryPolicy` / `DownloadCtx`（`ARIA_HF_ENDPOINT`、
      `ARIA_MS_ENDPOINT`、`ARIA_MS_ENDPOINT_ALT`）与 `ReleaseApi`（`ARIA_UPGRADE_API_GITHUB`、
      `ARIA_UPGRADE_API_GITEE`），全部用 `wiremock` 离线覆盖
- [x] `serve` 优雅退出：`serve_on(listener, app, shutdown)` + `run_serve` 接 `ctrl_c`
- [x] `setup` 非 TTY 回退：`Prompter` seam + `cmd_setup_with`
- [x] core：config（`ARIA_COMPUTE_HOME` 隔离）/ gateway（LANG）/ packing / systemone / typed /
      contract / error 全分支
- [x] de：checkpoint、tokenizer 夹具、model/config 映射、shortlist、score
- [x] dd：checkpoint、prompt、semif、pyjson、weights（定位/收集/反量化/LoRA）、letters、score
- [x] ffi：`null` 指针、非法 UTF-8、非法 track、坏 JSON、缓冲不足 `-2`、成功 `0`、thread-local 错误槽
- [x] bindings/rust：`decide_with_logits` / `decide_with_semif_out` / 交叉误用 / `resolve_checkpoint`
- [x] `cli/tests/cli.rs`：assert_cmd 覆盖 setup / download / list / check / clean / upgrade / serve /
      decide / version 与退出码（含 SIGINT 优雅退出）
- [x] `scripts/coverage.sh` + `.github/workflows/ci.yml`（fmt / clippy / test / coverage / lcov artifact）
- [x] 门禁口径：脚本解析 lcov `DA:<line>,0`，要求零未覆盖行（`lines: 8239/8239 (100.00%)`）；
      不用 `--fail-under-lines 100`（多实例下 summary 按「行 × 实例」成对计数会自相矛盾，
      见脚本头注释与 `requirements.md` §4.1）

## 阶段 C — 发布硬化

### T20 — Release 资产
- [ ] Tag 触发 `release.yml`：四 OS CLI + FFI 归档上传
- [ ] `aria-engine upgrade` 自更验证（GitHub 或 Gitee）

### T21 — 注册表发布
- [ ] crates.io `ariacompute-*` 拓扑发布
- [ ] npm / PyPI / pub.dev / Maven / CocoaPods（密钥与 XCFramework 齐备后）

## 阶段 D — eMudhra / emSign 证书私钥泄露响应（2026-10）

背景：eMudhra(emSign) 8 张中间 CA 私钥极弱（P-384 私钥标量 <256），可被反推伪造任意域名证书；
Rust(webpki-roots) / Go 等自带信任库需升级。本仓 Rust 二进制会向模型 hub 发起出向 TLS（下载权重），受信任库直接影响。

### 1. Rust（aria-engine / aria-cli）
- `Cargo.lock`：`webpki-roots` 1.0.9 -> 1.0.10（`cargo update -p webpki-roots --precise 1.0.10`）。
  1.0.10+ 在 rustls 信任库中移除 emSign 根；`reqwest` 与 `hf-hub` 均用 `rustls-tls`，出向校验依赖它。
- `cli/Cargo.toml`：显式钉住 `webpki-roots = "1.0.10"`，防止锁漂移回受影响版本。
- `Dockerfile`：构建(`rust:1-slim`)与运行时均 `apt-get install ca-certificates`（系统库已就绪）；
  但二进制信任库编译进产物（webpki-roots），故锁升级是决定性修复。
- 核验：`cargo tree -i webpki-roots` 须 >= 1.0.10。

### 2. 语言绑定（Go / Python / TS / RN）
- 均为 FFI-only（cgo / ctypes / koffi），**不做出向 TLS**，运行时信任库不在本事件范围内；
  实际网络路径在上面的 Rust 侧。
- `bindings/go/go.mod`：`go 1.22` -> `1.27.3`（工具链对齐；该绑定无出向 TLS，纯属一致性硬化）。

### 3. 检测与验证
- 新增 `scripts/check-emudhra.sh`：离线按 emSign 根（信任锚）指纹扫描系统 CA bundle，
  规避前向/拦截代理造成的在线探测假阳性。
- 构建后核验：`docker run --rm <engine-image> scripts/check-emudhra.sh /etc/ssl/certs/ca-certificates.crt`。

## 明确不做

- [ ] ~~恢复 Aria chat graph/kernel~~（永久非目标）
- [ ] ~~本仓训练 RLCD / LoRA~~（见 `model/afm-d`）
