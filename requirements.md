# requirements.md — Aria Engine（AFM-D / Rust）

> 本文件为 `engine` 仓库 **AFM-D 双轨 typed-decision 推理** 的功能边界、API、资产命名与验收标准。行为真源：`model/afm-d` Python 打分契约。产品壳（CLI / upgrade / FFI / bindings / publish）对齐已弃用的 Aria chat `deprecated/engine`，**不**恢复其 `weight.bin` / graph / kernel 聊天栈。

## 1. 目标与范围

用 **Rust + candle** 实现 AFM-D **推理运行时**（无-only）：

| 轨 | Hub / 本地 | 架构 |
| --- | --- | --- |
| Encoder `afm_de` | `ariacompute/afm-de` / Laya 布局 checkpoint | ModernBERT DecisionModel + MASK 选项头（Laya 对齐） |
| Decoder `afm_dd` | `ariacompute/afm-dd` / MiniCPM±LoRA | SemIf direct 首 token option-letter |

产品面：

1. **`aria-engine` CLI** — `setup` / `download` / `list` / `check` / `clean` / `upgrade` / `serve` / `decide` / `version`
2. **HTTP** — `POST /v1/systemone`、`GET /health`、`GET /v1/models`（axum）
3. **C FFI** — `libaria_ffi`（发布名 `libaria-engine_ffi`）：`aria_model_init` / `aria_systemone` / `aria_model_destroy`
4. **Language SDKs** — `bindings/{rust,typescript,react-native,python,flutter,kotlin,swift,go}`
5. **Release / publish** — `.github/workflows/release.yml` + cargo/npm/pub/pypi/maven/CocoaPods

### 1.1 非目标

- 训练 / RLCD / LoRA train（留在 `model/afm-d`）
- Aria chat `aria-quant-bundle` / GGUF / graph/kernel/session
- `router` / `router_api_key` / serve 向 aria-router 注册（本仓 setup 明确不做）
- 替换 Python train/eval/bench
- Metal / Vulkan / ANE（roadmap）；可选 `cuda` feature 预留，不阻塞 CPU 验收

## 2. 功能边界

| # | 特性 | 要求 |
|---|------|------|
| 1 | **core** | System One 类型、packing（1024/512/96）、typed answer、`~/.ariacompute` 五字段 config、gateway / preferred_hub |
| 2 | **de** | 加载 `rl_agent_config.json` + `model.safetensors` + tokenizer；Laya 序列 packing；shortlist（K>40）；logits→softmax→typed answer；candle DecisionHead 骨架；无权重时允许 `score_record_with_logits` 黄金路径 |
| 3 | **dd** | SemIf row（选项 2–16）；candle MiniCPM5-2B（pin `12a3808a956f869c767195e9266b59c4d21d92e2`）± PEFT merge-at-load；首位置字母 logits → `probs_from_semif_out`；底座优先 `checkpoint/base/`（download 落盘），否则 `AFM_DD_BASE` / HF cache；黄金路径仍可用 `score_from_semif_out` |
| 4 | **CLI setup** | 五字段 `engine.yml`：`site_url` / `upgrade_url` / `compute` / `hf_token` / `modelscope_api_token`；`--status`/`--clear`；区域 hub token 交互输入回显 `*`（不明码）；**禁止** `router` / `router_api_key` |
| 5 | **CLI download** | 按 `site_url` 选 HF vs ModelScope；**HTTP API**（tree/list + resolve），Bearer 注入 setup token；**不用** huggingface-cli / modelscope CLI；失败不落盘可见缓存；`afm-dd` 额外拉取 MiniCPM5-2B safetensors（pin rev）到 `base/`，**跳过** GGUF |
| 5b | **CLI list** | 合并本地完整 checkpoint 与区域 Hub 组织目录（HF `ariacompute` / MS `AriaCompute`）；每行标记 `downloaded` / `not downloaded`；Hub 不可达时降级为仅本地 |
| 6 | **CLI serve** | 读 `compute`（旗标可覆盖）；**不做** aria-router 注册 |
| 7 | **upgrade** | 自 GitHub/Gitee Releases 拉 `aria-engine_${VER}_${OS}` + `libaria-engine_ffi_${VER}_${OS}.tar.gz`，替换二进制与 `~/.ariacompute/lib/` |
| 8 | **FFI** | 仅 System One；**禁止** `aria_complete` / embed / transcribe |
| 9 | **bindings** | 对外主 API：`systemone` / `decide`；自动解析 `ARIA_FFI_LIB` 或 `~/.ariacompute/lib/libaria-engine_ffi.*` |
| 10 | **CI publish** | release 多 OS 构建上传；各语言 publish 工作流版本取自 tag（去 `v`） |

### 2.1 契约锁（禁止漂移）

- Encoder：`MAX_LEN=1024`，`HEAD_MAX_LEN=512`，`OPTION_DESC_MAX=96`，决策温度 **1.0**
- Decoder：选项 **2–16**，默认端口 **8011**
- System One：`{ state, questions }` → `{ answers, model }`；题型 `choice` / `score` / `noul`
- Noul：准则键 `true`/`false`（接受 yes/no 别名规范化）
- 权重：safetensors / PEFT；**禁止**声称 GGUF

## 3. Workspace

| Crate / 路径 | 角色 |
| --- | --- |
| `ariacompute-core` | 共享类型与 packing |
| `ariacompute-de` | Encoder |
| `ariacompute-dd` | Decoder |
| `aria-cli`（bin `aria-engine`） | CLI + serve + upgrade |
| `ariacompute-ffi`（lib `aria_ffi`） | C ABI |
| `bindings/rust` → `ariacompute-engine` | crates.io SDK |
| `bindings/*` | 其余语言 SDK |
| `.github/workflows/*` | release + publish + bindings CI |

## 4. 验收

| 级别 | 标准 |
| --- | --- |
| **A（已交付脚手架）** | `cargo test --workspace` 绿；packing / typed / SemIf map 黄金夹具；CLI `-v`；FFI 头与 crate 可链接；bindings 树与 publish yml 就位 |
| **B（权重前向）** | Hub `afm-de` / `afm-dd` checkpoint 上 candle 前向与 Python `afm_d.de.score` / `afm_d.dd.score` 标签一致（argmax + 概率差阈值，如 1e-3） |
| **C（发布）** | `release.yml` 产出 CLI+FFI 资产；`upgrade` 可自更；至少一种宿主 binding `systemone` E2E |

## 5. 文档与协作

- `README.md` / `AGENTS.md` 与本文件一致
- `task.md` 为实施勾选清单；改契约先改本文件
