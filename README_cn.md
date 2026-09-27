# Aria Engine

纯 Rust 推理运行时，面向 **AFM**：

| 轨 | Crate | 模型 |
| --- | --- | --- |
| Encoder（`afm_de`） | `ariacompute-de` | Laya 布局 ModernBERT DecisionModel |
| Decoder（`afm_dd`） | `ariacompute-dd` | MiniCPM5-2B ± PEFT LoRA（SemIf direct） |

产品面：**`aria-engine` CLI** · **`POST /v1/systemone`** · **`libaria_ffi` C ABI** · `bindings/` 下各语言 SDK。

## 构建

```bash
cargo build --workspace
cargo test --workspace
cargo build --release -p aria-cli -p ariacompute-ffi
```

可选 CUDA：`cargo build -p aria-cli --features cuda`（会传到 de/dd）。

## CLI

```bash
aria-engine setup                 # site_url + upgrade_url（GitHub/Gitee 组织根）
aria-engine download afm-de       # → ~/.ariacompute/models/afm-de
aria-engine list | check | clean
aria-engine upgrade [version]     # 从 Releases 更新 CLI + libaria-engine_ffi
aria-engine serve --track decoder --checkpoint ~/.ariacompute/models/afm-dd
aria-engine decide --track encoder --checkpoint ./ckpt --logits '[0.1,2.0]' < record.json
aria-engine -v
```

Decoder HTTP 默认监听：`127.0.0.1:8011`。Encoder 默认：`127.0.0.1:8010`。

## FFI

头文件：[`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`、`aria_systemone`、`aria_model_destroy`。

发布资产：`aria-engine_${VERSION}_${OS}.tar.gz|zip`、`libaria-engine_ffi_${VERSION}_${OS}.tar.gz`。

## 语言 SDK

| Binding | 包名 |
| --- | --- |
| Rust | `ariacompute-engine`（`cargo add`） |
| TypeScript | `@ariacompute/engine-ts` |
| React Native | `@ariacompute/engine-rn` |
| Python | `aria-engine`（PyPI） |
| Flutter | `package:aria_engine`（pub.dev） |
| Kotlin | `com.ariacompute:engine`（Maven Central） |
| Swift | CocoaPods `AriaEngine` |
| Go | `bindings/go`（cgo） |

## 发布工作流

GitHub Release 触发：`release.yml`（CLI + FFI 资产），以及 `publish-cargo.yml`、`publish-npm.yml`、`publish-pub.yml`、`publish-pypi.yml`、`publish-maven.yml`、`publish-cocoapods.yml`。

## 许可证

MIT。

## 工程约定

本仓库遵循 Harness Engineering 理念：

- [`AGENTS.md`](AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](requirements.md)：需求规格（功能边界 / 异常 / 验收标准，经人工审核门控）
- [`task.md`](task.md)：实施任务勾选清单
