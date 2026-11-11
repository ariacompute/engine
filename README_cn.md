<div align="center">
<a name="readme-top"></a>

# Aria Engine

**面向 AFM-D 类型化决策的纯 Rust 推理运行时**

**Encoder**（`afm_de`，Laya 布局 ModernBERT DecisionModel）与 **Decoder**<br/>
（`afm_dd`，MiniCPM5-2B ± PEFT LoRA）各自在同一份契约下发布类型化决策——<br/>
**System One**。产品面：**`aria-engine` CLI** · **`POST /v1/systemone`** ·<br/>
**`libaria_ffi` C ABI** · `bindings/` 下各语言 SDK。

[English](./README.md) · **简体中文** · [Agent 上下文](./AGENTS.md) · [需求](./requirements.md) · [任务](./task.md)

[![][release-shield]][release-link] [![][rust-shield]][rust-link] [![][license-shield]][license-link] [![][lastcommit-shield]][lastcommit-link]
<br/>
[![][httpapi-shield]][httpapi-link] [![][tests-shield]][tests-link] [![][sdks-shield]][sdks-link] [![][trust-shield]][trust-link]
</div>

<details>
<summary><kbd>Table of contents</kbd></summary>

#### TOC

**上手**

- [✨ 功能](#-功能)
- [🧱 工作区](#-工作区)
- [🚀 快速开始](#-快速开始)
- [🧰 命令行](#-命令行)
- [🔧 配置](#-配置)

**部署与服务**

- [🛳 部署](#-部署)
  - [`A` 源码运行](#a-源码运行)
  - [`B` Docker Compose](#b-docker-compose)
- [🔌 HTTP API](#-http-api)

**内部机制**

- [🧭 Decide（类型化决策）](#-decide类型化决策)
- [🧩 FFI 与语言 SDK](#-ffi-与语言-sdk)
- [🔏 信任库安全](#-信任库安全)
- [🔨 开发](#-开发)
- [📚 文档](#-文档)
- [📝 许可证](#-许可证)
- [🤝 工程约定](#-工程约定)

</details>

## ✨ 功能

Aria Engine 把「一个问题」变成「一个类型化答案」——*对这个状态，哪个选项是（或哪个等级、是否）正确决策？*——没有聊天循环。Encoder 与 Decoder 共用同一份 System One 契约，调用方无需关心是哪个模型作答。

- 🧮 **AFM-D 类型化决策** —— `choice` / `score` / `noul` 答案共用同一 System One 形态，不做开放式生成。
- ⚡ **两条轨** —— Encoder（ModernBERT）快速输出选项 logits；Decoder（MiniCPM5-2B）走 SemIf 风格直接打分。
- 🖥 **四种接入面** —— `aria-engine` CLI、`POST /v1/systemone` HTTP、`libaria_ffi` C ABI，以及八语言 SDK。
- 📦 **离线优先** —— 真实权重前向需要已下载的 checkpoint；黄金路径打分（`--logits` / `--semif-out`）无需权重也可运行。
- 🌍 **区域 hub** —— `.com` 解析 Hugging Face，`.cn` 解析 ModelScope，由 `aria-engine setup` 选定。
- 🔏 **钉死的信任库** —— 二进制内嵌 `webpki-roots >=1.0.10`，剔除被泄露的 emSign 根（见 [信任库安全](#-信任库安全)）。

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🧱 工作区

Rust 工作区，无运行时数据库、无持久化。只复用产品外壳模式（CLI、FFI 打包、`release.yml`）；已废弃的 chat graph 不再保留。

| Crate | 职责 |
|---|---|
| `ariacompute-core` | System One 类型、打包、`~/.ariacompute` 配置、gateway |
| `ariacompute-de` | Encoder checkpoint + 打包 + score/logits |
| `ariacompute-dd` | SemIf 行 + decoder map + scorer shell |
| `aria-cli` | bin `aria-engine`（`setup` / `download` / `upgrade` / `serve` / `decide`） |
| `ariacompute-ffi` | cdylib `aria_ffi` → 发布名 `libaria-engine_ffi` |
| `bindings/rust` | crates.io `ariacompute-engine` |

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🚀 快速开始

```bash
cargo build --workspace
cargo test --workspace
cargo build --release -p aria-cli -p ariacompute-ffi
```

可选 CUDA：`cargo build -p aria-cli --features cuda`（会传到 de/dd）。

```bash
# 1. 配置区域与 hub token（写入 ~/.ariacompute/engine.yml）
aria-engine setup --site-url https://ariacompute.cn --compute cuda

# 2. 下载 checkpoint（HF / ModelScope HTTP API，不依赖 hub CLI）
aria-engine download afm-de
aria-engine download afm-dd

# 3. 以本地 System One 服务启动 Encoder（→ http://127.0.0.1:8010）
aria-engine serve --model-name afm-de

# 4. 提一个类型化问题
curl -s http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "user wants a refund",
    "questions": {
      "q1": {
        "type": "choice",
        "instructions": "Pick the best action",
        "criteria": {
          "refund": "issue a full refund",
          "deny": "deny the request"
        }
      }
    }
  }'
```

真实权重前向需要完整本地 checkpoint（见 [`task.md`](task.md) 阶段 B）。此前可用 `decide` 的 `--logits` / `--semif-out` 做黄金路径打分。

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🧰 命令行

```bash
aria-engine download afm-de       # HF/ModelScope HTTP API（site_url + setup token）；不依赖 hub CLI
                                  # .cn → ModelScope /api/v1/.../repo?FilePath=… ；.com → HF resolve
aria-engine list | check | clean  # list = 本地 ∪ Hub 组织（ariacompute / AriaCompute）；标记 downloaded|not downloaded
                                  # check/clean：仅完整 checkpoint；失败下载会清理
aria-engine upgrade [version]     # 需要 upgrade_url
aria-engine serve …               # 见下方 HTTP API
aria-engine decide …              # 见下方 Decide
aria-engine -v
```

| 命令 | 用途 |
|---|---|
| `setup` | 写入 `~/.ariacompute/engine.yml`（区域、hub token、compute） |
| `download` | 经 HF/ModelScope HTTP API 拉取 AFM checkpoint |
| `list` / `check` / `clean` | 列举 / 校验 / 清理本地与 hub checkpoint |
| `upgrade` | 从 `upgrade_url` 升级二进制（需有 release） |
| `serve` | 启动本地 System One HTTP 服务 |
| `decide` | 对一份 JSON 做一次性类型化决策 |
| `-v` | 打印版本 |

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🔧 配置

`aria-engine setup` 写入 `~/.ariacompute/engine.yml`（**无** router 字段）：

| 字段 | 含义 | 默认 |
| --- | --- | --- |
| `site_url` | 站点（`.com` / `.cn`）→ hub 分区 | 语言环境（`LANG=zh*` → `.cn`） |
| `upgrade_url` | Releases 组织根（GitHub / Gitee） | 随 `site_url` |
| `compute` | `auto` \| `cpu` \| `cuda` | `auto` |
| `hf_token` | Hugging Face token（`.com`） | _(空)_ |
| `modelscope_api_token` | ModelScope token（`.cn`） | _(空)_ |

```bash
aria-engine setup                 # 交互
aria-engine setup --status        # 密钥脱敏
aria-engine setup --clear
aria-engine setup --site-url https://ariacompute.cn --compute cuda
```

区域 hub token：`.cn` 只问 ModelScope；否则只问 Hugging Face。交互输入时回显为 `*`，不明码显示。

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🛳 部署

支持两条路径：直接运行二进制（连已下载 checkpoint），或以容器方式跑整套栈。

### `A` 源码运行

```bash
cargo build --release -p aria-cli
aria-engine setup                 # 写入 ~/.ariacompute/engine.yml
aria-engine download afm-de       # 拉取 encoder checkpoint
aria-engine serve --model-name afm-de --bind 127.0.0.1:8010
```

encoder 服务默认监听 `127.0.0.1:8010`，decoder 监听 `127.0.0.1:8011`。可用 `--compute cpu|cuda` 覆盖 `engine.yml` 中的取值。跨主机暴露前，请在其前方放置一个终结 TLS 的反向代理（如 Nginx、Caddy）。

### `B` Docker Compose

将 System One HTTP 服务以容器方式运行。仓库附带一个参数化 `Dockerfile`、编排脚本
`docker/entrypoint.sh`、CPU 版 `docker-compose.yml`（两个服务：`engine-encoder`
在 **8010**、`engine-decoder` 在 **8011**），以及 GPU 覆盖文件
`docker-compose.cuda.yml`。

1. 基于模板生成 `.env`：

   ```bash
   cp .env.example .env          # 将 ARIA_DATA_DIR 设为宿主的 ~/.ariacompute（绝对路径）
   ```

   两个服务将宿主的 `ARIA_DATA_DIR`（默认 `~/.ariacompute`）绑定挂载到 `/data/aria`，
   因此**宿主已下载的** `engine.yml`（配置）与 `models/`（checkpoint）会被直接使用——
   **离线、无需下载**。entrypoint 仅在 `engine.yml` 缺失时依据环境变量生成它，
   **已挂载的配置会被尊重**（例如宿主的 `~/.ariacompute/engine.yml`）。

2. 构建并启动（CPU）：

   ```bash
   docker compose build
   docker compose up -d
   ```

3. 验证：

   ```bash
   curl -s http://localhost:8010/health     # {"ok":true,"service":"afm-engine","model":"afm-de"}
   curl -s http://localhost:8011/health     # decoder
   docker compose ps                        # 两个服务均 healthy
   ```

**模型**

- **离线（默认，`AUTO_DOWNLOAD=0`）**：权重取自宿主
  `ARIA_DATA_DIR/models/{afm-de,afm-dd}`。请确保两个 checkpoint 都已存在于宿主
  （在宿主执行一次 `aria-engine download afm-de && aria-engine download afm-dd`，
  或把它们拷入 `~/.ariacompute/models/`）。
- **自动下载**：在 `.env` 中设置 `AUTO_DOWNLOAD=1`，entrypoint 会在首次运行时
  执行 `aria-engine download <model>`（需联网与对应 hub token）。**下载失败不会
  导致容器崩溃**——只会打印告警；仅当 checkpoint 仍然缺失时才拒绝启动。默认关闭。

> **注意 — hub 选择与网络**：当 `SITE_URL` 被设置时，容器会据此重新生成
> `engine.yml`，从而**覆盖**挂载进来的宿主配置。设置
> `SITE_URL=https://ariacompute.cn` 可让下载走 **ModelScope**（`afm-de` 发布于此），
> 而不是宿主里写死的 HuggingFace。若宿主只能通过本机代理访问外网，请在 `.env` 中设置
> `HTTPS_PROXY=http://host.docker.internal:<端口>`（compose 已将
> `host.docker.internal` 映射到宿主网关）；容器内的 `127.0.0.1` 指的是容器自身，
> 而非宿主。

**GPU（CUDA）** —— 需要宿主机已装 NVIDIA 驱动 + `nvidia-container-toolkit`：

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml build
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```

覆盖文件会将构建基础切换为 CUDA 镜像、以 `--features cuda` 编译，并为每个服务申请一块 GPU，
同时设置 `COMPUTE=cuda`，让 candle 选择 CUDA 后端。

**配置（`.env`）**

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `ENGINE_IMAGE` | `aria-engine:latest` | 构建镜像的标签 |
| `BUILD_IMAGE` / `RUNTIME_IMAGE` | `rust:1-slim` / *(默认等于 `BUILD_IMAGE`)* | CPU 构建基础镜像；运行时复用它以保证 glibc 始终一致。仅 CUDA 覆盖文件会设置 `RUNTIME_IMAGE` |
| `FEATURES` | _(空)_ | 构建特性；CUDA 覆盖文件设为 `cuda` |
| `SITE_URL` / `UPGRADE_URL` | _(空)_ | 区域 hub（`.com`→HF，`.cn`→ModelScope）；设置后会覆盖挂载进来的配置 |
| `HF_TOKEN` / `MODELSCOPE_API_TOKEN` | _(空)_ | 当前区域的 hub token |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | _(空)_ | 下载用的出口代理（本机代理用 `host.docker.internal`） |
| `COMPUTE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `ENCODER_MODEL` / `ENCODER_PORT` | `afm-de` / `8010` | encoder 模型（为空则禁用该服务）+ 端口 |
| `DECODER_MODEL` / `DECODER_PORT` | `afm-dd` / `8011` | decoder 模型（为空则禁用该服务）+ 端口 |
| `ARIA_DATA_DIR` | `${HOME}/.ariacompute` | 宿主目录，绑定挂载到 `/data/aria`（配置 + 模型） |
| `AUTO_DOWNLOAD` | `0` | 设为 `1` 可在首次运行时拉取权重（失败不致命） |
| `RUST_LOG` | `info` | Rust tracing 过滤级别 |

容器内部绑定 `0.0.0.0`；在容器之外 `serve` 仍保留其 `127.0.0.1` 默认绑定。
**请勿提交 `.env`** —— 它可能包含 hub token。

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🔌 HTTP API

`aria-engine serve` 基于已下载的 AFM checkpoint 启动本地 System One HTTP 服务。**不做** aria-router 注册。

| 旗标 | 默认 | 含义 |
| --- | --- | --- |
| `--track` | `encoder` | `encoder` \| `decoder` |
| `--model-name` | `afm-de` / `afm-dd`（随 `--track`） | 响应中的 model id；省略 `--checkpoint` 时兼作 checkpoint |
| `--checkpoint` | `~/.ariacompute/models/<model-name>` | checkpoint 目录，或 `~/.ariacompute/models/` 下的缓存名 |
| `--bind` | `127.0.0.1:8010`（encoder）/ `127.0.0.1:8011`（decoder） | 监听地址 |
| `--compute` | 来自 `engine.yml`（`auto`） | `auto` \| `cpu` \| `cuda`（覆盖 setup；用于设备选择日志） |

`--model-name` 与 `--checkpoint` 二选一或同时提供。软布局检查：encoder 轨指向 decoder 形态目录（或相反）会直接报错。

| 方法 | 路径 | 响应 |
| --- | --- | --- |
| `GET` | `/health` 或 `/` | `{ "ok": true, "service": "afm-engine", "model": "<model-name>" }` |
| `GET` | `/v1/models` | OpenAI 风格列表，含一条 `--model-name` |
| `POST` | `/v1/systemone` | `{ "answers": { … }, "model": "<model-name>" }` |

`POST /v1/systemone` 请求体与 Decide 的 System One 形态相同：`{ "state": …, "questions": { "<qid>": { "type", "instructions", "criteria" } } }`（`type` = `choice` \| `score` \| `noul`）。题目非法 → HTTP **422**；打分失败 → **500**。

```bash
# Encoder（默认轨）用缓存名
aria-engine download afm-de
aria-engine serve --model-name afm-de
# → http://127.0.0.1:8010

# 显式 track + checkpoint 路径
aria-engine download afm-dd
aria-engine serve --track decoder --checkpoint ~/.ariacompute/models/afm-dd

# 自定义 bind + compute 覆盖
aria-engine serve --model-name afm-de --bind 0.0.0.0:8010 --compute cpu

# Health / models
curl -s http://127.0.0.1:8010/health
curl -s http://127.0.0.1:8010/v1/models

# System One — choice
curl -s http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "user wants a refund",
    "questions": {
      "q1": {
        "type": "choice",
        "instructions": "Pick the best action",
        "criteria": {
          "refund": "issue a full refund",
          "deny": "deny the request"
        }
      }
    }
  }'

# System One — score（criteria：2–10 个有序等级）
curl -s http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "ticket sat in queue for three days with no reply",
    "questions": {
      "q1": {
        "type": "score",
        "instructions": "Rate urgency from 0 (low) to 3 (critical)",
        "criteria": [
          "can wait until next week",
          "should be handled this week",
          "needs attention today",
          "page on-call now"
        ]
      }
    }
  }'

# System One — noul（true / false）
curl -s http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "customer wrote: this is unacceptable, I want a manager",
    "questions": {
      "q1": {
        "type": "noul",
        "instructions": "Is the user angry?",
        "criteria": {
          "false": "calm or neutral tone",
          "true": "angry or escalating"
        }
      }
    }
  }'
```

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🧭 Decide（类型化决策）

对一份 JSON（stdin 或 `--file`）做一次性 typed decision；每个题/记录打印一份 pretty JSON 答案。打包与 typed-answer 契约与 `POST /v1/systemone` 相同。

| 旗标 | 默认 | 含义 |
| --- | --- | --- |
| `--track` | `encoder` | `encoder` \| `decoder` |
| `--model-name` | `afm-de` / `afm-dd`（随 `--track`） | 缓存名；省略 `--checkpoint` 时作为 checkpoint |
| `--checkpoint` | `~/.ariacompute/models/<model-name>` | checkpoint 目录，或 `~/.ariacompute/models/` 下的缓存名 |
| `--file` | _(stdin)_ | 输入 JSON 路径 |
| `--logits` | — | Encoder **黄金路径**：选项原始 logits 的 JSON 数组（跳过权重前向） |
| `--semif-out` | — | Decoder **黄金路径**：SemIf 风格 JSON（跳过权重前向） |

未传 `--logits` / `--semif-out` 时，会对 checkpoint 做真实前向（需完整本地模型；见 [`task.md`](task.md) 阶段 B）。

**1. AFM-D record**（单题）— `task` + `options` + `state`：

```json
{
  "id": "choice-1",
  "task": "choice",
  "instructions": "Select the fruit",
  "options": [
    {"name": "apple", "description": "a red fruit"},
    {"name": "carrot", "description": "an orange vegetable"}
  ],
  "state": "The user wants something sweet and red."
}
```

```json
{
  "id": "score-1",
  "task": "score",
  "instructions": "Rate urgency from 0 (low) to 3 (critical)",
  "options": [
    {"name": "level-0", "description": "can wait until next week"},
    {"name": "level-1", "description": "should be handled this week"},
    {"name": "level-2", "description": "needs attention today"},
    {"name": "level-3", "description": "page on-call now"}
  ],
  "state": "ticket sat in queue for three days with no reply"
}
```

```json
{
  "id": "noul-1",
  "task": "noul",
  "instructions": "Is the user angry?",
  "options": [
    {"name": "false", "description": "calm or neutral tone"},
    {"name": "true", "description": "angry or escalating"}
  ],
  "state": "customer wrote: this is unacceptable, I want a manager"
}
```

`task` 为 `choice` \| `score` \| `noul`。`noul` 选项名规范化为 `true` / `false`（接受 `yes` / `no` 别名）。仓库夹具：[`tests/fixtures/encoder_choice.json`](tests/fixtures/encoder_choice.json)、[`encoder_noul.json`](tests/fixtures/encoder_noul.json)、[`decoder_choice.json`](tests/fixtures/decoder_choice.json)。

**2. System One body**（可多题）— 与 HTTP serve 同形：

```json
{
  "state": "user wants a refund",
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "Pick the best action",
      "criteria": {
        "refund": "issue a full refund",
        "deny": "deny the request"
      }
    },
    "q2": {
      "type": "score",
      "instructions": "Rate urgency from 0 (low) to 3 (critical)",
      "criteria": [
        "can wait until next week",
        "should be handled this week",
        "needs attention today",
        "page on-call now"
      ]
    },
    "q3": {
      "type": "noul",
      "instructions": "Is the user angry?",
      "criteria": {
        "false": "calm",
        "true": "angry"
      }
    }
  }
}
```

| `type` | `criteria` |
| --- | --- |
| `choice` | 对象：选项名 → 描述 |
| `score` | 数组：2–10 个有序等级文案 |
| `noul` | 对象含 `true` / `false`（可省略，有默认文案） |

```bash
# Encoder 黄金路径（无权重前向）— logits 长度 = 选项数
aria-engine decide \
  --track encoder \
  --model-name afm-de \
  --logits '[2.5, 0.1]' \
  --file tests/fixtures/encoder_choice.json

# 从 stdin；--checkpoint 可覆盖 model-name 路径
aria-engine decide --track encoder --checkpoint afm-de --logits '[2.5,0.1]' \
  < tests/fixtures/encoder_choice.json

# Decoder 黄金路径 — SemIf out JSON
aria-engine decide \
  --track decoder \
  --model-name afm-dd \
  --semif-out '{"option_ids":["a","b"],"probabilities":[0.1,0.9]}' \
  --file tests/fixtures/decoder_choice.json

# System One — choice（encoder 黄金路径；两选项 → 两 logits）
cat <<'EOF' | aria-engine decide --track encoder --model-name afm-de --logits '[1.0, 0.2]'
{
  "state": "user wants a refund",
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "Pick the best action",
      "criteria": {"refund": "issue a full refund", "deny": "deny the request"}
    }
  }
}
EOF

# System One — score（四个等级 → 四个 logits）
cat <<'EOF' | aria-engine decide --track encoder --model-name afm-de --logits '[0.1, 0.4, 2.0, 0.3]'
{
  "state": "ticket sat in queue for three days with no reply",
  "questions": {
    "q1": {
      "type": "score",
      "instructions": "Rate urgency from 0 (low) to 3 (critical)",
      "criteria": [
        "can wait until next week",
        "should be handled this week",
        "needs attention today",
        "page on-call now"
      ]
    }
  }
}
EOF

# System One — noul（false/true → 两 logits；也可用夹具）
aria-engine decide \
  --track encoder \
  --model-name afm-de \
  --logits '[0.2, 1.8]' \
  --file tests/fixtures/encoder_noul.json

cat <<'EOF' | aria-engine decide --track encoder --model-name afm-de --logits '[0.2, 1.8]'
{
  "state": "customer wrote: this is unacceptable, I want a manager",
  "questions": {
    "q1": {
      "type": "noul",
      "instructions": "Is the user angry?",
      "criteria": {"false": "calm or neutral tone", "true": "angry or escalating"}
    }
  }
}
EOF

# 真实前向（需已下载完整 checkpoint）
aria-engine decide --track encoder --model-name afm-de --file record.json
```

`--model-name afm-de`（或 `--checkpoint afm-de`）在路径存在时解析为 `~/.ariacompute/models/afm-de`（与 `serve` 相同）。

每个 record 打印一份 typed answer（如 `type`、`choice` / `score` / `noul`、`probabilities`、`confidence`）。System One 多题时按题顺序依次输出多个 JSON 文档。

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🧩 FFI 与语言 SDK

头文件：[`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`、`aria_systemone`、`aria_model_destroy`、`aria_last_error`。

发布资产：`aria-engine_${VERSION}_${OS}.tar.gz|zip`、`libaria-engine_ffi_${VERSION}_${OS}.tar.gz`。

原生 C ABI（`ariacompute-ffi` / `libaria-engine_ffi`）以及 `bindings/` 下的薄封装。API 面是 **System One**（`systemone`），不是 chat/complete。

| Binding | 路径 | Registry |
| --- | --- | --- |
| Rust | `bindings/rust`（`ariacompute-engine`） | crates.io |
| Python | `bindings/python` | PyPI `aria-engine` |
| Go | `bindings/go` | Go module |
| TypeScript | `bindings/typescript` | npm `@ariacompute/engine-ts` |
| React Native | `bindings/react-native` | npm `@ariacompute/engine-rn` |
| Flutter | `bindings/flutter` | pub.dev `aria_engine` |
| Swift | `bindings/swift` | CocoaPods `AriaEngine` |
| Kotlin | `bindings/kotlin` | Maven Central `com.ariacompute:engine` |

走 FFI 的 SDK 需要 `libaria-engine_ffi`（`aria-engine upgrade`、包内捆绑，或 `ARIA_FFI_LIB`）。`checkpoint` 指向已下载的 AFM 目录（例如 `aria-engine download afm-de` 后的 `~/.ariacompute/models/afm-de`）。Rust crate 原生链接 workspace crates，一般无需解压动态库。

```bash
cargo test -p ariacompute-ffi -p ariacompute-engine
./scripts/run-binding-tests.sh   # 主机矩阵（Rust / Python / Go / TS）
```

**Python**（`aria-engine`；`ARIA_FFI_LIB` 可覆盖）：

```bash
pip install aria-engine
# 可选：export ARIA_FFI_LIB=/path/to/libaria-engine_ffi.so
```

```python
from aria_engine import AriaEngine

eng = AriaEngine("/path/to/afm-de", "encoder")  # 或 ~/.ariacompute/models/afm-de
out = eng.systemone({
    "state": "user wants a refund",
    "questions": {
        "q1": {
            "type": "choice",
            "instructions": "Pick the best action",
            "criteria": {
                "refund": "issue a full refund",
                "deny": "deny the request",
            },
        }
    },
})
print(out["answers"]["q1"])
eng.destroy()
```

**TypeScript / Node**（`@ariacompute/engine-ts`）：

```bash
npm install @ariacompute/engine-ts
# 可选：export ARIA_FFI_LIB=/path/to/libaria-engine_ffi.so
```

```ts
import { AriaEngine } from "@ariacompute/engine-ts";

const eng = new AriaEngine("/path/to/afm-de", "encoder");
const out = eng.systemone({
  state: "user wants a refund",
  questions: {
    q1: {
      type: "choice",
      instructions: "Pick the best action",
      criteria: {
        refund: "issue a full refund",
        deny: "deny the request",
      },
    },
  },
}) as { answers: Record<string, unknown> };
console.log(out.answers.q1);
eng.destroy();
```

**React Native**（`@ariacompute/engine-rn`）：

```bash
npm install @ariacompute/engine-rn
```

```js
const { AriaEngine } = require("@ariacompute/engine-rn");

const eng = new AriaEngine("/path/to/afm-de", "encoder");
const out = await eng.systemone({
  state: "user wants a refund",
  questions: {
    q1: {
      type: "choice",
      instructions: "Pick the best action",
      "criteria": {
        refund: "issue a full refund",
        deny: "deny the request",
      },
    },
  },
});
console.log(out.answers.q1);
eng.destroy();
```

**Go**（cgo；链接 `libaria_ffi` / `libaria-engine_ffi`）：

```bash
export CGO_ENABLED=1
# 可选：export ARIA_FFI_LIB=/path/to/libaria-engine_ffi.so
go get github.com/ariacompute/engine/bindings/go@latest
```

```go
package main

import (
	"fmt"

	aria "github.com/ariacompute/engine/bindings/go"
)

func main() {
	eng, err := aria.Open("/path/to/afm-de", "encoder")
	if err != nil {
		panic(err)
	}
	defer eng.Destroy()
	out, err := eng.SystemOne(map[string]any{
		"state": "user wants a refund",
		"questions": map[string]any{
			"q1": map[string]any{
				"type":         "choice",
				"instructions": "Pick the best action",
				"criteria": map[string]any{
					"refund": "issue a full refund",
					"deny":   "deny the request",
				},
			},
		},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(out["answers"])
}
```

**Rust**（`ariacompute-engine` — 原生 API；一般无需解压 `libaria-engine_ffi`）：

```bash
cargo add ariacompute-engine
```

```rust
use aria_engine::{Engine, EngineTrack, SystemOneRequest};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eng = Engine::open("/path/to/afm-de", EngineTrack::Encoder)?;
    let req: SystemOneRequest = serde_json::from_value(json!({
        "state": "user wants a refund",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Pick the best action",
                "criteria": {
                    "refund": "issue a full refund",
                    "deny": "deny the request"
                }
            }
        }
    }))?;
    let out = eng.systemone(&req)?;
    println!("{out}");
    Ok(())
}
```

**Flutter**（`package:aria_engine`）：

```yaml
# pubspec.yaml
dependencies:
  aria_engine: ^0.1.0
```

```dart
import 'package:aria_engine/aria_engine.dart';

final eng = AriaEngine('/path/to/afm-de', track: 'encoder');
final out = eng.systemone({
  'state': 'user wants a refund',
  'questions': {
    'q1': {
      'type': 'choice',
      'instructions': 'Pick the best action',
      'criteria': {
        'refund': 'issue a full refund',
        'deny': 'deny the request',
      },
    },
  },
});
print(out['answers']);
eng.destroy();
```

**Swift**（CocoaPods `AriaEngine`）：

```ruby
# Podfile
pod 'AriaEngine'
```

```swift
import AriaEngine

let eng = AriaEngine(checkpoint: "/path/to/afm-de", track: "encoder")
let req = """
{
  "state": "user wants a refund",
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "Pick the best action",
      "criteria": {
        "refund": "issue a full refund",
        "deny": "deny the request"
      }
    }
  }
}
"""
let out = try eng.systemone(requestJson: req)
print(out)
eng.destroy()
```

**Kotlin**（`com.ariacompute:engine`）：

```kotlin
// build.gradle.kts
implementation("com.ariacompute:engine:0.1.0")
```

```kotlin
import com.ariacompute.engine.AriaEngine
import org.json.JSONObject

val eng = AriaEngine("/path/to/afm-de", "encoder")
val out = eng.systemone(
    JSONObject(
        """
        {
          "state": "user wants a refund",
          "questions": {
            "q1": {
              "type": "choice",
              "instructions": "Pick the best action",
              "criteria": {
                "refund": "issue a full refund",
                "deny": "deny the request"
              }
            }
          }
        }
        """.trimIndent()
    )
)
println(out.getJSONObject("answers"))
eng.destroy()
```

各语言更多说明见 `bindings/*/README.md`。

GitHub Release 触发：`release.yml`（CLI + FFI 资产），以及 `publish-cargo.yml`、`publish-npm.yml`、`publish-pub.yml`、`publish-pypi.yml`、`publish-maven.yml`、`publish-cocoapods.yml`。

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🔏 信任库安全

2026-10 的 eMudhra（emSign）中间 CA 私钥泄露事件，由 Rust 侧兜底：`aria-engine` / `aria-cli` 二进制做 hub 下载等出向 TLS 时，经 `reqwest` 与 `hf-hub`（均为 `rustls-tls`）使用内嵌的 **`webpki-roots`**，并在 `cli/Cargo.toml` 中钉死 `>=1.0.10`（剔除被泄露的 emSign 根）。Docker 构建已安装 `ca-certificates`（系统 store），但二进制内嵌 `webpki-roots`，因此锁升级才是决定性修复。各语言绑定（Go / Python / TS / RN）均为 FFI-only，自身不做任何出向 TLS，故其运行时信任库不在范围内；真实的网络路径就是上文的 Rust 侧。

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🔨 开发

| 目标 | 作用 |
| --- | --- |
| `cargo build --workspace` | 构建全部 crate |
| `cargo test --workspace` | 跑完整 workspace 测试套件 |
| `cargo build --release -p aria-cli -p ariacompute-ffi` | Release 二进制 + FFI cdylib |
| `cargo build -p aria-cli --features cuda` | 以 CUDA 后端构建（传到 de/dd） |
| `cargo test -p ariacompute-ffi -p ariacompute-engine` | FFI + 原生 Rust SDK 测试 |
| `./scripts/run-binding-tests.sh` | 主机矩阵（Rust / Python / Go / TS） |
| `./scripts/prepare-binding-fixture.sh` | 构建本地 SDK 测试用的 binding fixture |

```bash
cargo test --workspace
./scripts/prepare-binding-fixture.sh
./scripts/run-binding-tests.sh
```

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 📝 许可证

MIT，详见 [LICENSE](./LICENSE)。

<div align="right">

[![][back-to-top]](#readme-top)

</div>

## 🤝 工程约定

- [`AGENTS.md`](./AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](./requirements.md)：需求规格（功能边界 / 异常 / 验收标准，经人工审核门控）
- [`task.md`](./task.md)：实施任务勾选清单

<div align="right">

[![][back-to-top]](#readme-top)

</div>

---

<div align="center">

Copyright © 2026 [ariacompute][org]. <br/>
This project is [MIT](./LICENSE) licensed.

</div>

<!-- 链接定义 -->

[org]: https://github.com/ariacompute

[release-shield]: https://img.shields.io/github/v/release/ariacompute/engine?style=flat-square&labelColor=black&color=555
[release-link]: https://github.com/ariacompute/engine/releases
[rust-shield]: https://img.shields.io/badge/rust-edition_2021-555?style=flat-square&labelColor=black
[rust-link]: ./Cargo.toml
[license-shield]: https://img.shields.io/github/license/ariacompute/engine?style=flat-square&labelColor=black&color=555
[license-link]: ./LICENSE
[lastcommit-shield]: https://img.shields.io/github/last-commit/ariacompute/engine?style=flat-square&labelColor=black&color=555
[lastcommit-link]: https://github.com/ariacompute/engine/commits

[httpapi-shield]: https://img.shields.io/badge/HTTP_API-3_routes-555?style=flat-square&labelColor=black
[httpapi-link]: #-http-api
[tests-shield]: https://img.shields.io/badge/tests-cargo_test-555?style=flat-square&labelColor=black
[tests-link]: ./Cargo.toml
[sdks-shield]: https://img.shields.io/badge/SDKs-8_languages-555?style=flat-square&labelColor=black
[sdks-link]: #-ffi-与语言-sdk
[trust-shield]: https://img.shields.io/badge/trust-webpki_roots_1.0.10-555?style=flat-square&labelColor=black
[trust-link]: #-信任库安全

[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square
