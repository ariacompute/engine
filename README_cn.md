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

## 配置（`aria-engine setup`）

写入 `~/.ariacompute/engine.yml`（**无** router 字段）：

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

## CLI

```bash
aria-engine download afm-de       # HF/ModelScope HTTP API（site_url + setup token）；不依赖 hub CLI
                                  # .cn → ModelScope /api/v1/.../repo?FilePath=… ；.com → HF resolve
aria-engine list | check | clean  # list = 本地 ∪ Hub 组织（ariacompute / AriaCompute）；标记 downloaded|not downloaded
                                  # check/clean：仅完整 checkpoint；失败下载会清理
aria-engine upgrade [version]     # 需要 upgrade_url
aria-engine serve …               # 见下方 Serve
aria-engine decide …              # 见下方 Decide
aria-engine -v
```

## Serve（`aria-engine serve`）

基于已下载的 AFM checkpoint 启动本地 System One HTTP 服务。**不做** aria-router 注册。

### 旗标

| 旗标 | 默认 | 含义 |
| --- | --- | --- |
| `--track` | `encoder` | `encoder` \| `decoder` |
| `--model-name` | `afm-de` / `afm-dd`（随 `--track`） | 响应中的 model id；省略 `--checkpoint` 时兼作 checkpoint |
| `--checkpoint` | `~/.ariacompute/models/<model-name>` | checkpoint 目录，或 `~/.ariacompute/models/` 下的缓存名 |
| `--bind` | `127.0.0.1:8010`（encoder）/ `127.0.0.1:8011`（decoder） | 监听地址 |
| `--compute` | 来自 `engine.yml`（`auto`） | `auto` \| `cpu` \| `cuda`（覆盖 setup；用于设备选择日志） |

`--model-name` 与 `--checkpoint` 二选一或同时提供。软布局检查：encoder 轨指向 decoder 形态目录（或相反）会直接报错。

### 端点

| 方法 | 路径 | 响应 |
| --- | --- | --- |
| `GET` | `/health` 或 `/` | `{ "ok": true, "service": "afm-engine", "model": "<model-name>" }` |
| `GET` | `/v1/models` | OpenAI 风格列表，含一条 `--model-name` |
| `POST` | `/v1/systemone` | `{ "answers": { … }, "model": "<model-name>" }` |

`POST /v1/systemone` 请求体与 Decide 的 System One 形态相同：`{ "state": …, "questions": { "<qid>": { "type", "instructions", "criteria" } } }`（`type` = `choice` \| `score` \| `noul`）。题目非法 → HTTP **422**；打分失败 → **500**。

### 示例

```bash
# Encoder（默认轨）用缓存名
aria-engine download afm-de
aria-engine serve --model-name afm-de
# → http://127.0.0.1:8010

# 显式 track + checkpoint 路径
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

真实权重前向需要完整本地 checkpoint（见 [`task.md`](task.md) 阶段 B）。此前可用 `decide` 的 `--logits` / `--semif-out` 做黄金路径打分。

## Decide（`aria-engine decide`）

对一份 JSON（stdin 或 `--file`）做一次性 typed decision；每个题/记录打印一份 pretty JSON 答案。打包与 typed-answer 契约与 `POST /v1/systemone` 相同。

### 旗标

| 旗标 | 默认 | 含义 |
| --- | --- | --- |
| `--track` | `encoder` | `encoder` \| `decoder` |
| `--model-name` | `afm-de` / `afm-dd`（随 `--track`） | 缓存名；省略 `--checkpoint` 时作为 checkpoint |
| `--checkpoint` | `~/.ariacompute/models/<model-name>` | checkpoint 目录，或 `~/.ariacompute/models/` 下的缓存名 |
| `--file` | _(stdin)_ | 输入 JSON 路径 |
| `--logits` | — | Encoder **黄金路径**：选项原始 logits 的 JSON 数组（跳过权重前向） |
| `--semif-out` | — | Decoder **黄金路径**：SemIf 风格 JSON（跳过权重前向） |

未传 `--logits` / `--semif-out` 时，会对 checkpoint 做真实前向（需完整本地模型；见 [`task.md`](task.md) 阶段 B）。

### 输入形态

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

### 示例

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

### 输出

每个 record 打印一份 typed answer（如 `type`、`choice` / `score` / `noul`、`probabilities`、`confidence`）。System One 多题时按题顺序依次输出多个 JSON 文档。

## FFI

头文件：[`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`、`aria_systemone`、`aria_model_destroy`。

发布资产：`aria-engine_${VERSION}_${OS}.tar.gz|zip`、`libaria-engine_ffi_${VERSION}_${OS}.tar.gz`。

## 语言 SDK

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

C 头文件：[`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`、`aria_systemone`、`aria_model_destroy`、`aria_last_error`。

走 FFI 的 SDK 需要 `libaria-engine_ffi`（`aria-engine upgrade`、包内捆绑，或 `ARIA_FFI_LIB`）。`checkpoint` 指向已下载的 AFM 目录（例如 `aria-engine download afm-de` 后的 `~/.ariacompute/models/afm-de`）。Rust crate 原生链接 workspace crates，一般无需解压动态库。

```bash
cargo test -p ariacompute-ffi -p ariacompute-engine
./scripts/run-binding-tests.sh   # 主机矩阵（Rust / Python / Go / TS）
```

### 示例

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
      criteria: {
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

## 发布工作流

GitHub Release 触发：`release.yml`（CLI + FFI 资产），以及 `publish-cargo.yml`、`publish-npm.yml`、`publish-pub.yml`、`publish-pypi.yml`、`publish-maven.yml`、`publish-cocoapods.yml`。

## Docker / Docker Compose 部署

将 System One HTTP 服务以容器方式运行。仓库附带一个参数化 `Dockerfile`、编排脚本
`docker/entrypoint.sh`、CPU 版 `docker-compose.yml`（两个服务：`engine-encoder`
在 **8010**、`engine-decoder` 在 **8011**），以及 GPU 覆盖文件
`docker-compose.cuda.yml`。

### 构建与运行（CPU）

```bash
cp .env.example .env          # 设置 SITE_URL / HF_TOKEN 或 MODELSCOPE_API_TOKEN / COMPUTE
docker compose build
docker compose up -d
```

两个服务共享挂载在 `/data/aria` 的 `aria-data` 命名卷，其中存放 `engine.yml`
（配置）与 `models/`（checkpoint）。entrypoint 仅在 `engine.yml` 缺失时依据环境变量
生成它，**已挂载的配置会被尊重**（不会被覆盖）。

### 模型

- **离线（默认）**：预先用宿主的已下载权重填充该卷
  （在宿主执行 `aria-engine download afm-de && aria-engine download afm-dd`，
  或挂载一个包含 `~/.ariacompute/models/*` 的宿主目录）。
- **自动下载**：在 `.env` 中设置 `AUTO_DOWNLOAD=1`，entrypoint 会在首次运行时
  执行 `aria-engine download <model>`（需联网与对应 hub token）。默认关闭。

### GPU（CUDA）

需要宿主机已装 NVIDIA 驱动 + `nvidia-container-toolkit`。覆盖文件会将构建基础切换为
CUDA 镜像、以 `--features cuda` 编译，并为每个服务申请一块 GPU：

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml build
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```

设置 `COMPUTE=cuda`（覆盖文件已设置），让 candle 选择 CUDA 后端。

### 配置（`.env`）

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `ENGINE_IMAGE` | `aria-engine:latest` | 构建镜像的标签 |
| `BUILD_IMAGE` / `RUNTIME_IMAGE` | `rust:1-slim` / `debian:bookworm-slim` | CPU 构建/运行基础镜像 |
| `FEATURES` | _(空)_ | 构建特性；CUDA 覆盖文件设为 `cuda` |
| `SITE_URL` / `UPGRADE_URL` | _(空)_ | 区域 hub（`.com`→HF，`.cn`→ModelScope） |
| `HF_TOKEN` / `MODELSCOPE_API_TOKEN` | _(空)_ | 当前区域的 hub token |
| `COMPUTE` | `auto` | `auto` \| `cpu` \| `cuda` |
| `ENCODER_MODEL` / `ENCODER_PORT` | `afm-de` / `8010` | encoder 服务模型与端口 |
| `DECODER_MODEL` / `DECODER_PORT` | `afm-dd` / `8011` | decoder 服务模型与端口 |
| `AUTO_DOWNLOAD` | `0` | 设为 `1` 可在首次运行时拉取权重 |
| `RUST_LOG` | `info` | Rust tracing 过滤级别 |

### 验证

```bash
curl -s http://localhost:8010/health     # {"ok":true,"service":"afm-engine","model":"afm-de"}
curl -s http://localhost:8011/health     # decoder
docker compose ps                        # 两个服务均 healthy
```

容器内部绑定 `0.0.0.0`；在容器之外 `serve` 仍保留其 `127.0.0.1` 默认绑定。
**请勿提交 `.env`** —— 它可能包含 hub token。

## 许可证

MIT。

## 工程约定

本仓库遵循 Harness Engineering 理念：

- [`AGENTS.md`](AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](requirements.md)：需求规格（功能边界 / 异常 / 验收标准，经人工审核门控）
- [`task.md`](task.md)：实施任务勾选清单
