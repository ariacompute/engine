# ariaengine (Python)

`ariaengine` 是 Aria 推理引擎的 Python SDK（ctypes 封装 `libariaengine_ffi`）。
通过 PyPI 安装的 **平台 wheel 内已捆绑动态库**，开箱即用，无需手动设置环境变量。

## 安装（PyPI 发布版）

```bash
pip install ariaengine
```

> 支持 Linux (x86_64 / aarch64 manylinux)、macOS (x86_64 / arm64)、Windows (x86_64)，
> Python 3.10+。wheel 内自带 `libariaengine_ffi.so` / `.dylib` / `.dll`。

## 用法

```python
from ariaengine import Engine
with Engine("/path/to/bundle") as eng:
    print(eng.complete([{"role":"user","content":"hi"}], {"max_tokens": 16}))
```

## 按模型名自动下载

`Engine` 同时接受本地 bundle 路径或 Aria 模型名（如 `gemma-4-e2b-it_q4`）。含 `/`
或本地已存在的值直接加载；否则由 SDK 从**本区公开 hub** 自动下载（`.com` → Hugging Face，
`.cn` → ModelScope；`site` 默认 `https://ariacompute.com`）到 `~/.ariacompute/models/{model}` 再加载。
**不再请求 Dashboard**（避免 HTTP 403）。需授权的 hub 文件请传入 `hf_token`（`.com`）或
`modelscope_api_token`（`.cn`），字段名与 `ariaengine setup` 相同；未传则读
`~/.ariacompute/engine.yml`。Dashboard `sk-` / `bfvk-` token 不会当作 hub 凭证。
不读环境变量 `HF_TOKEN` / `MODELSCOPE_API_TOKEN`。
缓存中已有有效 bundle 时直接复用，不重复下载。

```python
with Engine("gemma-4-e2b-it_q4") as eng:
    print(eng.complete([{"role":"user","content":"hi"}], {"max_tokens": 16}))

# Gated Hugging Face files (instance setup; does not write engine.yml):
eng = Engine()
eng.setup(hf_token="hf_...")
eng.open("gemma-4-e2b-it_q4")
print(eng.complete([{"role":"user","content":"hi"}], {"max_tokens": 16}))
```

`Engine.setup` sets Config / Run fields on **that instance only**. CLI `ariaengine setup` still writes `~/.ariacompute/engine.yml`; the SDK never does. Empty fields still fall back to reading that file for hub download. Also: `eng.setup_status()`, `eng.setup_clear()`.

## 动态库查找顺序

`_load_lib()` 按以下顺序解析：

1. 显式传入的 `path`
2. 环境变量 `ARIAENGINE_FFI_LIB`（源码安装 / 自定义路径时使用）
3. wheel 内捆绑的 `ariaengine/lib/<libariaengine_ffi.so|dylib|dll>`
4. `~/.ariacompute/lib/`（与 `ariaengine upgrade` 相同目录）
5. 都找不到 → 从本区 Releases 下载最新正式版 `libariaengine_ffi_{ver}_{os}.tar.gz` 解压到 `~/.ariacompute/lib/`（`upgrade_url` 优先；否则 `.com` → GitHub，`.cn` → Gitee）。失败明确报错。

`ariaengine download` 只拉模型 bundle，不会装 FFI；Python SDK 在 `Engine(...)` 加载前自动补齐。

## 源码安装（开发）

```bash
export ARIAENGINE_FFI_LIB=/path/to/libariaengine_ffi.so
pip install -e .
```

## 构建 wheel（cibuildwheel）

```bash
bash scripts/build-python-ffi.sh   # 编译 libariaengine_ffi 并拷入 ariaengine/lib/
pip install cibuildwheel
cibuildwheel --platform linux      # 或 macos / windows
```

## 发布到 PyPI

GitHub Release 触发 `.github/workflows/publish-pypi.yml`：五平台构建 + `twine upload`。
需要仓库 Secret `PYPI_TOKEN`（PyPI API token，project-scoped，owner `ariacompute`）。
版本号取 tag 去 `v`（也支持 `workflow_dispatch` 手动输入）。
