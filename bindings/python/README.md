# aria-engine (Python)

`aria-engine` 是 Aria 推理引擎的 Python SDK（ctypes 封装 `libaria_ffi`）。
通过 PyPI 安装的 **平台 wheel 内已捆绑动态库**，开箱即用，无需手动设置环境变量。

## 安装（PyPI 发布版）

```bash
pip install aria-engine
```

> 支持 Linux (x86_64 / aarch64 manylinux)、macOS (x86_64 / arm64)、Windows (x86_64)，
> Python 3.10+。wheel 内自带 `libaria_ffi.so` / `.dylib` / `.dll`。

## 用法

```python
from aria_engine import Engine
with Engine("/path/to/bundle") as eng:
    print(eng.complete([{"role":"user","content":"hi"}], {"max_tokens": 16}))
```

## 按模型名自动下载

`Engine` 同时接受本地 bundle 路径或 Aria 模型名（如 `gemma-4-e2b-it_q4`）。含 `/`
或本地已存在的值直接加载；否则由 SDK 从 **Dashboard 私有源** 自动下载（需 `token`，
`site` 默认 `https://ariacompute.com`）到 `~/.ariacompute/models/{model}` 再加载。
缓存中已有有效 bundle 时直接复用，不重复下载。

```python
with Engine("gemma-4-e2b-it_q4", token="API_TOKEN") as eng:
    print(eng.complete([{"role":"user","content":"hi"}], {"max_tokens": 16}))
```

## 动态库查找顺序

`_load_lib()` 按以下顺序解析：

1. 显式传入的 `path`
2. 环境变量 `ARIA_FFI_LIB`（源码安装 / 自定义路径时使用）
3. wheel 内捆绑的 `aria_engine/lib/<libaria_ffi.so|dylib|dll>`
4. 都找不到 → `RuntimeError`（提示安装 wheel 或设置 `ARIA_FFI_LIB`）

## 源码安装（开发）

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
pip install -e .
```

## 构建 wheel（cibuildwheel）

```bash
bash scripts/build-python-ffi.sh   # 编译 libaria_ffi 并拷入 aria_engine/lib/
pip install cibuildwheel
cibuildwheel --platform linux      # 或 macos / windows
```

## 发布到 PyPI

GitHub Release 触发 `.github/workflows/publish-pypi.yml`：五平台构建 + `twine upload`。
需要仓库 Secret `PYPI_TOKEN`（PyPI API token，project-scoped，owner `ariacompute`）。
版本号取 tag 去 `v`（也支持 `workflow_dispatch` 手动输入）。
