
## ariaengine 命名统一（requirements §3.4 / §3.7）

### T94 — CLI / Release / 核心 crate / FFI / SDK 重命名
- [x] CLI 二进制 `ariaengine`；Release `ariaengine_*` + `libariaengine_ffi_*`
- [x] 核心 crate：`ariaengine-openai`、`ariaengine-{kernel,graph,inference}`；FFI `ariacompute-ariaengine-ffi` / `ariaengine_ffi`
- [x] 环境变量 `ARIAENGINE_VERSION`、`ARIAENGINE_BIN`、`ARIAENGINE_FFI_LIB`
- [x] SDK：PyPI `ariaengine`、Go `package ariaengine`、Rust `ariacompute-ariaengine`、Flutter `ariaengine`、npm `@ariacompute/ariaengine-{ts,rn}`
- [x] serve `engine_download` 解析 `ariaengine_` / `libariaengine_ffi_` 前缀
- [x] router/model 文档字符串同步
