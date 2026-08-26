# Aria Go binding

Requires `ARIA_INCLUDE`, `ARIA_LIBDIR` (or pkg-config), and build tag `aria_ffi`:

```bash
export ARIA_INCLUDE=../../ffi/include
export ARIA_LIBDIR=../../target/debug
export ARIA_BUNDLE=/path/to/tiny-q4
go test -tags aria_ffi ./...
```

Go modules are consumed via the Git release tag (`proxy.golang.org`); no separate registry job.

`Open` / `OpenModel` also install `libaria_ffi` into `~/.ariacompute/lib/` when it is not already on `ARIA_FFI_LIB` or in that cache directory (same Releases asset as `aria-engine upgrade`). Set `ARIA_LIBDIR` / `LD_LIBRARY_PATH` so the cgo-linked binary can load it, or keep using a locally built `target/debug`.

## Auto-download by model name

`aria.OpenModel(modelRef, token, site)` accepts a local bundle path **or** an
Aria model name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or already on
disk is loaded directly; otherwise the SDK downloads it from the regional public
hub (same as `aria-engine download`: `.com` → Hugging Face, `.cn` → ModelScope;
`site` defaults to `https://ariacompute.com`) into `~/.ariacompute/models/{model}`
and then loads it. Dashboard is not used. A Dashboard `sk-` / `bfvk-` token is
ignored for hub auth. Token is optional for public models. A valid cached bundle
is reused without re-downloading.

Gated files: pass `HFToken` (`.com`) or `ModelScopeAPIToken` (`.cn`) — same keys as
`aria-engine auth` (`hf_token` / `modelscope_api_token`). If omitted, the SDK reads
`~/.ariacompute/config.yml`.

```go
eng, err := aria.OpenModel("gemma-4-e2b-it_q4", "", "")
eng, err = aria.OpenModelOpts("gemma-4-e2b-it_q4", aria.DownloadOptions{HFToken: "hf_..."})
```
