# Aria Go binding

Requires `ARIA_INCLUDE`, `ARIA_LIBDIR` (or pkg-config), and build tag `aria_ffi`:

```bash
export ARIA_INCLUDE=../../ffi/include
export ARIA_LIBDIR=../../target/debug
export ARIA_BUNDLE=/path/to/tiny-q4
go test -tags aria_ffi ./...
```

Go modules are consumed via the Git release tag (`proxy.golang.org`); no separate registry job.

## Auto-download by model name

`aria.OpenModel(modelRef, token, site)` accepts a local bundle path **or** an
Aria model name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or already on
disk is loaded directly; otherwise the SDK downloads it from the Dashboard
private source (requires `token`; `site` defaults to `https://ariacompute.com`)
into `~/.ariacompute/models/{model}` and then loads it. A valid cached bundle is
reused without re-downloading.

```go
eng, err := aria.OpenModel("gemma-4-e2b-it_q4", "API_TOKEN", "")
```
