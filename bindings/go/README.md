# Aria Go binding

Requires `ARIA_INCLUDE`, `ARIA_LIBDIR` (or pkg-config), and build tag `aria_ffi`:

```bash
export ARIA_INCLUDE=../../ffi/include
export ARIA_LIBDIR=../../target/debug
export ARIA_BUNDLE=/path/to/tiny-q4
go test -tags aria_ffi ./...
```

Go modules are consumed via the Git release tag (`proxy.golang.org`); no separate registry job.
