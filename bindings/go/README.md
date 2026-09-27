# Go binding

```go
eng, err := aria_engine.Open("/path/to/ckpt", "encoder")
resp, err := eng.SystemOne(map[string]any{"state": map[string]any{}, "questions": map[string]any{}})
```

Requires `cargo build -p ariacompute-ffi` so `-laria_ffi` resolves.
