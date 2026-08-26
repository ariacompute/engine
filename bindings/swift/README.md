# AriaEngine (Swift)

XCFramework / module map over `libaria_ffi`. CocoaPods name: `AriaEngine`.

```swift
import AriaEngine
let eng = try AriaEngine(bundlePath: "/path/to/bundle")
let json = try eng.complete(messagesJson: #"[{"role":"user","content":"hi"}]"#, optionsJson: #"{"max_tokens":16}"#)
```

See `Sources/AriaEngine/AriaEngine.swift`. Publish via `pod trunk push` on GitHub Release (`release.yml`, fail-pass).

## Auto-download by model name

`AriaEngine.open(_:token:site:)` accepts a local bundle path **or** an Aria model
name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or already on disk is
loaded directly; otherwise the SDK downloads it from the regional public hub
(same as `aria-engine download`: `.com` → Hugging Face, `.cn` → ModelScope;
`site` defaults to `https://ariacompute.com`) into `~/.ariacompute/models/{model}`
and then loads it. Dashboard is not used. A Dashboard `sk-` / `bfvk-` token is
ignored for hub auth. Token is optional for public models. Gated files: pass
`hfToken` / `modelscopeApiToken` (same as `aria-engine auth`); if omitted, reads
`~/.ariacompute/config.yml`. A valid cached bundle is reused without re-downloading.

```swift
let eng = try AriaEngine.open("gemma-4-e2b-it_q4")
let gated = try AriaEngine.open("gemma-4-e2b-it_q4", hfToken: "hf_...")
```
