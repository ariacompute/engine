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
loaded directly; otherwise the SDK downloads it from the Dashboard private
source (requires `token`; `site` defaults to `https://ariacompute.com`) into
`~/.ariacompute/models/{model}` and then loads it. A valid cached bundle is
reused without re-downloading.

```swift
let eng = try await AriaEngine.open("gemma-4-e2b-it_q4", token: "API_TOKEN")
```
