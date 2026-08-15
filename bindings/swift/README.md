# AriaEngine (Swift)

XCFramework / module map over `libaria_ffi`. CocoaPods name: `AriaEngine`.

```swift
import AriaEngine
let eng = try AriaEngine(bundlePath: "/path/to/bundle")
let json = try eng.complete(messagesJson: #"[{"role":"user","content":"hi"}]"#, optionsJson: #"{"max_tokens":16}"#)
```

See `Sources/AriaEngine/AriaEngine.swift`. Publish via `pod trunk push` on GitHub Release (`release.yml`, fail-pass).
