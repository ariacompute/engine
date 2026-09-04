# Aria Kotlin / Android

Maven coordinates: `com.ariacompute:engine`.

JNI must load `libaria_ffi`. `AriaEngine.open` / construction installs it into `~/.ariacompute/lib/` when it is not already on `ARIA_FFI_LIB` or loadable via `System.loadLibrary`. Implement `native*` methods in a small JNI `.so` that forwards to `aria.h`, or use JNA.

```kotlin
AriaEngine("/path/to/bundle").use { eng ->
  println(eng.complete("""[{"role":"user","content":"hi"}]"""))
}
```

Published to Maven Central via `publish-maven.yml` (vanniktech plugin -> Central Portal, automatic release).

## Auto-download by model name

`AriaEngine.open(modelRef, token, site)` accepts a local bundle path **or** an
Aria model name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or already on
disk is loaded directly; otherwise the SDK downloads it from the regional public
hub (same as `aria-engine download`: `.com` → Hugging Face, `.cn` → ModelScope;
`site` defaults to `https://ariacompute.com`) into `~/.ariacompute/models/{model}`
and then loads it. Dashboard is not used. A Dashboard `sk-` / `sk-bf-` token is
ignored for hub auth. Token is optional for public models. Gated files: pass
`hfToken` / `modelscopeApiToken` (same as `aria-engine setup`); if omitted, reads
`~/.ariacompute/engine.yml`. Instance `setup` is in-memory only (does not write
that file). A valid cached bundle is reused without re-downloading.

```kotlin
AriaEngine.open("gemma-4-e2b-it_q4").use { eng ->
  println(eng.complete("""[{"role":"user","content":"hi"}]"""))
}
val gated = AriaEngine()
gated.setup(hfToken = "hf_...")
gated.open("gemma-4-e2b-it_q4")
```
