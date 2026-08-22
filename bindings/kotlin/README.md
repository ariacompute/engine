# Aria Kotlin / Android

Maven coordinates: `com.ariacompute:engine`.

JNI must load `libaria_ffi`. Implement `native*` methods in a small JNI `.so` that forwards to `aria.h`, or use JNA.

```kotlin
AriaEngine("/path/to/bundle").use { eng ->
  println(eng.complete("""[{"role":"user","content":"hi"}]"""))
}
```

Published to Maven Central via `publish-maven.yml` (vanniktech plugin -> Central Portal, automatic release).

## Auto-download by model name

`AriaEngine.open(modelRef, token, site)` accepts a local bundle path **or** an
Aria model name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or already on
disk is loaded directly; otherwise the SDK downloads it from the Dashboard
private source (requires `token`; `site` defaults to `https://ariacompute.com`)
into `~/.ariacompute/models/{model}` and then loads it. A valid cached bundle is
reused without re-downloading.

```kotlin
AriaEngine.open("gemma-4-e2b-it_q4", "DASHBOARD_TOKEN").use { eng ->
  println(eng.complete("""[{"role":"user","content":"hi"}]"""))
}
```
