# Aria Kotlin / Android

Maven coordinates: `com.ariacompute:engine`.

JNI must load `libaria_ffi`. Implement `native*` methods in a small JNI `.so` that forwards to `aria.h`, or use JNA.

```kotlin
AriaEngine("/path/to/bundle").use { eng ->
  println(eng.complete("""[{"role":"user","content":"hi"}]"""))
}
```

Published via `release.yml` (`./gradlew publish`, fail-pass).
