# aria_engine (Flutter)

Flutter FFI binding for Aria Engine. Published to pub.dev under publisher
[ariacompute.com](https://pub.dev/publishers/ariacompute.com/packages).

```yaml
dependencies:
  aria_engine: ^0.1.0
```

```dart
import 'package:aria_engine/aria_engine.dart';

final eng = AriaEngine('/path/to/bundle');
print(eng.complete('[{"role":"user","content":"hi"}]'));
eng.dispose();
```

JNI/FFI must load `libaria_ffi` (`ARIA_FFI_LIB` or `libPath`).

## Auto-download by model name

`AriaEngine.open(modelRef, token: site: libPath:)` accepts a local bundle path
**or** an Aria model name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or
already on disk is loaded directly; otherwise the SDK downloads it from the
regional public hub (same as `aria-engine download`: `.com` → Hugging Face,
`.cn` → ModelScope; `site` defaults to `https://ariacompute.com`) into
`~/.ariacompute/models/{model}` and then loads it. Dashboard is not used. A
Dashboard `sk-` / `bfvk-` token is ignored for hub auth. Token is optional for
public models. Gated files: pass `hfToken` / `modelscopeApiToken` (same as
`aria-engine auth`); if omitted, reads `~/.ariacompute/config.yml`. A valid
cached bundle is reused without re-downloading.

```dart
final eng = await AriaEngine.open("gemma-4-e2b-it_q4");
final gated = await AriaEngine.open("gemma-4-e2b-it_q4", hfToken: "hf_...");
```

Published to pub.dev via `publish-pub.yml`. Secret `PUB_CREDENTIALS` is the
JSON from `dart pub login` **as a publisher member** (browser Google
account, e.g. the ariacompute.com publisher admin). The WSL/local OS user
that owns the file, and the GitHub Actions identity, are not the pub.dev
user. Device-farm: `.github/workflows/bindings-mobile.yml`.
