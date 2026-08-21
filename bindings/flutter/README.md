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

Published to pub.dev via `publish-pub.yml` (`flutter pub publish`, publisher
ariacompute.com). Device-farm: `.github/workflows/bindings-mobile.yml`.
