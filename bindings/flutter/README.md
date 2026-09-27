# package:aria_engine

Flutter FFI binding for AFM-D `aria_systemone`.

```dart
final eng = AriaEngine('/path/to/ckpt', track: 'decoder');
final resp = eng.systemone({'state': {}, 'questions': {}});
```
