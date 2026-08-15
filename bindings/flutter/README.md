# aria_engine (Flutter)

```dart
final eng = AriaEngine('/path/to/bundle');
print(eng.complete('[{"role":"user","content":"hi"}]'));
eng.dispose();
```

Publish to pub.dev via `release.yml` (`dart pub publish`, fail-pass). Device-farm: `.github/workflows/bindings-mobile.yml`.
