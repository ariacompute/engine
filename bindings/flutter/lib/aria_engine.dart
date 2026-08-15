import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef AriaInitC = Pointer Function(Pointer<Utf8>);
typedef AriaInitDart = Pointer Function(Pointer<Utf8>);
typedef AriaDestroyC = Void Function(Pointer);
typedef AriaDestroyDart = void Function(Pointer);
typedef AriaCompleteC = Int32 Function(
    Pointer, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, IntPtr);
typedef AriaCompleteDart = int Function(
    Pointer, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, int);

class AriaEngine {
  late final DynamicLibrary _lib;
  late final Pointer _handle;

  AriaEngine(String bundlePath, {String? libPath}) {
    final path = libPath ??
        Platform.environment['ARIA_FFI_LIB'] ??
        (Platform.isMacOS ? 'libaria_ffi.dylib' : 'libaria_ffi.so');
    _lib = DynamicLibrary.open(path);
    final init = _lib.lookupFunction<AriaInitC, AriaInitDart>('aria_model_init');
    final p = bundlePath.toNativeUtf8();
    _handle = init(p);
    malloc.free(p);
    if (_handle.address == 0) {
      throw StateError('aria_model_init failed');
    }
  }

  String complete(String messagesJson,
      {String optionsJson = '{"max_tokens":16}', String toolsJson = '[]'}) {
    final complete =
        _lib.lookupFunction<AriaCompleteC, AriaCompleteDart>('aria_complete');
    final out = malloc<Uint8>(256 * 1024).cast<Utf8>();
    final m = messagesJson.toNativeUtf8();
    final o = optionsJson.toNativeUtf8();
    final t = toolsJson.toNativeUtf8();
    final rc = complete(_handle, m, o, t, out, 256 * 1024);
    malloc.free(m);
    malloc.free(o);
    malloc.free(t);
    if (rc != 0) {
      malloc.free(out);
      throw StateError('aria_complete failed');
    }
    final s = out.toDartString();
    malloc.free(out);
    return s;
  }

  void dispose() {
    final destroy =
        _lib.lookupFunction<AriaDestroyC, AriaDestroyDart>('aria_model_destroy');
    destroy(_handle);
  }
}
