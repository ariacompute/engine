import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

typedef _InitC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>);
typedef _InitDart = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>);
typedef _DestroyC = Void Function(Pointer<Void>);
typedef _DestroyDart = void Function(Pointer<Void>);
typedef _SysC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, IntPtr);
typedef _SysDart = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>, int);

class AriaEngine {
  late final DynamicLibrary _lib;
  late final Pointer<Void> _handle;
  late final _DestroyDart _destroy;
  late final _SysDart _systemone;

  AriaEngine(String checkpoint, {String track = 'encoder'}) {
    _lib = _openLib();
    final init = _lib.lookupFunction<_InitC, _InitDart>('aria_model_init');
    _destroy = _lib.lookupFunction<_DestroyC, _DestroyDart>('aria_model_destroy');
    _systemone = _lib.lookupFunction<_SysC, _SysDart>('aria_systemone');
    final cPath = checkpoint.toNativeUtf8();
    final cTrack = track.toNativeUtf8();
    _handle = init(cPath, cTrack);
    malloc.free(cPath);
    malloc.free(cTrack);
    if (_handle == nullptr) {
      throw StateError('aria_model_init failed');
    }
  }

  Map<String, dynamic> systemone(Map<String, dynamic> request) {
    final req = jsonEncode(request).toNativeUtf8();
    final out = calloc<Uint8>(1 << 20);
    final rc = _systemone(_handle, req, out.cast<Utf8>(), 1 << 20);
    malloc.free(req);
    if (rc != 0) {
      calloc.free(out);
      throw StateError('aria_systemone rc=$rc');
    }
    final text = out.cast<Utf8>().toDartString();
    calloc.free(out);
    return jsonDecode(text) as Map<String, dynamic>;
  }

  void destroy() => _destroy(_handle);

  static DynamicLibrary _openLib() {
    final env = Platform.environment['ARIA_FFI_LIB'];
    if (env != null && env.isNotEmpty) {
      return DynamicLibrary.open(env);
    }
    final home = Platform.environment['ARIA_COMPUTE_HOME'] ??
        '${Platform.environment['HOME']}/.ariacompute';
    if (Platform.isMacOS) {
      return DynamicLibrary.open('$home/lib/libaria-engine_ffi.dylib');
    }
    if (Platform.isWindows) {
      return DynamicLibrary.open('$home/lib/aria-engine_ffi.dll');
    }
    return DynamicLibrary.open('$home/lib/libaria-engine_ffi.so');
  }
}
