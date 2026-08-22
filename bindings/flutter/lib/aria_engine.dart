import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:http/http.dart' as http;
import 'package:archive/archive.dart';

typedef AriaInitC = Pointer Function(Pointer<Utf8>);
typedef AriaInitDart = Pointer Function(Pointer<Utf8>);
typedef AriaDestroyC = Void Function(Pointer);
typedef AriaDestroyDart = void Function(Pointer);
typedef AriaCompleteC = Int32 Function(
    Pointer, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, IntPtr);
typedef AriaCompleteDart = int Function(
    Pointer, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, int);

const _defaultSite = 'https://ariacompute.com';

String _ariaHome() {
  final override = Platform.environment['ARIA_COMPUTE_HOME'];
  if (override != null && override.isNotEmpty) return override;
  final home = Platform.isWindows
      ? Platform.environment['USERPROFILE']
      : Platform.environment['HOME'];
  return '${home ?? '.'}/.ariacompute';
}

String _cacheDir(String model) => '${_ariaHome()}/models/$model';

(String slug, String quant) _parseBundleName(String model) {
  if (model.isEmpty || model.contains('/') || model.contains('\\')) {
    throw ArgumentError('invalid model name: $model');
  }
  final idx = model.lastIndexOf('_q');
  if (idx != -1) {
    final slug = model.substring(0, idx);
    final suffix = model.substring(idx + 2);
    final quant = switch (suffix) {
      '4' => 'int4',
      '8' => 'int8',
      '326' || '3.26' => 'int326',
      _ => throw ArgumentError('unknown quant suffix _q$suffix'),
    };
    if (slug.isEmpty) throw ArgumentError('invalid model name: $model');
    return (slug, quant);
  }
  return (model, 'int4');
}

bool _isValidBundle(Directory dir) {
  final weight = File('${dir.path}/weight.bin');
  final config = File('${dir.path}/config.json');
  if (!weight.existsSync() || !config.existsSync()) return false;
  try {
    final meta = jsonDecode(config.readAsStringSync()) as Map<String, dynamic>;
    return meta['format'] == 'aria-quant-bundle';
  } catch (_) {
    return false;
  }
}

bool _isLocalRef(String ref) =>
    ref.contains('/') || ref.contains('\\') || Directory(ref).existsSync();

Future<String> downloadModel(String model,
    {required String token, String site = _defaultSite}) async {
  if (token.isEmpty) {
    throw ArgumentError('api token is required to download a model');
  }
  final (slug, quant) = _parseBundleName(model);
  final cache = Directory(_cacheDir(model));
  if (cache.existsSync() && _isValidBundle(cache)) {
    return cache.path;
  }

  final metaUrl = Uri.parse(
      '${site.replaceAll(RegExp(r'/$'), '')}/api/dashboard/models/'
      '${Uri.encodeComponent(slug)}/download'
      '?quant=${Uri.encodeComponent(quant)}&sdk=v1.0&format=json');
  final metaResp = await http.get(metaUrl, headers: {'Authorization': 'Bearer $token'});
  if (metaResp.statusCode != 200) {
    throw Exception('dashboard request failed: ${metaResp.statusCode}');
  }
  final url = (jsonDecode(metaResp.body) as Map<String, dynamic>)['url'] as String?;
  if (url == null || url.isEmpty) {
    throw Exception('dashboard meta returned empty url');
  }

  final zipResp = await http.get(Uri.parse(url),
      headers: {'Authorization': 'Bearer $token'});
  if (zipResp.statusCode != 200) {
    throw Exception('download stream failed: ${zipResp.statusCode}');
  }
  final data = zipResp.bodyBytes;

  final staging = Directory('${_cacheDir('.$model.partial')}');
  if (staging.existsSync()) staging.deleteSync(recursive: true);
  staging.createSync(recursive: true);
  _extractZip(data, staging);
  if (!_isValidBundle(staging)) {
    staging.deleteSync(recursive: true);
    throw Exception('downloaded archive did not contain a valid aria-quant-bundle');
  }
  if (cache.existsSync()) cache.deleteSync(recursive: true);
  staging.renameSync(cache.path);
  return cache.path;
}

void _extractZip(Uint8List data, Directory dest) {
  final archive = ZipDecoder().decodeBytes(data);
  for (final file in archive) {
    final out = File('${dest.path}/${file.name}');
    if (file.isFile) {
      out.parent.createSync(recursive: true);
      out.writeAsBytesSync(file.content as List<int>);
    } else {
      out.createSync(recursive: true);
    }
  }
  // flatten a single top-level subdir
  final entries = dest.listSync().where((e) => !e.path.split('/').last.startsWith('.')).toList();
  if (entries.length == 1 && entries[0] is Directory) {
    final inner = entries[0] as Directory;
    if (File('${inner.path}/config.json').existsSync()) {
      for (final f in inner.listSync()) {
        f.renameSync('${dest.path}/${f.path.split('/').last}');
      }
      inner.deleteSync(recursive: true);
    }
  }
}

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

  /// Open a model by reference. A value containing a separator or already on
  /// disk is a local path; otherwise it is a model name that is downloaded
  /// (requires [token]) then loaded.
  static Future<AriaEngine> open(String modelRef,
      {String? token, String site = _defaultSite, String? libPath}) async {
    if (_isLocalRef(modelRef)) return AriaEngine(modelRef, libPath: libPath);
    if (token == null || token.isEmpty) {
      throw ArgumentError("model name '$modelRef' requires an api token to download");
    }
    final bundle = await downloadModel(modelRef, token: token, site: site);
    return AriaEngine(bundle, libPath: libPath);
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
