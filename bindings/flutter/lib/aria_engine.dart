import 'dart:convert';
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

const _defaultSite = 'https://ariacompute.com';
const _defaultSdk = 'v1.0';
const _hubRequired = ['config.json', 'weight.bin'];
const _hubOptional = [
  'tokenizer.json',
  'tokenizer.model',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
];

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
    var suffix = model.substring(idx + 2);
    if (suffix.endsWith('_channel') || suffix.endsWith('_group')) {
      suffix = suffix.substring(0, suffix.lastIndexOf('_'));
    }
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

String _preferredPublicHub(String site) =>
    site.toLowerCase().contains('ariacompute.cn') ? 'modelscope' : 'huggingface';

String? _hubBearer(String? token) {
  final t = token?.trim() ?? '';
  if (t.isEmpty) return null;
  final low = t.toLowerCase();
  if (low.startsWith('sk-') || low.startsWith('bfvk-')) return null;
  return t;
}

List<String> _hubPathNames(String model) {
  final names = <String>[model];
  var lower = model.toLowerCase();
  var core = model;
  for (final suf in ['_channel', '_group']) {
    if (lower.endsWith(suf)) {
      core = model.substring(0, model.length - suf.length);
      lower = core.toLowerCase();
      break;
    }
  }
  final stems = <String>[core];
  if (lower.endsWith('_q326')) {
    stems.add('${core.substring(0, core.length - 5)}q3.26');
  } else if (lower.endsWith('_q3.26')) {
    stems.add('${core.substring(0, core.length - 6)}q326');
  }
  for (final stem in stems) {
    for (final share in ['', '_channel', '_group']) {
      final cand = '$stem$share';
      if (!names.contains(cand)) names.add(cand);
    }
  }
  return names;
}

List<String> _hubFileUrls(String source, String model, String file) {
  final urls = <String>[];
  for (final name in _hubPathNames(model)) {
    if (source == 'modelscope') {
      for (final repo in ['AriaCompute/$name', 'AriaCompute/model']) {
        urls.add(
            'https://www.modelscope.cn/models/$repo/resolve/master/$_defaultSdk/$name/$file');
        urls.add(
            'https://modelscope.cn/models/$repo/resolve/master/$_defaultSdk/$name/$file');
      }
    } else {
      for (final repo in ['ariacompute/$name', 'ariacompute/model']) {
        urls.add(
            'https://huggingface.co/$repo/resolve/main/$_defaultSdk/$name/$file');
      }
    }
  }
  return urls;
}

class _HubAuthException implements Exception {
  final int code;
  final String source;
  _HubAuthException(this.code, this.source);
  @override
  String toString() {
    final field =
        source == 'modelscope' ? 'modelscope_api_token' : 'hf_token';
    return 'auth failed HTTP $code; set $field via aria-engine auth (do not pass a Dashboard sk-/bfvk- key as the hub token)';
  }
}

Future<void> _fetchUrlToFile(String url, String dest, String? token) async {
  final client = HttpClient();
  try {
    final req = await client.getUrl(Uri.parse(url));
    if (token != null && token.isNotEmpty) {
      req.headers.set(HttpHeaders.authorizationHeader, 'Bearer $token');
    }
    final resp = await req.close();
    if (resp.statusCode == 401 || resp.statusCode == 403) {
      await resp.drain();
      throw HttpException('HTTP ${resp.statusCode}', uri: Uri.parse(url));
    }
    if (resp.statusCode != 200) {
      await resp.drain();
      throw HttpException('HTTP ${resp.statusCode}', uri: Uri.parse(url));
    }
    await File(dest).parent.create(recursive: true);
    await resp.pipe(File(dest).openWrite());
  } finally {
    client.close();
  }
}

Future<bool> _fetchHubFile(String source, String model, String file,
    String dest, String? token, {required bool required}) async {
  Object? last;
  for (final url in _hubFileUrls(source, model, file)) {
    try {
      await _fetchUrlToFile(url, dest, token);
      return true;
    } catch (e) {
      last = e;
      final msg = e.toString();
      if (msg.contains('HTTP 401') || msg.contains('HTTP 403')) {
        final code = msg.contains('401') ? 401 : 403;
        throw _HubAuthException(code, source);
      }
    }
  }
  if (required) {
    throw Exception('$source: missing $file${last != null ? ': $last' : ''}');
  }
  return false;
}

/// Download [model] from the regional public hub into
/// `~/.ariacompute/models/{model}`. Dashboard is not used.
Future<String> downloadModel(String model,
    {String? token, String site = _defaultSite}) async {
  _parseBundleName(model);
  final source = _preferredPublicHub(site);
  final hubToken = _hubBearer(token);
  final cache = Directory(_cacheDir(model));
  if (cache.existsSync() && _isValidBundle(cache)) {
    return cache.path;
  }

  final staging = Directory('${_ariaHome()}/models/.$model.partial');
  try {
    if (staging.existsSync()) staging.deleteSync(recursive: true);
    staging.createSync(recursive: true);
    for (final file in _hubRequired) {
      await _fetchHubFile(
          source, model, file, '${staging.path}/$file', hubToken,
          required: true);
    }
    for (final extra in _hubOptional) {
      try {
        await _fetchHubFile(
            source, model, extra, '${staging.path}/$extra', hubToken,
            required: false);
      } catch (_) {}
    }
    if (!_isValidBundle(staging)) {
      throw Exception(
          '$source fetch completed but bundle invalid (need weight.bin + aria-quant-bundle config.json)');
    }
    if (cache.existsSync()) cache.deleteSync(recursive: true);
    staging.renameSync(cache.path);
    return cache.path;
  } catch (e) {
    if (staging.existsSync()) staging.deleteSync(recursive: true);
    rethrow;
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
  /// disk is a local path; otherwise it is a model name downloaded from the
  /// regional public hub then loaded.
  static Future<AriaEngine> open(String modelRef,
      {String? token, String site = _defaultSite, String? libPath}) async {
    if (_isLocalRef(modelRef)) return AriaEngine(modelRef, libPath: libPath);
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
