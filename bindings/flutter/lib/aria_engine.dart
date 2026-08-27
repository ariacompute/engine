import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'auth.dart';

export 'auth.dart';

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

String _unquoteYaml(String v) {
  final t = v.trim();
  if (t.length >= 2 &&
      ((t.startsWith('"') && t.endsWith('"')) ||
          (t.startsWith("'") && t.endsWith("'")))) {
    return t.substring(1, t.length - 1);
  }
  return t;
}

String? _configYmlScalar(String key) {
  try {
    final raw = File('${_ariaHome()}/config.yml').readAsStringSync();
    for (final line in raw.split('\n')) {
      if (line.startsWith(' ') || line.startsWith('\t')) continue;
      final s = line.trim();
      if (s.isEmpty || s.startsWith('#') || !s.contains(':')) continue;
      final idx = s.indexOf(':');
      if (s.substring(0, idx).trim() != key) continue;
      final v = _unquoteYaml(s.substring(idx + 1));
      return v.isEmpty ? null : v;
    }
  } catch (_) {
    return null;
  }
  return null;
}

String? _resolveHubToken(String source,
    {String? token, String? hfToken, String? modelscopeApiToken}) {
  final named = source == 'modelscope' ? modelscopeApiToken : hfToken;
  final field =
      source == 'modelscope' ? 'modelscope_api_token' : 'hf_token';
  for (final cand in [named, token, _configYmlScalar(field)]) {
    final b = _hubBearer(cand);
    if (b != null) return b;
  }
  return null;
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
    {String? token,
    String? hfToken,
    String? modelscopeApiToken,
    String site = _defaultSite}) async {
  _parseBundleName(model);
  final source = _preferredPublicHub(site);
  final hubToken = _resolveHubToken(source,
      token: token, hfToken: hfToken, modelscopeApiToken: modelscopeApiToken);
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

const _sdkUa = 'aria-engine-sdk/0.1.0';

String _ffiLibName() {
  if (Platform.isWindows) return 'aria_ffi.dll';
  if (Platform.isMacOS || Platform.isIOS) return 'libaria_ffi.dylib';
  return 'libaria_ffi.so';
}

String _libDir() => '${_ariaHome()}/lib';

String? _cachedFfiPath() {
  final p = '${_libDir()}/${_ffiLibName()}';
  return File(p).existsSync() ? p : null;
}

String ffiAssetOs({String? system, String? machine}) {
  final sys = (system ?? Platform.operatingSystem).toLowerCase();
  var mach = (machine ?? '').toLowerCase();
  if (mach.isEmpty) {
    try {
      final r = Process.runSync('uname', ['-m']);
      if (r.exitCode == 0) mach = (r.stdout as String).trim().toLowerCase();
    } catch (_) {}
  }
  if ((sys == 'linux' || sys == 'android') &&
      (mach == 'x86_64' || mach == 'amd64')) {
    return 'linux_x86_64';
  }
  if ((sys == 'linux' || sys == 'android') &&
      (mach == 'aarch64' || mach == 'arm64')) {
    return 'linux_arm64';
  }
  if (sys == 'macos' || sys == 'ios' || sys == 'darwin') return 'macos';
  if (sys.startsWith('win') && (mach == 'x86_64' || mach == 'amd64' || mach.isEmpty)) {
    return 'windows_x86_64';
  }
  throw StateError('unsupported platform $sys/$mach for libaria_ffi');
}

String _stripV(String tag) {
  final t = tag.trim();
  if (t.startsWith('v') || t.startsWith('V')) return t.substring(1);
  return t;
}

(int, int, int)? _parseSemver(String tag) {
  final core = _stripV(tag).split('-').first.split('+').first;
  final parts = core.split('.');
  if (parts.isEmpty || int.tryParse(parts[0]) == null) return null;
  return (
    int.parse(parts[0]),
    parts.length > 1 ? int.tryParse(parts[1]) ?? 0 : 0,
    parts.length > 2 ? int.tryParse(parts[2]) ?? 0 : 0,
  );
}

String selectLatestStable(List releases) {
  String? bestTag;
  var bestKey = (-1, -1, -1);
  for (final rel in releases) {
    final map = rel as Map<String, dynamic>;
    if (map['draft'] == true || map['prerelease'] == true) continue;
    final tag = '${map['tag_name'] ?? map['tag'] ?? ''}';
    final parsed = _parseSemver(tag);
    if (parsed == null) continue;
    if (parsed.$1 > bestKey.$1 ||
        (parsed.$1 == bestKey.$1 && parsed.$2 > bestKey.$2) ||
        (parsed.$1 == bestKey.$1 && parsed.$2 == bestKey.$2 && parsed.$3 > bestKey.$3)) {
      bestKey = parsed;
      bestTag = tag;
    }
  }
  if (bestTag == null) {
    throw StateError('no stable release found for libaria_ffi');
  }
  return _stripV(bestTag);
}

String _upgradeOrg(String? site) {
  final cfg = _configYmlScalar('upgrade_url');
  if (cfg != null && cfg.isNotEmpty) return cfg.replaceAll(RegExp(r'/$'), '');
  final hint = (site ?? _configYmlScalar('site_url') ?? _defaultSite).toLowerCase();
  if (hint.contains('ariacompute.cn') || hint.contains('gitee.com')) {
    return 'https://gitee.com/ariacompute';
  }
  return 'https://github.com/ariacompute';
}

String _releasesApiUrl(String org) {
  final owner = org.replaceAll(RegExp(r'/$'), '').split('/').last;
  if (org.toLowerCase().contains('gitee.com')) {
    return 'https://gitee.com/api/v5/repos/$owner/engine/releases?per_page=30';
  }
  return 'https://api.github.com/repos/$owner/engine/releases?per_page=30';
}

Future<List<int>> _httpGetBytes(String url, {String? dest}) async {
  final client = HttpClient();
  try {
    final req = await client.getUrl(Uri.parse(url));
    req.headers.set(HttpHeaders.userAgentHeader, _sdkUa);
    final resp = await req.close();
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      await resp.drain();
      throw HttpException('HTTP ${resp.statusCode}', uri: Uri.parse(url));
    }
    final bytes = await resp.fold<List<int>>(<int>[], (p, e) => p..addAll(e));
    if (dest != null) {
      await File(dest).parent.create(recursive: true);
      await File(dest).writeAsBytes(bytes);
    }
    return bytes;
  } finally {
    client.close();
  }
}

String extractFfiArchive(String archive, String destDir, {String? want}) {
  final name = want ?? _ffiLibName();
  final tarBytes = gzip.decode(File(archive).readAsBytesSync());
  var offset = 0;
  while (offset + 512 <= tarBytes.length) {
    final header = tarBytes.sublist(offset, offset + 512);
    if (header.every((b) => b == 0)) break;
    final entryName =
        String.fromCharCodes(header.sublist(0, 100)).replaceAll('\x00', '');
    final sizeStr =
        String.fromCharCodes(header.sublist(124, 136)).replaceAll('\x00', '').trim();
    final size = int.tryParse(sizeStr, radix: 8) ?? 0;
    final typeFlag = header[156];
    offset += 512;
    final isFile = typeFlag == 0 || typeFlag == 48;
    final base = entryName.split('/').last;
    if (isFile && base == name) {
      Directory(destDir).createSync(recursive: true);
      final dest = '$destDir/$name';
      File(dest).writeAsBytesSync(tarBytes.sublist(offset, offset + size));
      try {
        Process.runSync('chmod', ['755', dest]);
      } catch (_) {}
      return dest;
    }
    offset += ((size + 511) ~/ 512) * 512;
  }
  throw StateError('$name not found in $archive');
}

/// Return a path to libaria_ffi, downloading the latest stable Release if needed.
Future<String> ensureFfiLib({String? site}) async {
  final env = Platform.environment['ARIA_FFI_LIB'];
  if (env != null && File(env).existsSync()) return env;
  final cached = _cachedFfiPath();
  if (cached != null) return cached;

  final org = _upgradeOrg(site);
  final raw = await _httpGetBytes(_releasesApiUrl(org));
  final releases = jsonDecode(utf8.decode(raw));
  if (releases is! List) {
    throw StateError('unexpected releases payload from $org');
  }
  final ver = selectLatestStable(releases);
  final assetName = 'libaria_ffi_${ver}_${ffiAssetOs()}.tar.gz';
  String? url;
  for (final rel in releases) {
    final map = rel as Map<String, dynamic>;
    final tag = '${map['tag_name'] ?? map['tag'] ?? ''}';
    if (_stripV(tag) != ver) continue;
    final assets = map['assets'] as List? ?? [];
    for (final asset in assets) {
      final a = asset as Map<String, dynamic>;
      if (a['name'] == assetName) {
        url = (a['browser_download_url'] ?? a['direct_asset_url']) as String?;
        break;
      }
    }
    if (url != null) break;
  }
  if (url == null) {
    throw StateError('release asset not found: $assetName');
  }
  final staging = Directory('${_ariaHome()}/tmp/ffi-$ver');
  if (staging.existsSync()) staging.deleteSync(recursive: true);
  staging.createSync(recursive: true);
  final archive = '${staging.path}/$assetName';
  try {
    await _httpGetBytes(url, dest: archive);
    return extractFfiArchive(archive, _libDir(), want: _ffiLibName());
  } finally {
    if (staging.existsSync()) staging.deleteSync(recursive: true);
  }
}

class AriaEngine {
  DynamicLibrary? _lib;
  Pointer? _handle;
  AuthConfig _cfg = defaultAuthConfig();
  String? _token;
  String? _libPath;

  /// Empty construct, or a local bundle directory.
  AriaEngine([String? bundlePath, {String? libPath}]) {
    _libPath = libPath;
    if (bundlePath != null) {
      _bindAndInit(bundlePath, libPath: libPath);
    }
  }

  bool _prefersCn() {
    final lang =
        '${Platform.environment['LANG'] ?? ''}${Platform.environment['LC_ALL'] ?? ''}'
            .toLowerCase();
    return lang.contains('zh') ||
        lang.contains('.cn') ||
        lang.startsWith('cn');
  }

  /// Set Config / Run fields on this instance only. Does not write config.yml.
  AriaEngine auth(
      {String? cloudApiKey,
      String? cloudUrl,
      String? siteUrl,
      String? upgradeUrl,
      String? hybridMode,
      String? hybridExecution,
      bool? hybridSemantic,
      int? hybridSemanticTimeoutMs,
      int? hybridSemanticCacheSize,
      String? compute,
      String? hfToken,
      String? modelscopeApiToken}) {
    _cfg = applyAuth(_cfg,
        cloudApiKey: cloudApiKey,
        cloudUrl: cloudUrl,
        siteUrl: siteUrl,
        upgradeUrl: upgradeUrl,
        hybridMode: hybridMode,
        hybridExecution: hybridExecution,
        hybridSemantic: hybridSemantic,
        hybridSemanticTimeoutMs: hybridSemanticTimeoutMs,
        hybridSemanticCacheSize: hybridSemanticCacheSize,
        compute: compute,
        hfToken: hfToken,
        modelscopeApiToken: modelscopeApiToken,
        prefersCn: _prefersCn());
    return this;
  }

  Map<String, Object> authStatus() => _cfg.toMap();

  /// Reset instance defaults. Does not delete ~/.ariacompute/config.yml.
  AriaEngine authClear() {
    _cfg = defaultAuthConfig();
    return this;
  }

  void _bindAndInit(String bundlePath, {String? libPath}) {
    final path = libPath ??
        _libPath ??
        Platform.environment['ARIA_FFI_LIB'] ??
        _cachedFfiPath() ??
        (Platform.isWindows
            ? 'aria_ffi.dll'
            : Platform.isMacOS
                ? 'libaria_ffi.dylib'
                : 'libaria_ffi.so');
    _lib = DynamicLibrary.open(path);
    final init = _lib!.lookupFunction<AriaInitC, AriaInitDart>('aria_model_init');
    final p = bundlePath.toNativeUtf8();
    _handle = init(p);
    malloc.free(p);
    if (_handle!.address == 0) {
      throw StateError('aria_model_init failed');
    }
  }

  /// Download (if needed) and load a model using instance auth.
  Future<AriaEngine> openUsingAuth(String modelRef, {String? libPath}) async {
    final site = _cfg.siteUrl.isEmpty ? _defaultSite : _cfg.siteUrl;
    final resolvedLib = libPath ?? _libPath ?? await ensureFfiLib(site: site);
    final bundle = _isLocalRef(modelRef)
        ? modelRef
        : await downloadModel(modelRef,
            token: _token,
            hfToken: _cfg.hfToken.isEmpty ? null : _cfg.hfToken,
            modelscopeApiToken:
                _cfg.modelscopeApiToken.isEmpty ? null : _cfg.modelscopeApiToken,
            site: site);
    if (_handle != null) dispose();
    _bindAndInit(bundle, libPath: resolvedLib);
    return this;
  }

  /// Open a model by reference. A value containing a separator or already on
  /// disk is a local path; otherwise it is a model name downloaded from the
  /// regional public hub then loaded.
  static Future<AriaEngine> open(String modelRef,
      {String? token,
      String? hfToken,
      String? modelscopeApiToken,
      String site = _defaultSite,
      String? libPath}) async {
    final eng = AriaEngine(libPath: libPath);
    eng._token = token;
    if (site != _defaultSite || hfToken != null || modelscopeApiToken != null) {
      eng.auth(
          siteUrl: site == _defaultSite ? null : site,
          hfToken: hfToken,
          modelscopeApiToken: modelscopeApiToken);
      if (site != _defaultSite && eng._cfg.siteUrl.isEmpty) {
        eng.auth(siteUrl: site);
      }
    }
    await eng.openUsingAuth(modelRef, libPath: libPath);
    return eng;
  }

  String complete(String messagesJson,
      {String optionsJson = '{"max_tokens":16}', String toolsJson = '[]'}) {
    final lib = _lib;
    final handle = _handle;
    if (lib == null || handle == null) {
      throw StateError('engine not opened');
    }
    final complete =
        lib.lookupFunction<AriaCompleteC, AriaCompleteDart>('aria_complete');
    final out = malloc<Uint8>(256 * 1024).cast<Utf8>();
    final m = messagesJson.toNativeUtf8();
    final o = optionsJson.toNativeUtf8();
    final t = toolsJson.toNativeUtf8();
    final rc = complete(handle, m, o, t, out, 256 * 1024);
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
    final lib = _lib;
    final handle = _handle;
    if (lib == null || handle == null) return;
    final destroy =
        lib.lookupFunction<AriaDestroyC, AriaDestroyDart>('aria_model_destroy');
    destroy(handle);
    _handle = null;
  }
}

/// Instance `open` lives on an extension so it does not clash with [AriaEngine.open].
extension AriaEngineAuthOpen on AriaEngine {
  Future<AriaEngine> open(String modelRef, {String? libPath}) =>
      openUsingAuth(modelRef, libPath: libPath);
}
