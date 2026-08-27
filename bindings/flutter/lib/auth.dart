/// Instance-level Engine auth (in-memory; does not write config.yml).
library;

const intlCloud = 'https://gateway.ariacompute.com';
const intlSite = 'https://ariacompute.com';
const intlUpgrade = 'https://github.com/ariacompute';
const cnCloud = 'https://gateway.ariacompute.cn';
const cnSite = 'https://ariacompute.cn';
const cnUpgrade = 'https://gitee.com/ariacompute';

const hybridModes = {'cost', 'balance', 'intelligence'};
const hybridExecutions = {'hybrid', 'device', 'cloud'};
const computes = {'auto', 'cpu', 'cuda'};

class AuthConfig {
  String cloudApiKey;
  String cloudUrl;
  String siteUrl;
  String upgradeUrl;
  String hybridMode;
  String hybridExecution;
  bool hybridSemantic;
  int hybridSemanticTimeoutMs;
  int hybridSemanticCacheSize;
  String compute;
  String hfToken;
  String modelscopeApiToken;

  AuthConfig({
    this.cloudApiKey = '',
    this.cloudUrl = '',
    this.siteUrl = '',
    this.upgradeUrl = '',
    this.hybridMode = 'balance',
    this.hybridExecution = 'hybrid',
    this.hybridSemantic = true,
    this.hybridSemanticTimeoutMs = 800,
    this.hybridSemanticCacheSize = 512,
    this.compute = 'auto',
    this.hfToken = '',
    this.modelscopeApiToken = '',
  });

  AuthConfig copy() => AuthConfig(
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
      );

  Map<String, Object> toMap() => {
        'cloud_api_key': cloudApiKey,
        'cloud_url': cloudUrl,
        'site_url': siteUrl,
        'upgrade_url': upgradeUrl,
        'hybrid_mode': hybridMode,
        'hybrid_execution': hybridExecution,
        'hybrid_semantic': hybridSemantic,
        'hybrid_semantic_timeout_ms': hybridSemanticTimeoutMs,
        'hybrid_semantic_cache_size': hybridSemanticCacheSize,
        'compute': compute,
        'hf_token': hfToken,
        'modelscope_api_token': modelscopeApiToken,
      };
}

AuthConfig defaultAuthConfig() => AuthConfig();

String? _gatewayRegion(String url) {
  final lower = url.toLowerCase();
  if (lower.contains('ariacompute.cn') ||
      lower.contains('gitee.com/ariacompute')) {
    return 'cn';
  }
  if (lower.contains('ariacompute.com') ||
      lower.contains('github.com/ariacompute')) {
    return 'intl';
  }
  return null;
}

(String, String, String) _pairUrls(String region) => region == 'cn'
    ? (cnCloud, cnSite, cnUpgrade)
    : (intlCloud, intlSite, intlUpgrade);

AuthConfig fillAuthUrls(AuthConfig cfg) {
  final out = cfg.copy();
  final region = _gatewayRegion(out.siteUrl) ??
      _gatewayRegion(out.cloudUrl) ??
      _gatewayRegion(out.upgradeUrl);
  if (region == null) return out;
  final (cloud, site, upgrade) = _pairUrls(region);
  if (out.cloudUrl.isEmpty) out.cloudUrl = cloud;
  if (out.siteUrl.isEmpty) out.siteUrl = site;
  if (out.upgradeUrl.isEmpty) out.upgradeUrl = upgrade;
  return out;
}

bool _localePrefersCn() {
  final lang =
      '${String.fromEnvironment('LANG', defaultValue: '')}${String.fromEnvironment('LC_ALL', defaultValue: '')}'
          .toLowerCase();
  // fromEnvironment is compile-time; also read Platform in callers that have dart:io.
  return lang.contains('zh') || lang.contains('.cn') || lang.startsWith('cn');
}

typedef ProbeDashboard = bool Function(String siteUrl, String apiKey);

/// Replace in tests to avoid a real Dashboard probe.
ProbeDashboard probeDashboard = (_, __) => false;

(String, String, String) detectGatewayPair(String apiKey,
    {bool prefersCn = false}) {
  final key = apiKey.trim();
  final first = prefersCn ? 'cn' : 'intl';
  final second = first == 'cn' ? 'intl' : 'cn';
  for (final region in [first, second]) {
    final (cloud, site, upgrade) = _pairUrls(region);
    if (key.isNotEmpty && probeDashboard(site, key)) {
      return (cloud, site, upgrade);
    }
  }
  return _pairUrls(first);
}

AuthConfig applyAuth(AuthConfig existing,
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
    String? modelscopeApiToken,
    bool prefersCn = false}) {
  final out = existing.copy();
  if (cloudApiKey != null) out.cloudApiKey = cloudApiKey;
  if (cloudUrl != null) out.cloudUrl = cloudUrl;
  if (siteUrl != null) out.siteUrl = siteUrl;
  if (upgradeUrl != null) out.upgradeUrl = upgradeUrl;
  if (hybridMode != null) out.hybridMode = hybridMode;
  if (hybridExecution != null) out.hybridExecution = hybridExecution;
  if (hybridSemantic != null) out.hybridSemantic = hybridSemantic;
  if (hybridSemanticTimeoutMs != null) {
    out.hybridSemanticTimeoutMs = hybridSemanticTimeoutMs;
  }
  if (hybridSemanticCacheSize != null) {
    out.hybridSemanticCacheSize = hybridSemanticCacheSize;
  }
  if (compute != null) out.compute = compute;
  if (hfToken != null) out.hfToken = hfToken;
  if (modelscopeApiToken != null) out.modelscopeApiToken = modelscopeApiToken;
  if (!hybridModes.contains(out.hybridMode)) {
    throw ArgumentError('invalid hybrid_mode: ${out.hybridMode}');
  }
  if (!hybridExecutions.contains(out.hybridExecution)) {
    throw ArgumentError('invalid hybrid_execution: ${out.hybridExecution}');
  }
  if (!computes.contains(out.compute)) {
    throw ArgumentError('invalid compute: ${out.compute}');
  }
  if (out.hybridSemanticTimeoutMs <= 0 || out.hybridSemanticCacheSize <= 0) {
    throw ArgumentError(
        'hybrid_semantic_timeout_ms / cache_size must be positive integers');
  }
  var filled = fillAuthUrls(out);
  if (filled.cloudApiKey.isNotEmpty &&
      (filled.cloudUrl.isEmpty ||
          filled.siteUrl.isEmpty ||
          filled.upgradeUrl.isEmpty)) {
    final (cloud, site, upgrade) =
        detectGatewayPair(filled.cloudApiKey, prefersCn: prefersCn);
    if (filled.cloudUrl.isEmpty) filled.cloudUrl = cloud;
    if (filled.siteUrl.isEmpty) filled.siteUrl = site;
    if (filled.upgradeUrl.isEmpty) filled.upgradeUrl = upgrade;
  }
  return filled;
}

// Silence unused helper in this library; locale is passed by AriaEngine.
bool localePrefersCnFromLang(String lang) {
  final lower = lang.toLowerCase();
  return lower.contains('zh') ||
      lower.contains('.cn') ||
      lower.startsWith('cn') ||
      _localePrefersCn();
}
