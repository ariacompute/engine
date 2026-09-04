/// Instance-level Engine setup (in-memory; does not write engine.yml).
library;

const intlSite = 'https://ariacompute.com';
const intlUpgrade = 'https://github.com/ariacompute';
const cnSite = 'https://ariacompute.cn';
const cnUpgrade = 'https://gitee.com/ariacompute';

const computes = {'auto', 'cpu', 'cuda'};

class SetupConfig {
  String router;
  String routerApiKey;
  String siteUrl;
  String upgradeUrl;
  String compute;
  String hfToken;
  String modelscopeApiToken;

  SetupConfig({
    this.router = '',
    this.routerApiKey = '',
    this.siteUrl = '',
    this.upgradeUrl = '',
    this.compute = 'auto',
    this.hfToken = '',
    this.modelscopeApiToken = '',
  });

  SetupConfig copy() => SetupConfig(
        router: router,
        routerApiKey: routerApiKey,
        siteUrl: siteUrl,
        upgradeUrl: upgradeUrl,
        compute: compute,
        hfToken: hfToken,
        modelscopeApiToken: modelscopeApiToken,
      );

  Map<String, Object> toMap() => {
        'router': router,
        'router_api_key': routerApiKey,
        'site_url': siteUrl,
        'upgrade_url': upgradeUrl,
        'compute': compute,
        'hf_token': hfToken,
        'modelscope_api_token': modelscopeApiToken,
      };
}

SetupConfig defaultSetupConfig() => SetupConfig();

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

(String, String) _pairUrls(String region) =>
    region == 'cn' ? (cnSite, cnUpgrade) : (intlSite, intlUpgrade);

SetupConfig fillSetupUrls(SetupConfig cfg) {
  final region = _gatewayRegion(cfg.siteUrl) ?? _gatewayRegion(cfg.upgradeUrl);
  if (region == null) return cfg;
  final (site, upgrade) = _pairUrls(region);
  if (cfg.siteUrl.isEmpty) cfg.siteUrl = site;
  if (cfg.upgradeUrl.isEmpty) cfg.upgradeUrl = upgrade;
  return cfg;
}

void _validateRouterApiKey(String key) {
  final t = key.trim();
  if (t.isEmpty) return;
  if (t.startsWith('sk-aria_') || t.startsWith('sk-bf-')) return;
  throw ArgumentError('router_api_key must start with sk-aria_ or sk-bf-');
}

SetupConfig applySetup(SetupConfig existing,
    {String? router,
    String? routerApiKey,
    String? siteUrl,
    String? upgradeUrl,
    String? compute,
    String? hfToken,
    String? modelscopeApiToken}) {
  final out = existing.copy();
  if (router != null) out.router = router;
  if (routerApiKey != null) {
    _validateRouterApiKey(routerApiKey);
    out.routerApiKey = routerApiKey;
  }
  if (siteUrl != null) out.siteUrl = siteUrl;
  if (upgradeUrl != null) out.upgradeUrl = upgradeUrl;
  if (compute != null) out.compute = compute;
  if (hfToken != null) out.hfToken = hfToken;
  if (modelscopeApiToken != null) out.modelscopeApiToken = modelscopeApiToken;
  if (!computes.contains(out.compute)) {
    throw ArgumentError('invalid compute: ${out.compute}');
  }
  _validateRouterApiKey(out.routerApiKey);
  return fillSetupUrls(out);
}
