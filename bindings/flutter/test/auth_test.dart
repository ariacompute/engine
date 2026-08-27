import 'package:aria_engine/auth.dart';
import 'package:test/test.dart';

void main() {
  test('defaults', () {
    final cfg = defaultAuthConfig();
    expect(cfg.hybridMode, 'balance');
    expect(cfg.hybridExecution, 'hybrid');
    expect(cfg.hybridSemantic, isTrue);
    expect(cfg.hybridSemanticTimeoutMs, 800);
    expect(cfg.hybridSemanticCacheSize, 512);
    expect(cfg.compute, 'auto');
  });

  test('invalid enum', () {
    expect(
        () => applyAuth(defaultAuthConfig(), hybridMode: 'fast'), throwsArgumentError);
    expect(() => applyAuth(defaultAuthConfig(), hybridExecution: 'local'),
        throwsArgumentError);
    expect(
        () => applyAuth(defaultAuthConfig(), compute: 'gpu'), throwsArgumentError);
  });

  test('fill urls from cn site', () {
    final got = fillAuthUrls(AuthConfig(siteUrl: cnSite));
    expect(got.cloudUrl, cnCloud);
    expect(got.upgradeUrl, cnUpgrade);
  });

  test('all fields roundtrip', () {
    final st = applyAuth(
      defaultAuthConfig(),
      cloudApiKey: 'sk-test',
      cloudUrl: cnCloud,
      siteUrl: cnSite,
      upgradeUrl: cnUpgrade,
      hybridMode: 'cost',
      hybridExecution: 'device',
      hybridSemantic: false,
      hybridSemanticTimeoutMs: 250,
      hybridSemanticCacheSize: 16,
      compute: 'cpu',
      hfToken: 'hf_abc',
      modelscopeApiToken: 'ms_xyz',
    );
    expect(st.cloudApiKey, 'sk-test');
    expect(st.hybridMode, 'cost');
    expect(st.hfToken, 'hf_abc');
    expect(st.siteUrl, cnSite);
  });

  test('partial merge', () {
    final once = applyAuth(defaultAuthConfig(),
        hfToken: 'hf_one', hybridMode: 'intelligence');
    final st = applyAuth(once, compute: 'cuda');
    expect(st.hfToken, 'hf_one');
    expect(st.hybridMode, 'intelligence');
    expect(st.compute, 'cuda');
  });

  test('invalid enum leaves existing', () {
    final once = applyAuth(defaultAuthConfig(), hybridMode: 'cost');
    expect(() => applyAuth(once, hybridMode: 'nope'), throwsArgumentError);
    expect(once.hybridMode, 'cost');
  });

  test('fills urls from site tld', () {
    final st =
        applyAuth(defaultAuthConfig(), siteUrl: 'https://ariacompute.cn');
    expect(st.cloudUrl, cnCloud);
    expect(st.upgradeUrl, cnUpgrade);
  });

  test('detect urls from key mocked', () {
    final prev = probeDashboard;
    probeDashboard = (site, key) => site.contains('ariacompute.cn');
    try {
      final st = applyAuth(defaultAuthConfig(), cloudApiKey: 'sk-region');
      expect(st.siteUrl, cnSite);
      expect(st.cloudUrl, cnCloud);
      expect(st.upgradeUrl, cnUpgrade);
    } finally {
      probeDashboard = prev;
    }
  });
}
