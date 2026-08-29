import 'package:aria_engine/auth.dart';
import 'package:test/test.dart';

void main() {
  test('defaults', () {
    final cfg = defaultAuthConfig();
    expect(cfg.compute, 'auto');
    expect(cfg.router, '');
  });

  test('invalid enum', () {
    expect(
        () => applyAuth(defaultAuthConfig(), compute: 'gpu'), throwsArgumentError);
  });

  test('fill urls from cn site', () {
    final got = fillAuthUrls(AuthConfig(siteUrl: cnSite));
    expect(got.upgradeUrl, cnUpgrade);
  });

  test('all fields roundtrip', () {
    final st = applyAuth(
      defaultAuthConfig(),
      router: 'http://127.0.0.1:8080',
      siteUrl: cnSite,
      upgradeUrl: cnUpgrade,
      compute: 'cpu',
      hfToken: 'hf_abc',
      modelscopeApiToken: 'ms_xyz',
    );
    expect(st.router, 'http://127.0.0.1:8080');
    expect(st.compute, 'cpu');
    expect(st.hfToken, 'hf_abc');
  });

  test('invalid enum leaves state', () {
    final once = applyAuth(defaultAuthConfig(), compute: 'cpu');
    expect(() => applyAuth(once, compute: 'gpu'), throwsArgumentError);
    expect(once.compute, 'cpu');
  });
}
