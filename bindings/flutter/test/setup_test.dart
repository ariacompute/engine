import 'package:ariaengine/setup.dart';
import 'package:test/test.dart';

void main() {
  test('defaults', () {
    final cfg = defaultSetupConfig();
    expect(cfg.compute, 'auto');
    expect(cfg.router, '');
  });

  test('invalid enum', () {
    expect(
        () => applySetup(defaultSetupConfig(), compute: 'gpu'), throwsArgumentError);
  });

  test('fill urls from cn site', () {
    final got = fillSetupUrls(SetupConfig(siteUrl: cnSite));
    expect(got.upgradeUrl, cnUpgrade);
  });

  test('all fields roundtrip', () {
    final st = applySetup(
      defaultSetupConfig(),
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
    final once = applySetup(defaultSetupConfig(), compute: 'cpu');
    expect(() => applySetup(once, compute: 'gpu'), throwsArgumentError);
    expect(once.compute, 'cpu');
  });
}
