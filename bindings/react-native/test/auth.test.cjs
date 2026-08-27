const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {
  applyAuth,
  defaultAuthConfig,
  fillAuthUrls,
  authHooks,
  CN_CLOUD,
  CN_SITE,
  CN_UPGRADE,
} = require('../src/auth.js');

test('auth defaults', () => {
  const cfg = defaultAuthConfig();
  assert.equal(cfg.hybrid_mode, 'balance');
  assert.equal(cfg.hybrid_execution, 'hybrid');
  assert.equal(cfg.hybrid_semantic, true);
  assert.equal(cfg.hybrid_semantic_timeout_ms, 800);
  assert.equal(cfg.hybrid_semantic_cache_size, 512);
  assert.equal(cfg.compute, 'auto');
});

test('auth invalid enum', () => {
  assert.throws(() => applyAuth(defaultAuthConfig(), { hybrid_mode: 'fast' }));
  assert.throws(() => applyAuth(defaultAuthConfig(), { hybrid_execution: 'local' }));
  assert.throws(() => applyAuth(defaultAuthConfig(), { compute: 'gpu' }));
});

test('auth fills urls from cn site', () => {
  const got = fillAuthUrls({ site_url: CN_SITE, cloud_url: '', upgrade_url: '' });
  assert.equal(got.cloud_url, CN_CLOUD);
  assert.equal(got.upgrade_url, CN_UPGRADE);
});

test('auth instance all fields via applyAuth', () => {
  const st = applyAuth(defaultAuthConfig(), {
    cloud_api_key: 'sk-test',
    cloud_url: CN_CLOUD,
    site_url: CN_SITE,
    upgrade_url: CN_UPGRADE,
    hybrid_mode: 'cost',
    hybrid_execution: 'device',
    hybrid_semantic: false,
    hybrid_semantic_timeout_ms: 250,
    hybrid_semantic_cache_size: 16,
    compute: 'cpu',
    hf_token: 'hf_abc',
    modelscope_api_token: 'ms_xyz',
  });
  assert.equal(st.cloud_api_key, 'sk-test');
  assert.equal(st.hybrid_mode, 'cost');
  assert.equal(st.hf_token, 'hf_abc');
});

test('auth partial merge', () => {
  const once = applyAuth(defaultAuthConfig(), { hf_token: 'hf_one', hybrid_mode: 'intelligence' });
  const st = applyAuth(once, { compute: 'cuda' });
  assert.equal(st.hf_token, 'hf_one');
  assert.equal(st.hybrid_mode, 'intelligence');
  assert.equal(st.compute, 'cuda');
});

test('auth invalid enum leaves state', () => {
  const once = applyAuth(defaultAuthConfig(), { hybrid_mode: 'cost' });
  assert.throws(() => applyAuth(once, { hybrid_mode: 'nope' }));
  assert.equal(once.hybrid_mode, 'cost');
});

test('auth fills urls from site tld', () => {
  const st = applyAuth(defaultAuthConfig(), { site_url: 'https://ariacompute.cn' });
  assert.equal(st.cloud_url, CN_CLOUD);
  assert.equal(st.upgrade_url, CN_UPGRADE);
});

test('auth does not write config.yml', () => {
  const prev = process.env.ARIA_COMPUTE_HOME;
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'aria-rn-auth-'));
  process.env.ARIA_COMPUTE_HOME = home;
  try {
    applyAuth(defaultAuthConfig(), {
      cloud_api_key: 'sk-test',
      site_url: 'https://ariacompute.com',
      hf_token: 'hf_x',
    });
    assert.equal(fs.existsSync(path.join(home, 'config.yml')), false);
  } finally {
    if (prev === undefined) delete process.env.ARIA_COMPUTE_HOME;
    else process.env.ARIA_COMPUTE_HOME = prev;
  }
});

test('auth detect urls from key mocked', () => {
  const prev = authHooks.probeDashboard;
  authHooks.probeDashboard = (site) => String(site).includes('ariacompute.cn');
  try {
    const st = applyAuth(defaultAuthConfig(), { cloud_api_key: 'sk-region' });
    assert.equal(st.site_url, CN_SITE);
    assert.equal(st.cloud_url, CN_CLOUD);
    assert.equal(st.upgrade_url, CN_UPGRADE);
  } finally {
    authHooks.probeDashboard = prev;
  }
});
