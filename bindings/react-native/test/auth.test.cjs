const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {
  applyAuth,
  defaultAuthConfig,
  fillAuthUrls,
  CN_SITE,
  CN_UPGRADE,
} = require('../src/auth.js');

test('auth defaults', () => {
  const cfg = defaultAuthConfig();
  assert.equal(cfg.compute, 'auto');
  assert.equal(cfg.router, '');
});

test('auth invalid enum', () => {
  assert.throws(() => applyAuth(defaultAuthConfig(), { compute: 'gpu' }));
});

test('auth fills urls from cn site', () => {
  const got = fillAuthUrls({ site_url: CN_SITE, upgrade_url: '' });
  assert.equal(got.upgrade_url, CN_UPGRADE);
});

test('auth instance all fields via applyAuth', () => {
  const st = applyAuth(defaultAuthConfig(), {
    router: 'http://127.0.0.1:8080',
    site_url: CN_SITE,
    upgrade_url: CN_UPGRADE,
    compute: 'cpu',
    hf_token: 'hf_abc',
    modelscope_api_token: 'ms_xyz',
  });
  assert.equal(st.router, 'http://127.0.0.1:8080');
  assert.equal(st.hf_token, 'hf_abc');
});

test('auth partial merge', () => {
  const once = applyAuth(defaultAuthConfig(), { hf_token: 'hf_one', router: 'http://127.0.0.1:1' });
  const st = applyAuth(once, { compute: 'cuda' });
  assert.equal(st.hf_token, 'hf_one');
  assert.equal(st.router, 'http://127.0.0.1:1');
  assert.equal(st.compute, 'cuda');
});

test('auth invalid enum leaves state', () => {
  const once = applyAuth(defaultAuthConfig(), { compute: 'cpu' });
  assert.throws(() => applyAuth(once, { compute: 'gpu' }));
  assert.equal(once.compute, 'cpu');
});

test('auth fills urls from site tld', () => {
  const st = applyAuth(defaultAuthConfig(), { site_url: 'https://ariacompute.cn' });
  assert.equal(st.upgrade_url, CN_UPGRADE);
});

test('auth does not write config.yml', () => {
  const prev = process.env.ARIA_COMPUTE_HOME;
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'aria-rn-auth-'));
  process.env.ARIA_COMPUTE_HOME = home;
  try {
    applyAuth(defaultAuthConfig(), {
      router: 'http://127.0.0.1:8080',
      site_url: 'https://ariacompute.com',
      hf_token: 'hf_x',
    });
    assert.equal(fs.existsSync(path.join(home, 'config.yml')), false);
  } finally {
    if (prev === undefined) delete process.env.ARIA_COMPUTE_HOME;
    else process.env.ARIA_COMPUTE_HOME = prev;
  }
});
