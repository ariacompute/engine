const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {
  applySetup,
  defaultSetupConfig,
  fillSetupUrls,
  CN_SITE,
  CN_UPGRADE,
} = require('../src/setup.js');

test('setup defaults', () => {
  const cfg = defaultSetupConfig();
  assert.equal(cfg.compute, 'auto');
  assert.equal(cfg.router, '');
});

test('setup invalid enum', () => {
  assert.throws(() => applySetup(defaultSetupConfig(), { compute: 'gpu' }));
});

test('setup fills urls from cn site', () => {
  const got = fillSetupUrls({ site_url: CN_SITE, upgrade_url: '' });
  assert.equal(got.upgrade_url, CN_UPGRADE);
});

test('setup instance all fields via applySetup', () => {
  const st = applySetup(defaultSetupConfig(), {
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

test('setup partial merge', () => {
  const once = applySetup(defaultSetupConfig(), { hf_token: 'hf_one', router: 'http://127.0.0.1:1' });
  const st = applySetup(once, { compute: 'cuda' });
  assert.equal(st.hf_token, 'hf_one');
  assert.equal(st.router, 'http://127.0.0.1:1');
  assert.equal(st.compute, 'cuda');
});

test('setup invalid enum leaves state', () => {
  const once = applySetup(defaultSetupConfig(), { compute: 'cpu' });
  assert.throws(() => applySetup(once, { compute: 'gpu' }));
  assert.equal(once.compute, 'cpu');
});

test('setup fills urls from site tld', () => {
  const st = applySetup(defaultSetupConfig(), { site_url: 'https://ariacompute.cn' });
  assert.equal(st.upgrade_url, CN_UPGRADE);
});

test('setup does not write engine.yml', () => {
  const prev = process.env.ARIA_COMPUTE_HOME;
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'aria-rn-setup-'));
  process.env.ARIA_COMPUTE_HOME = home;
  try {
    applySetup(defaultSetupConfig(), {
      router: 'http://127.0.0.1:8080',
      site_url: 'https://ariacompute.com',
      hf_token: 'hf_x',
    });
    assert.equal(fs.existsSync(path.join(home, 'engine.yml')), false);
    assert.equal(fs.existsSync(path.join(home, 'config.yml')), false);
  } finally {
    if (prev === undefined) delete process.env.ARIA_COMPUTE_HOME;
    else process.env.ARIA_COMPUTE_HOME = prev;
  }
});
