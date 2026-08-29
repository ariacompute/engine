/** Instance-level Engine setup (in-memory; does not write engine.yml). */

const INTL_SITE = 'https://ariacompute.com';
const INTL_UPGRADE = 'https://github.com/ariacompute';
const CN_SITE = 'https://ariacompute.cn';
const CN_UPGRADE = 'https://gitee.com/ariacompute';

function defaultSetupConfig() {
  return {
    router: '',
    site_url: '',
    upgrade_url: '',
    compute: 'auto',
    hf_token: '',
    modelscope_api_token: '',
  };
}

function gatewayRegion(url) {
  const lower = (url || '').toLowerCase();
  if (lower.includes('ariacompute.cn') || lower.includes('gitee.com/ariacompute')) return 'cn';
  if (lower.includes('ariacompute.com') || lower.includes('github.com/ariacompute')) return 'intl';
  return undefined;
}

function pairUrls(region) {
  return region === 'cn' ? [CN_SITE, CN_UPGRADE] : [INTL_SITE, INTL_UPGRADE];
}

function fillSetupUrls(cfg) {
  const out = { ...cfg };
  const region = gatewayRegion(out.site_url) || gatewayRegion(out.upgrade_url);
  if (!region) return out;
  const [site, upgrade] = pairUrls(region);
  if (!out.site_url) out.site_url = site;
  if (!out.upgrade_url) out.upgrade_url = upgrade;
  return out;
}

function applySetup(existing, updates) {
  const out = { ...existing };
  for (const [k, v] of Object.entries(updates || {})) {
    if (v === undefined) continue;
    out[k] = v;
  }
  if (!['auto', 'cpu', 'cuda'].includes(out.compute)) {
    throw new Error(`invalid compute: ${out.compute}`);
  }
  return fillSetupUrls(out);
}

module.exports = {
  INTL_SITE,
  INTL_UPGRADE,
  CN_SITE,
  CN_UPGRADE,
  defaultSetupConfig,
  fillSetupUrls,
  applySetup,
};
