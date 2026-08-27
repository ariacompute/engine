/** Instance-level Engine auth (in-memory; does not write config.yml). */

const INTL_CLOUD = 'https://gateway.ariacompute.com';
const INTL_SITE = 'https://ariacompute.com';
const INTL_UPGRADE = 'https://github.com/ariacompute';
const CN_CLOUD = 'https://gateway.ariacompute.cn';
const CN_SITE = 'https://ariacompute.cn';
const CN_UPGRADE = 'https://gitee.com/ariacompute';

function defaultAuthConfig() {
  return {
    cloud_api_key: '',
    cloud_url: '',
    site_url: '',
    upgrade_url: '',
    hybrid_mode: 'balance',
    hybrid_execution: 'hybrid',
    hybrid_semantic: true,
    hybrid_semantic_timeout_ms: 800,
    hybrid_semantic_cache_size: 512,
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
  return region === 'cn' ? [CN_CLOUD, CN_SITE, CN_UPGRADE] : [INTL_CLOUD, INTL_SITE, INTL_UPGRADE];
}

function fillAuthUrls(cfg) {
  const out = { ...cfg };
  const region =
    gatewayRegion(out.site_url) || gatewayRegion(out.cloud_url) || gatewayRegion(out.upgrade_url);
  if (!region) return out;
  const [cloud, site, upgrade] = pairUrls(region);
  if (!out.cloud_url) out.cloud_url = cloud;
  if (!out.site_url) out.site_url = site;
  if (!out.upgrade_url) out.upgrade_url = upgrade;
  return out;
}

function localePrefersCn() {
  const lang = `${process.env.LANG || ''}${process.env.LC_ALL || ''}`.toLowerCase();
  return lang.includes('zh') || lang.includes('.cn') || lang.startsWith('cn');
}

const authHooks = {
  probeDashboard: () => false,
};

function detectGatewayPair(apiKey) {
  const key = (apiKey || '').trim();
  const first = localePrefersCn() ? 'cn' : 'intl';
  const second = first === 'cn' ? 'intl' : 'cn';
  for (const region of [first, second]) {
    const [cloud, site, upgrade] = pairUrls(region);
    if (key && authHooks.probeDashboard(site, key)) return [cloud, site, upgrade];
  }
  return pairUrls(first);
}

function applyAuth(existing, updates) {
  const out = { ...existing };
  for (const [k, v] of Object.entries(updates || {})) {
    if (v === undefined) continue;
    out[k] = v;
  }
  if (!['cost', 'balance', 'intelligence'].includes(out.hybrid_mode)) {
    throw new Error(`invalid hybrid_mode: ${out.hybrid_mode}`);
  }
  if (!['hybrid', 'device', 'cloud'].includes(out.hybrid_execution)) {
    throw new Error(`invalid hybrid_execution: ${out.hybrid_execution}`);
  }
  if (!['auto', 'cpu', 'cuda'].includes(out.compute)) {
    throw new Error(`invalid compute: ${out.compute}`);
  }
  const timeout = Number(out.hybrid_semantic_timeout_ms);
  const cache = Number(out.hybrid_semantic_cache_size);
  if (!Number.isInteger(timeout) || !Number.isInteger(cache) || timeout <= 0 || cache <= 0) {
    throw new Error('hybrid_semantic_timeout_ms / cache_size must be positive integers');
  }
  out.hybrid_semantic = Boolean(out.hybrid_semantic);
  out.hybrid_semantic_timeout_ms = timeout;
  out.hybrid_semantic_cache_size = cache;
  const filled = fillAuthUrls(out);
  if (filled.cloud_api_key && !(filled.cloud_url && filled.site_url && filled.upgrade_url)) {
    const [cloud, site, upgrade] = detectGatewayPair(filled.cloud_api_key);
    if (!filled.cloud_url) filled.cloud_url = cloud;
    if (!filled.site_url) filled.site_url = site;
    if (!filled.upgrade_url) filled.upgrade_url = upgrade;
  }
  return filled;
}

module.exports = {
  INTL_CLOUD,
  INTL_SITE,
  INTL_UPGRADE,
  CN_CLOUD,
  CN_SITE,
  CN_UPGRADE,
  defaultAuthConfig,
  fillAuthUrls,
  applyAuth,
  authHooks,
  detectGatewayPair,
};
