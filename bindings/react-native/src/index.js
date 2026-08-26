/**
 * React Native surface — native module bridges to libaria_ffi (Turbo Module / JSI).
 * JS API mirrors @ariacompute/engine-ts for docs samples.
 *
 * Auto-download: `AriaEngine.open(modelRef, { token, site })` resolves a model
 * name via the regional public hub (Hugging Face / ModelScope — same as
 * `aria-engine download`; Dashboard is not used), writes files to
 * `~/.ariacompute/models/{model}`, then loads via native init.
 */
import { NativeModules } from 'react-native';

const AriaEngineModule = NativeModules.AriaEngine;

const DEFAULT_SITE = 'https://ariacompute.com';
const DEFAULT_SDK = 'v1.0';
const HUB_REQUIRED = ['config.json', 'weight.bin'];
const HUB_OPTIONAL = [
  'tokenizer.json',
  'tokenizer.model',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
];

function parseBundleName(model) {
  if (!model || model.includes('/') || model.includes('\\')) {
    throw new Error(`invalid model name: ${model}`);
  }
  const idx = model.lastIndexOf('_q');
  if (idx !== -1) {
    const slug = model.slice(0, idx);
    let suffix = model.slice(idx + 2);
    if (suffix.endsWith('_channel') || suffix.endsWith('_group')) {
      suffix = suffix.slice(0, suffix.lastIndexOf('_'));
    }
    const quant =
      suffix === '4'
        ? 'int4'
        : suffix === '8'
        ? 'int8'
        : suffix === '326' || suffix === '3.26'
        ? 'int326'
        : (() => {
            throw new Error(`unknown quant suffix _q${suffix}`);
          })();
    if (!slug) throw new Error(`invalid model name: ${model}`);
    return { slug, quant };
  }
  return { slug: model, quant: 'int4' };
}

function ariaHome() {
  if (process.env.ARIA_COMPUTE_HOME) return process.env.ARIA_COMPUTE_HOME;
  return `${require('os').homedir()}/.ariacompute`;
}

function cacheDir(model) {
  return `${ariaHome()}/models/${model}`;
}

function isLocalRef(ref) {
  return ref.includes('/') || ref.includes('\\') || require('fs').existsSync(ref);
}

function preferredPublicHub(site) {
  if (site && String(site).toLowerCase().includes('ariacompute.cn')) return 'modelscope';
  return 'huggingface';
}

function hubBearer(token) {
  const t = (token || '').trim();
  if (!t) return undefined;
  const low = t.toLowerCase();
  if (low.startsWith('sk-') || low.startsWith('bfvk-')) return undefined;
  return t;
}

function unquoteYaml(v) {
  const t = (v || '').trim();
  if (t.length >= 2 && ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'")))) {
    return t.slice(1, -1);
  }
  return t;
}

function configYmlScalar(key) {
  const fs = require('fs');
  const path = require('path');
  try {
    const raw = fs.readFileSync(path.join(ariaHome(), 'config.yml'), 'utf8');
    for (const line of raw.split('\n')) {
      if (line.startsWith(' ') || line.startsWith('\t')) continue;
      const s = line.trim();
      if (!s || s.startsWith('#') || !s.includes(':')) continue;
      const idx = s.indexOf(':');
      if (s.slice(0, idx).trim() !== key) continue;
      const v = unquoteYaml(s.slice(idx + 1));
      return v || undefined;
    }
  } catch {
    return undefined;
  }
  return undefined;
}

function resolveHubToken(source, opts = {}) {
  const named = source === 'modelscope' ? opts.modelscopeApiToken : opts.hfToken;
  const field = source === 'modelscope' ? 'modelscope_api_token' : 'hf_token';
  for (const cand of [named, opts.token, configYmlScalar(field)]) {
    const b = hubBearer(cand);
    if (b) return b;
  }
  return undefined;
}

function hubPathNames(model) {
  const names = [model];
  let lower = model.toLowerCase();
  let core = model;
  for (const suf of ['_channel', '_group']) {
    if (lower.endsWith(suf)) {
      core = model.slice(0, -suf.length);
      lower = core.toLowerCase();
      break;
    }
  }
  const stems = [core];
  if (lower.endsWith('_q326')) stems.push(`${core.slice(0, -5)}q3.26`);
  else if (lower.endsWith('_q3.26')) stems.push(`${core.slice(0, -6)}q326`);
  for (const stem of stems) {
    for (const share of ['', '_channel', '_group']) {
      const cand = `${stem}${share}`;
      if (!names.includes(cand)) names.push(cand);
    }
  }
  return names;
}

function hubFileUrls(source, model, file) {
  const urls = [];
  for (const name of hubPathNames(model)) {
    if (source === 'modelscope') {
      for (const repo of [`AriaCompute/${name}`, 'AriaCompute/model']) {
        urls.push(`https://www.modelscope.cn/models/${repo}/resolve/master/${DEFAULT_SDK}/${name}/${file}`);
        urls.push(`https://modelscope.cn/models/${repo}/resolve/master/${DEFAULT_SDK}/${name}/${file}`);
      }
    } else {
      for (const repo of [`ariacompute/${name}`, 'ariacompute/model']) {
        urls.push(`https://huggingface.co/${repo}/resolve/main/${DEFAULT_SDK}/${name}/${file}`);
      }
    }
  }
  return urls;
}

function isValidBundleSync(dir) {
  const fs = require('fs');
  const path = require('path');
  try {
    if (!fs.statSync(path.join(dir, 'weight.bin')).isFile()) return false;
    const meta = JSON.parse(fs.readFileSync(path.join(dir, 'config.json'), 'utf8'));
    return meta.format === 'aria-quant-bundle';
  } catch {
    return false;
  }
}

async function isValidBundle(dir) {
  if (AriaEngineModule?.isValidBundle) {
    try {
      return await AriaEngineModule.isValidBundle(dir);
    } catch {
      /* fall through */
    }
  }
  return isValidBundleSync(dir);
}

async function fetchUrlToFile(url, dest, token) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  const resp = await fetch(url, { headers });
  if (resp.status === 401 || resp.status === 403) {
    const err = new Error(`HTTP ${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  if (!resp.ok) {
    const err = new Error(`HTTP ${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  const fs = require('fs');
  const path = require('path');
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const buf = Buffer.from(await resp.arrayBuffer());
  fs.writeFileSync(dest, buf);
}

async function fetchHubFile(source, model, file, dest, token, required) {
  let last;
  for (const url of hubFileUrls(source, model, file)) {
    try {
      await fetchUrlToFile(url, dest, token);
      return true;
    } catch (e) {
      last = e;
      if (e && (e.status === 401 || e.status === 403)) {
        const field = source === 'modelscope' ? 'modelscope_api_token' : 'hf_token';
        throw new Error(
          `auth failed HTTP ${e.status}; set ${field} via aria-engine auth (do not pass a Dashboard sk-/bfvk- key as the hub token)`,
        );
      }
    }
  }
  if (required) throw new Error(`${source}: missing ${file}${last ? `: ${last}` : ''}`);
  return false;
}

async function downloadModel(model, tokenOrOpts, site = DEFAULT_SITE) {
  const opts =
    tokenOrOpts && typeof tokenOrOpts === 'object'
      ? tokenOrOpts
      : { token: tokenOrOpts, site };
  parseBundleName(model);
  const fs = require('fs');
  const path = require('path');
  const resolvedSite = opts.site ?? site ?? DEFAULT_SITE;
  const source = preferredPublicHub(resolvedSite);
  const hubToken = resolveHubToken(source, opts);
  const cache = cacheDir(model);
  if (fs.existsSync(cache) && (await isValidBundle(cache))) {
    return cache;
  }
  const staging = path.join(ariaHome(), 'models', `.${model}.partial`);
  try {
    if (fs.existsSync(staging)) fs.rmSync(staging, { recursive: true, force: true });
    fs.mkdirSync(staging, { recursive: true });
    for (const file of HUB_REQUIRED) {
      await fetchHubFile(source, model, file, path.join(staging, file), hubToken, true);
    }
    for (const extra of HUB_OPTIONAL) {
      try {
        await fetchHubFile(source, model, extra, path.join(staging, extra), hubToken, false);
      } catch {
        /* optional tokenizer sidecar */
      }
    }
    if (!(await isValidBundle(staging))) {
      throw new Error(
        `${source} fetch completed but bundle invalid (need weight.bin + aria-quant-bundle config.json)`,
      );
    }
    if (fs.existsSync(cache)) fs.rmSync(cache, { recursive: true, force: true });
    fs.renameSync(staging, cache);
    return cache;
  } catch (e) {
    if (fs.existsSync(staging)) fs.rmSync(staging, { recursive: true, force: true });
    throw e;
  }
}

const SDK_UA = 'aria-engine-sdk/0.1.0';
const zlib = require('zlib');

function ffiLibName(platform = process.platform) {
  if (platform === 'win32' || String(platform).toLowerCase().startsWith('win')) return 'aria_ffi.dll';
  if (platform === 'darwin') return 'libaria_ffi.dylib';
  return 'libaria_ffi.so';
}

function libDir() {
  const path = require('path');
  return path.join(ariaHome(), 'lib');
}

function cachedFfiPath(platform = process.platform) {
  const path = require('path');
  const fs = require('fs');
  const candidate = path.join(libDir(), ffiLibName(platform));
  return fs.existsSync(candidate) ? candidate : undefined;
}

function ffiAssetOs(platform = process.platform, arch = process.arch) {
  const p = String(platform).toLowerCase();
  const a = String(arch).toLowerCase();
  if (p === 'linux' && (a === 'x64' || a === 'x86_64' || a === 'amd64')) return 'linux_x86_64';
  if (p === 'linux' && (a === 'arm64' || a === 'aarch64')) return 'linux_arm64';
  if (p === 'darwin' || p === 'macos') return 'macos';
  if ((p === 'win32' || p.startsWith('win')) && (a === 'x64' || a === 'x86_64' || a === 'amd64')) {
    return 'windows_x86_64';
  }
  throw new Error(`unsupported platform ${platform}/${arch} for libaria_ffi`);
}

function stripV(tag) {
  const t = String(tag).trim();
  return t.startsWith('v') || t.startsWith('V') ? t.slice(1) : t;
}

function selectLatestStable(releases) {
  let bestTag;
  let bestKey = [-1, -1, -1];
  for (const rel of releases) {
    if (rel.draft || rel.prerelease) continue;
    const tag = String(rel.tag_name || rel.tag || '');
    const core = stripV(tag).split('-')[0].split('+')[0];
    const parts = core.split('.');
    if (!parts.length || !/^\d+$/.test(parts[0])) continue;
    const parsed = [
      Number(parts[0]),
      parts[1] && /^\d+$/.test(parts[1]) ? Number(parts[1]) : 0,
      parts[2] && /^\d+$/.test(parts[2]) ? Number(parts[2]) : 0,
    ];
    if (
      parsed[0] > bestKey[0] ||
      (parsed[0] === bestKey[0] && parsed[1] > bestKey[1]) ||
      (parsed[0] === bestKey[0] && parsed[1] === bestKey[1] && parsed[2] > bestKey[2])
    ) {
      bestKey = parsed;
      bestTag = tag;
    }
  }
  if (!bestTag) throw new Error('no stable release found for libaria_ffi');
  return stripV(bestTag);
}

function upgradeOrg(site) {
  const cfg = configYmlScalar('upgrade_url');
  if (cfg) return cfg.replace(/\/$/, '');
  const hint = String(site || configYmlScalar('site_url') || DEFAULT_SITE).toLowerCase();
  if (hint.includes('ariacompute.cn') || hint.includes('gitee.com')) {
    return 'https://gitee.com/ariacompute';
  }
  return 'https://github.com/ariacompute';
}

function releasesApiUrl(org) {
  const owner = org.replace(/\/$/, '').split('/').pop() || 'ariacompute';
  if (org.toLowerCase().includes('gitee.com')) {
    return `https://gitee.com/api/v5/repos/${owner}/engine/releases?per_page=30`;
  }
  return `https://api.github.com/repos/${owner}/engine/releases?per_page=30`;
}

function extractFfiArchive(archive, destDir, want = ffiLibName()) {
  const fs = require('fs');
  const path = require('path');
  const tar = zlib.gunzipSync(fs.readFileSync(archive));
  let offset = 0;
  while (offset + 512 <= tar.length) {
    const header = tar.subarray(offset, offset + 512);
    if (header.every((b) => b === 0)) break;
    const name = header.subarray(0, 100).toString('utf8').replace(/\0.*$/, '');
    const sizeOctal = header.subarray(124, 136).toString('utf8').replace(/\0/g, '').trim();
    const size = Number.parseInt(sizeOctal, 8) || 0;
    const typeFlag = header[156];
    offset += 512;
    const isFile = typeFlag === 0 || typeFlag === 48;
    if (isFile && path.basename(name) === want) {
      fs.mkdirSync(destDir, { recursive: true });
      const dest = path.join(destDir, want);
      fs.writeFileSync(dest, tar.subarray(offset, offset + size));
      try {
        fs.chmodSync(dest, 0o755);
      } catch {
        /* windows */
      }
      return dest;
    }
    offset += Math.ceil(size / 512) * 512;
  }
  throw new Error(`${want} not found in ${archive}`);
}

async function httpGetBytes(url) {
  const resp = await fetch(url, { headers: { 'User-Agent': SDK_UA }, redirect: 'follow' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} ${url}`);
  return Buffer.from(await resp.arrayBuffer());
}

async function ensureFfiLib(site) {
  const fs = require('fs');
  const path = require('path');
  const env = process.env.ARIA_FFI_LIB;
  if (env && fs.existsSync(env)) return env;
  const cached = cachedFfiPath();
  if (cached) return cached;
  const org = upgradeOrg(site);
  const raw = await httpGetBytes(releasesApiUrl(org));
  const releases = JSON.parse(raw.toString('utf8'));
  if (!Array.isArray(releases)) throw new Error('unexpected releases payload');
  const ver = selectLatestStable(releases);
  const assetName = `libaria_ffi_${ver}_${ffiAssetOs()}.tar.gz`;
  let url;
  for (const rel of releases) {
    const tag = String(rel.tag_name || rel.tag || '');
    if (stripV(tag) !== ver) continue;
    for (const asset of rel.assets || []) {
      if (asset.name === assetName) {
        url = asset.browser_download_url || asset.direct_asset_url;
        break;
      }
    }
    if (url) break;
  }
  if (!url) throw new Error(`release asset not found: ${assetName}`);
  const staging = path.join(ariaHome(), 'tmp', `ffi-${ver}`);
  fs.rmSync(staging, { recursive: true, force: true });
  fs.mkdirSync(staging, { recursive: true });
  const archive = path.join(staging, assetName);
  try {
    fs.writeFileSync(archive, await httpGetBytes(url));
    return extractFfiArchive(archive, libDir(), ffiLibName());
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
}

export class AriaEngine {
  constructor(bundlePath) {
    this.bundlePath = bundlePath;
    // NativeModules.AriaEngine.init(bundlePath) when linked.
  }

  static async open(modelRef, opts = {}) {
    await ensureFfiLib(opts.site);
    if (isLocalRef(modelRef)) return new AriaEngine(modelRef);
    const bundle = await downloadModel(modelRef, opts);
    return new AriaEngine(bundle);
  }

  async complete(messages, options = { max_tokens: 16 }, tools = []) {
    // return NativeModules.AriaEngine.complete(...)
    return {
      success: true,
      response: '',
      function_calls: [],
      note: 'Link native module to libaria_ffi',
    };
  }

  async close() {}
}
