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

async function downloadModel(model, token, site = DEFAULT_SITE) {
  parseBundleName(model);
  const fs = require('fs');
  const path = require('path');
  const source = preferredPublicHub(site);
  const hubToken = hubBearer(token);
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

export class AriaEngine {
  constructor(bundlePath) {
    this.bundlePath = bundlePath;
    // NativeModules.AriaEngine.init(bundlePath) when linked.
  }

  static async open(modelRef, opts = {}) {
    if (isLocalRef(modelRef)) return new AriaEngine(modelRef);
    const bundle = await downloadModel(modelRef, opts.token, opts.site ?? DEFAULT_SITE);
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
