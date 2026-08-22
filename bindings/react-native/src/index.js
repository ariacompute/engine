/**
 * React Native surface — native module bridges to libaria_ffi (Turbo Module / JSI).
 * JS API mirrors @ariacompute/engine-ts for docs samples.
 *
 * Auto-download: `AriaEngine.open(modelRef, { token, site })` resolves a model
 * name via the Dashboard private source (requires `token`), downloads the zip,
 * extracts it through the native `AriaEngineModule` (which bridges unzip and
 * writes to `~/.ariacompute/models/{model}`), then loads via the native init.
 */
import { NativeModules, Platform } from 'react-native';

const AriaEngineModule = NativeModules.AriaEngine;

const DEFAULT_SITE = 'https://ariacompute.com';

function parseBundleName(model) {
  if (!model || model.includes('/') || model.includes('\\')) {
    throw new Error(`invalid model name: ${model}`);
  }
  const idx = model.lastIndexOf('_q');
  if (idx !== -1) {
    const slug = model.slice(0, idx);
    const suffix = model.slice(idx + 2);
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

async function downloadModel(model, token, site = DEFAULT_SITE) {
  if (!token) throw new Error('dashboard token is required to download a model');
  const { slug, quant } = parseBundleName(model);
  const cache = cacheDir(model);
  if (require('fs').existsSync(cache) && (await AriaEngineModule?.isValidBundle?.(cache))) {
    return cache;
  }
  if (!AriaEngineModule?.downloadModel) {
    throw new Error('native AriaEngineModule.downloadModel bridge is not linked');
  }
  const metaUrl = `${site.replace(/\/$/, '')}/api/dashboard/models/${encodeURIComponent(
    slug,
  )}/download?quant=${encodeURIComponent(quant)}&sdk=v1.0&format=json`;
  const metaResp = await fetch(metaUrl, { headers: { Authorization: `Bearer ${token}` } });
  if (!metaResp.ok) throw new Error(`dashboard request failed: ${metaResp.status}`);
  const meta = await metaResp.json();
  if (!meta.url) throw new Error('dashboard meta returned empty url');
  // Delegates streaming + zip extraction (and bundle validation) to native.
  return AriaEngineModule.downloadModel(model, meta.url, token, cache);
}

export class AriaEngine {
  constructor(bundlePath) {
    this.bundlePath = bundlePath;
    // NativeModules.AriaEngine.init(bundlePath) when linked.
  }

  static async open(modelRef, opts = {}) {
    if (isLocalRef(modelRef)) return new AriaEngine(modelRef);
    if (!opts.token) {
      throw new Error(`model name '${modelRef}' requires a dashboard token to download`);
    }
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
