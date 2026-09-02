import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import * as zlib from "node:zlib";

export interface GenerateOptions {
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
  stop?: string[];
  response_format?: { type: "text" | "json_object" | "json_schema"; json_schema?: unknown };
}

export interface Turn {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface Generation {
  success: boolean;
  response: string;
  error?: string;
  tokens_used?: number;
  finish_reason?: string;
}

export interface CompleteResult {
  success: boolean;
  response: string;
  error?: string;
  generation?: Generation;
}

export interface OpenOptions {
  /** Legacy generic hub token. Dashboard sk-/bfvk- keys are ignored. */
  token?: string;
  /** Hugging Face hub token (`.com`). Same field as `aria-engine setup` `hf_token`. */
  hfToken?: string;
  /** ModelScope hub token (`.cn`). Same field as `aria-engine setup` `modelscope_api_token`. */
  modelscopeApiToken?: string;
  /** Site used to pick the regional hub. Defaults to https://ariacompute.com (.com → HF, .cn → ModelScope). */
  site?: string;
  /** Explicit path to the FFI library. */
  ffiLib?: string;
}

const DEFAULT_SITE = "https://ariacompute.com";
export const INTL_SITE = "https://ariacompute.com";
export const INTL_UPGRADE = "https://github.com/ariacompute";
export const CN_SITE = "https://ariacompute.cn";
export const CN_UPGRADE = "https://gitee.com/ariacompute";

export interface SetupConfig {
  router: string;
  router_api_key: string;
  site_url: string;
  upgrade_url: string;
  compute: string;
  hf_token: string;
  modelscope_api_token: string;
}

export function defaultSetupConfig(): SetupConfig {
  return {
    router: "",
    router_api_key: "",
    site_url: "",
    upgrade_url: "",
    compute: "auto",
    hf_token: "",
    modelscope_api_token: "",
  };
}

function gatewayRegion(url: string): "cn" | "intl" | undefined {
  const lower = (url || "").toLowerCase();
  if (lower.includes("ariacompute.cn") || lower.includes("gitee.com/ariacompute")) return "cn";
  if (lower.includes("ariacompute.com") || lower.includes("github.com/ariacompute")) return "intl";
  return undefined;
}

function pairUrls(region: "cn" | "intl"): [string, string] {
  return region === "cn" ? [CN_SITE, CN_UPGRADE] : [INTL_SITE, INTL_UPGRADE];
}

export function fillSetupUrls(cfg: SetupConfig): SetupConfig {
  const out = { ...cfg };
  const region = gatewayRegion(out.site_url) || gatewayRegion(out.upgrade_url);
  if (!region) return out;
  const [site, upgrade] = pairUrls(region);
  if (!out.site_url) out.site_url = site;
  if (!out.upgrade_url) out.upgrade_url = upgrade;
  return out;
}

export function applySetup(existing: SetupConfig, updates: Partial<SetupConfig>): SetupConfig {
  const out: SetupConfig = { ...existing };
  for (const [k, v] of Object.entries(updates)) {
    if (v === undefined) continue;
    (out as unknown as Record<string, unknown>)[k] = v;
  }
  if (!["auto", "cpu", "cuda"].includes(out.compute)) {
    throw new Error(`invalid compute: ${out.compute}`);
  }
  return fillSetupUrls(out);
}

const DEFAULT_SDK = "v1.0";
const HUB_REQUIRED = ["config.json", "weight.bin"] as const;
const HUB_OPTIONAL = [
  "tokenizer.json",
  "tokenizer.model",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "vocab.json",
  "merges.txt",
] as const;

function ariaHome(): string {
  return process.env.ARIA_COMPUTE_HOME || path.join(os.homedir(), ".ariacompute");
}

function cacheDir(model: string): string {
  return path.join(ariaHome(), "models", model);
}

/** Parse `slug`/`quant` from a model name such as `gemma-4-e2b-it_q4`. */
export function parseBundleName(model: string): { slug: string; quant: string } {
  if (!model || model.includes("/") || model.includes("\\")) {
    throw new Error(`invalid model name: ${model}`);
  }
  const idx = model.lastIndexOf("_q");
  if (idx !== -1) {
    const slug = model.slice(0, idx);
    let suffix = model.slice(idx + 2);
    if (suffix.endsWith("_channel") || suffix.endsWith("_group")) {
      suffix = suffix.slice(0, suffix.lastIndexOf("_"));
    }
    const quant =
      suffix === "4"
        ? "int4"
        : suffix === "8"
          ? "int8"
          : suffix === "326" || suffix === "3.26"
            ? "int326"
            : (() => {
                throw new Error(`unknown quant suffix _q${suffix}`);
              })();
    if (!slug) throw new Error(`invalid model name: ${model}`);
    return { slug, quant };
  }
  return { slug: model, quant: "int4" };
}

function isValidBundle(dir: string): boolean {
  try {
    if (!fs.statSync(path.join(dir, "weight.bin")).isFile()) return false;
    const meta = JSON.parse(fs.readFileSync(path.join(dir, "config.json"), "utf8"));
    return meta.format === "aria-quant-bundle";
  } catch {
    return false;
  }
}

export function preferredPublicHub(site?: string): "huggingface" | "modelscope" {
  if (site && site.toLowerCase().includes("ariacompute.cn")) return "modelscope";
  return "huggingface";
}

export function hubBearer(token?: string): string | undefined {
  const t = (token || "").trim();
  if (!t) return undefined;
  const low = t.toLowerCase();
  if (low.startsWith("sk-") || low.startsWith("bfvk-")) return undefined;
  return t;
}

function unquoteYaml(v: string): string {
  const t = v.trim();
  if (t.length >= 2 && ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'")))) {
    return t.slice(1, -1);
  }
  return t;
}

export function configYmlScalar(key: string): string | undefined {
  const home = ariaHome();
  for (const name of ["engine.yml", "config.yml"]) {
    try {
      const raw = fs.readFileSync(path.join(home, name), "utf8");
      for (const line of raw.split("\n")) {
        if (line.startsWith(" ") || line.startsWith("\t")) continue;
        const s = line.trim();
        if (!s || s.startsWith("#") || !s.includes(":")) continue;
        const idx = s.indexOf(":");
        if (s.slice(0, idx).trim() !== key) continue;
        const v = unquoteYaml(s.slice(idx + 1));
        if (v) return v;
      }
    } catch {
      continue;
    }
  }
  return undefined;
}

export function resolveHubToken(
  source: "huggingface" | "modelscope",
  opts: Pick<OpenOptions, "token" | "hfToken" | "modelscopeApiToken"> = {},
): string | undefined {
  const named = source === "modelscope" ? opts.modelscopeApiToken : opts.hfToken;
  const field = source === "modelscope" ? "modelscope_api_token" : "hf_token";
  for (const cand of [named, opts.token, configYmlScalar(field)]) {
    const b = hubBearer(cand);
    if (b) return b;
  }
  return undefined;
}

function hubPathNames(model: string): string[] {
  const names = [model];
  let lower = model.toLowerCase();
  let core = model;
  for (const suf of ["_channel", "_group"]) {
    if (lower.endsWith(suf)) {
      core = model.slice(0, -suf.length);
      lower = core.toLowerCase();
      break;
    }
  }
  const stems = [core];
  if (lower.endsWith("_q326")) stems.push(`${core.slice(0, -5)}q3.26`);
  else if (lower.endsWith("_q3.26")) stems.push(`${core.slice(0, -6)}q326`);
  for (const stem of stems) {
    for (const share of ["", "_channel", "_group"]) {
      const cand = `${stem}${share}`;
      if (!names.includes(cand)) names.push(cand);
    }
  }
  return names;
}

export function hubFileUrls(
  source: "huggingface" | "modelscope",
  model: string,
  file: string,
  sdk: string = DEFAULT_SDK,
): string[] {
  const urls: string[] = [];
  for (const name of hubPathNames(model)) {
    if (source === "modelscope") {
      for (const repo of [`AriaCompute/${name}`, "AriaCompute/model"]) {
        urls.push(`https://www.modelscope.cn/models/${repo}/resolve/master/${sdk}/${name}/${file}`);
        urls.push(`https://modelscope.cn/models/${repo}/resolve/master/${sdk}/${name}/${file}`);
      }
    } else {
      for (const repo of [`ariacompute/${name}`, "ariacompute/model"]) {
        urls.push(`https://huggingface.co/${repo}/resolve/main/${sdk}/${name}/${file}`);
      }
    }
  }
  return urls;
}

class HubHttpError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function fetchUrlToFile(url: string, dest: string, token?: string): Promise<void> {
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  const resp = await fetch(url, { headers });
  if (resp.status === 401 || resp.status === 403) {
    throw new HubHttpError(resp.status, `HTTP ${resp.status}`);
  }
  if (!resp.ok) throw new HubHttpError(resp.status, `HTTP ${resp.status}`);
  if (!resp.body) throw new Error("empty response body");
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  await pipeline(Readable.fromWeb(resp.body as any), fs.createWriteStream(dest));
}

async function fetchHubFile(
  source: "huggingface" | "modelscope",
  model: string,
  file: string,
  dest: string,
  token: string | undefined,
  required: boolean,
): Promise<boolean> {
  let last: unknown;
  for (const url of hubFileUrls(source, model, file)) {
    try {
      await fetchUrlToFile(url, dest, token);
      return true;
    } catch (e) {
      last = e;
      if (e instanceof HubHttpError && (e.status === 401 || e.status === 403)) {
        const field = source === "modelscope" ? "modelscope_api_token" : "hf_token";
        throw new Error(
          `auth failed HTTP ${e.status}; set ${field} via aria-engine setup (do not pass a Dashboard sk-/bfvk- key as the hub token)`,
        );
      }
    }
  }
  if (required) throw new Error(`${source}: missing ${file}${last ? `: ${last}` : ""}`);
  return false;
}

const encoder = new TextEncoder();

/** Download `model` from the regional public hub into
 * `~/.ariacompute/models/{model}` and return that directory.
 * Matches aria-engine download: .com → Hugging Face, .cn → ModelScope.
 * Dashboard is not used. Skips the download when a valid bundle is already cached. */
export async function downloadModel(
  model: string,
  tokenOrOpts?: string | OpenOptions,
  site: string = DEFAULT_SITE,
): Promise<string> {
  const opts: OpenOptions =
    tokenOrOpts && typeof tokenOrOpts === "object"
      ? tokenOrOpts
      : { token: tokenOrOpts, site };
  parseBundleName(model);
  const resolvedSite = opts.site ?? site ?? DEFAULT_SITE;
  const source = preferredPublicHub(resolvedSite);
  const hubToken = resolveHubToken(source, opts);
  const cache = cacheDir(model);
  if (fs.existsSync(cache) && isValidBundle(cache)) return cache;

  const staging = path.join(ariaHome(), "models", `.${model}.partial`);
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
    if (!isValidBundle(staging)) {
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

const SDK_UA = "aria-engine-sdk/0.1.0";

export function ffiLibName(platform: string = process.platform): string {
  if (platform === "win32" || platform.toLowerCase().startsWith("win")) return "aria-engine_ffi.dll";
  if (platform === "darwin") return "libaria-engine_ffi.dylib";
  return "libaria-engine_ffi.so";
}

function libDir(): string {
  return path.join(ariaHome(), "lib");
}

export function cachedFfiPath(platform: string = process.platform): string | undefined {
  const candidate = path.join(libDir(), ffiLibName(platform));
  return fs.existsSync(candidate) ? candidate : undefined;
}

function bundledFfiPath(): string | undefined {
  const candidate = path.join(__dirname, "lib", ffiLibName());
  return fs.existsSync(candidate) ? candidate : undefined;
}

export function ffiAssetOs(platform: string = process.platform, arch: string = process.arch): string {
  const p = platform.toLowerCase();
  const a = arch.toLowerCase();
  if (p === "linux" && (a === "x64" || a === "x86_64" || a === "amd64")) return "linux_x86_64";
  if (p === "linux" && (a === "arm64" || a === "aarch64")) return "linux_arm64";
  if (p === "darwin" || p === "macos") return "macos";
  if ((p === "win32" || p.startsWith("win")) && (a === "x64" || a === "x86_64" || a === "amd64")) {
    return "windows_x86_64";
  }
  throw new Error(`unsupported platform ${platform}/${arch} for libaria-engine_ffi`);
}

function stripV(tag: string): string {
  const t = tag.trim();
  return t.startsWith("v") || t.startsWith("V") ? t.slice(1) : t;
}

function parseSemver(tag: string): [number, number, number] | undefined {
  const core = stripV(tag).split("-", 1)[0].split("+", 1)[0];
  const parts = core.split(".");
  if (!parts.length || !/^\d+$/.test(parts[0])) return undefined;
  return [
    Number(parts[0]),
    parts[1] && /^\d+$/.test(parts[1]) ? Number(parts[1]) : 0,
    parts[2] && /^\d+$/.test(parts[2]) ? Number(parts[2]) : 0,
  ];
}

export function selectLatestStable(releases: Array<Record<string, unknown>>): string {
  let bestTag: string | undefined;
  let bestKey: [number, number, number] = [-1, -1, -1];
  for (const rel of releases) {
    if (rel.draft || rel.prerelease) continue;
    const tag = String(rel.tag_name || rel.tag || "");
    const parsed = parseSemver(tag);
    if (!parsed) continue;
    if (
      parsed[0] > bestKey[0] ||
      (parsed[0] === bestKey[0] && parsed[1] > bestKey[1]) ||
      (parsed[0] === bestKey[0] && parsed[1] === bestKey[1] && parsed[2] > bestKey[2])
    ) {
      bestKey = parsed;
      bestTag = tag;
    }
  }
  if (!bestTag) throw new Error("no stable release found for libaria-engine_ffi");
  return stripV(bestTag);
}

function upgradeOrg(site?: string): string {
  const cfg = configYmlScalar("upgrade_url");
  if (cfg) return cfg.replace(/\/$/, "");
  const hint = (site || configYmlScalar("site_url") || DEFAULT_SITE).toLowerCase();
  if (hint.includes("ariacompute.cn") || hint.includes("gitee.com")) {
    return "https://gitee.com/ariacompute";
  }
  return "https://github.com/ariacompute";
}

function releasesApiUrl(org: string): string {
  const owner = org.replace(/\/$/, "").split("/").pop() || "ariacompute";
  if (org.toLowerCase().includes("gitee.com")) {
    return `https://gitee.com/api/v5/repos/${owner}/engine/releases?per_page=30`;
  }
  return `https://api.github.com/repos/${owner}/engine/releases?per_page=30`;
}

export function extractFfiArchive(archive: string, destDir: string, want: string = ffiLibName()): string {
  const tar = zlib.gunzipSync(fs.readFileSync(archive));
  let offset = 0;
  while (offset + 512 <= tar.length) {
    const header = tar.subarray(offset, offset + 512);
    if (header.every((b) => b === 0)) break;
    const name = header.subarray(0, 100).toString("utf8").replace(/\0.*$/, "");
    const sizeOctal = header.subarray(124, 136).toString("utf8").replace(/\0/g, "").trim();
    const size = Number.parseInt(sizeOctal, 8) || 0;
    const typeFlag = header[156];
    offset += 512;
    const isFile = typeFlag === 0 || typeFlag === 48; // '\0' or '0'
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

function httpGetBytesSync(url: string): Buffer {
  const script =
    'fetch(process.env.ARIA_FFI_URL,{headers:{"User-Agent":"aria-engine-sdk/0.1.0"},redirect:"follow"}).then(async r=>{if(!r.ok){process.stderr.write("HTTP "+r.status);process.exit(1)}process.stdout.write(Buffer.from(await r.arrayBuffer()))}).catch(e=>{process.stderr.write(String(e&&e.message||e));process.exit(1)})';
  const r = spawnSync(process.execPath, ["-e", script], {
    env: { ...process.env, ARIA_FFI_URL: url },
    encoding: "buffer",
    maxBuffer: 256 * 1024 * 1024,
  });
  if (r.status !== 0) {
    throw new Error(`download failed: ${(r.stderr && r.stderr.toString()) || "exit " + r.status}`);
  }
  return r.stdout as Buffer;
}

async function httpGetBytes(url: string): Promise<Buffer> {
  const resp = await fetch(url, { headers: { "User-Agent": SDK_UA }, redirect: "follow" });
  if (!resp.ok) throw new Error(`HTTP ${resp.status} ${url}`);
  return Buffer.from(await resp.arrayBuffer());
}

function installFfiFromReleases(raw: Buffer, archiveBytes: (assetUrl: string) => Buffer): string {
  let releases: Array<Record<string, unknown>>;
  try {
    releases = JSON.parse(raw.toString("utf8"));
  } catch {
    throw new Error("invalid releases JSON");
  }
  if (!Array.isArray(releases)) throw new Error("unexpected releases payload");
  const ver = selectLatestStable(releases);
  const assetName = `libaria-engine_ffi_${ver}_${ffiAssetOs()}.tar.gz`;
  let url: string | undefined;
  for (const rel of releases) {
    const tag = String(rel.tag_name || rel.tag || "");
    if (stripV(tag) !== ver) continue;
    const assets = (rel.assets as Array<Record<string, string>> | undefined) || [];
    for (const asset of assets) {
      if (asset.name === assetName) {
        url = asset.browser_download_url || asset.direct_asset_url;
        break;
      }
    }
    if (url) break;
  }
  if (!url) throw new Error(`release asset not found: ${assetName}`);
  const staging = path.join(ariaHome(), "tmp", `ffi-${ver}`);
  fs.rmSync(staging, { recursive: true, force: true });
  fs.mkdirSync(staging, { recursive: true });
  const archive = path.join(staging, assetName);
  try {
    fs.writeFileSync(archive, archiveBytes(url));
    return extractFfiArchive(archive, libDir(), ffiLibName());
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
}

/** Return a path to libaria-engine_ffi, downloading the latest stable Release if needed. */
export async function ensureFfiLib(site?: string): Promise<string> {
  const env = process.env.ARIA_FFI_LIB;
  if (env && fs.existsSync(env)) return env;
  const bundled = bundledFfiPath();
  if (bundled) return bundled;
  const cached = cachedFfiPath();
  if (cached) return cached;
  const org = upgradeOrg(site);
  const raw = await httpGetBytes(releasesApiUrl(org));
  return installFfiFromReleases(raw, (assetUrl) => httpGetBytesSync(assetUrl));
}

function ensureFfiLibSync(site?: string): string {
  const env = process.env.ARIA_FFI_LIB;
  if (env && fs.existsSync(env)) return env;
  const bundled = bundledFfiPath();
  if (bundled) return bundled;
  const cached = cachedFfiPath();
  if (cached) return cached;
  const org = upgradeOrg(site);
  const raw = httpGetBytesSync(releasesApiUrl(org));
  return installFfiFromReleases(raw, httpGetBytesSync);
}

function loadLib(ffiLib?: string, site?: string): any {
  const libPath =
    ffiLib ||
    process.env.ARIA_FFI_LIB ||
    bundledFfiPath() ||
    cachedFfiPath() ||
    ensureFfiLibSync(site);
  if (!libPath) throw new Error("Cannot locate libaria-engine_ffi");
  const koffiNS = require("koffi") as unknown as { load: (p: string) => any };
  return koffiNS.load(libPath);
}

export function isLocalRef(modelRef: string): boolean {
  return modelRef.includes("/") || modelRef.includes("\\") || fs.existsSync(modelRef);
}

export class Engine {
  private lib: any = null;
  private handle: unknown = null;
  private fnInit: any;
  private fnDestroy: any;
  private fnComplete: any;
  private fnEmbed: any;
  private fnTranscribe: any;
  private fnLastError: any;
  private cfg: SetupConfig = defaultSetupConfig();
  private opts: OpenOptions = {};

  /** Empty construct, or a local bundle directory. */
  constructor(bundle?: string, opts: OpenOptions = {}) {
    this.opts = opts;
    if (opts.site || opts.hfToken || opts.modelscopeApiToken) {
      this.setup({
        site_url: opts.site,
        hf_token: opts.hfToken,
        modelscope_api_token: opts.modelscopeApiToken,
      });
    }
    if (bundle) this.bindAndInit(bundle, opts.ffiLib);
  }

  /** Set Config / Run fields on this instance only. Does not write engine.yml. */
  setup(updates: Partial<SetupConfig>): this {
    this.cfg = applySetup(this.cfg, updates);
    return this;
  }

  setupStatus(): SetupConfig {
    return { ...this.cfg };
  }

  /** Reset instance defaults. Does not delete ~/.ariacompute/engine.yml. */
  setupClear(): this {
    this.cfg = defaultSetupConfig();
    return this;
  }

  private bindAndInit(bundle: string, ffiLib?: string): void {
    this.lib = loadLib(ffiLib || this.opts.ffiLib, this.cfg.site_url || this.opts.site);
    this.fnInit = this.lib.func("aria_model_init", "void*", ["str"]);
    this.fnDestroy = this.lib.func("aria_model_destroy", "void", ["void*"]);
    this.fnComplete = this.lib.func("aria_complete", "int", ["void*", "str", "str", "str", "void*", "size_t"]);
    this.fnEmbed = this.lib.func("aria_embed", "int", ["void*", "str", "void*", "size_t"]);
    this.fnTranscribe = this.lib.func("aria_transcribe", "int", ["void*", "void*", "size_t", "str", "void*", "size_t"]);
    this.fnLastError = this.lib.func("aria_last_error", "str", []);
    this.handle = this.fnInit(bundle) as unknown;
    if (!this.handle) {
      const err = this.fnLastError();
      throw new Error(err || "init failed");
    }
  }

  /** Download (if needed) and load a model using instance setup. */
  async open(modelRef: string): Promise<this> {
    await ensureFfiLib(this.cfg.site_url || this.opts.site);
    const openOpts: OpenOptions = {
      ...this.opts,
      site: this.cfg.site_url || this.opts.site,
      hfToken: this.cfg.hf_token || this.opts.hfToken,
      modelscopeApiToken: this.cfg.modelscope_api_token || this.opts.modelscopeApiToken,
    };
    const bundle = isLocalRef(modelRef) ? modelRef : await downloadModel(modelRef, openOpts);
    if (this.handle) this.close();
    this.bindAndInit(bundle, this.opts.ffiLib);
    return this;
  }

  /** Auto-detect: a value containing a separator or already on disk is a local
   * path; otherwise it is a model name downloaded from the regional public hub. */
  static async open(modelRef: string, opts: OpenOptions = {}): Promise<Engine> {
    const eng = new Engine(undefined, opts);
    await eng.open(modelRef);
    return eng;
  }

  complete(messages: Turn[], opts: GenerateOptions = {}): CompleteResult {
    const messagesJson = JSON.stringify(messages);
    const optionsJson = JSON.stringify(opts || { max_tokens: 16 });
    const toolsJson = JSON.stringify([]);
    const buf = Buffer.alloc(1 << 16);
    const rc = this.fnComplete(
      this.handle,
      messagesJson,
      optionsJson,
      toolsJson,
      buf,
      buf.length,
    );
    if (rc !== 0) {
      const err = this.fnLastError();
      return { success: false, response: "", error: err || "complete failed" };
    }
    try {
      const parsed = JSON.parse(buf.toString("utf8").replace(/\0+$/, ""));
      return { success: true, response: parsed.response ?? "", generation: parsed };
    } catch {
      return { success: true, response: buf.toString("utf8").replace(/\0+$/, "") };
    }
  }

  embed(text: string): number[] {
    const buf = Buffer.alloc(1 << 20);
    const rc = this.fnEmbed(this.handle, JSON.stringify({ input: text }), buf, buf.length);
    if (rc !== 0) {
      const err = this.fnLastError();
      throw new Error(err || "embed failed");
    }
    return JSON.parse(buf.toString("utf8").replace(/\0+$/, ""));
  }

  transcribe(pcm: Uint8Array): string {
    const buf = Buffer.alloc(1 << 16);
    const rc = this.fnTranscribe(
      this.handle,
      Buffer.from(pcm),
      pcm.length,
      null,
      buf,
      buf.length,
    );
    if (rc !== 0) {
      const err = this.fnLastError();
      throw new Error(err || "transcribe failed");
    }
    return buf.toString("utf8").replace(/\0+$/, "");
  }

  close(): void {
    if (this.handle && this.fnDestroy) this.fnDestroy(this.handle);
    this.handle = null;
  }
}

export const _internal = { encoder };
