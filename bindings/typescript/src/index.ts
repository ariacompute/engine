import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

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
  /** Hugging Face hub token (`.com`). Same field as `aria-engine auth` `hf_token`. */
  hfToken?: string;
  /** ModelScope hub token (`.cn`). Same field as `aria-engine auth` `modelscope_api_token`. */
  modelscopeApiToken?: string;
  /** Site used to pick the regional hub. Defaults to https://ariacompute.com (.com → HF, .cn → ModelScope). */
  site?: string;
  /** Explicit path to the FFI library. */
  ffiLib?: string;
}

const DEFAULT_SITE = "https://ariacompute.com";
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
  try {
    const raw = fs.readFileSync(path.join(ariaHome(), "config.yml"), "utf8");
    for (const line of raw.split("\n")) {
      if (line.startsWith(" ") || line.startsWith("\t")) continue;
      const s = line.trim();
      if (!s || s.startsWith("#") || !s.includes(":")) continue;
      const idx = s.indexOf(":");
      if (s.slice(0, idx).trim() !== key) continue;
      const v = unquoteYaml(s.slice(idx + 1));
      return v || undefined;
    }
  } catch {
    return undefined;
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
          `auth failed HTTP ${e.status}; set ${field} via aria-engine auth (do not pass a Dashboard sk-/bfvk- key as the hub token)`,
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

function loadLib(ffiLib?: string): any {
  const libPath = ffiLib || process.env.ARIA_FFI_LIB;
  if (!libPath) throw new Error("aria FFI lib not found; set ARIA_FFI_LIB or pass ffiLib");
  const koffiNS = require("koffi") as unknown as { load: (p: string) => any };
  return koffiNS.load(libPath);
}

export function isLocalRef(modelRef: string): boolean {
  return modelRef.includes("/") || modelRef.includes("\\") || fs.existsSync(modelRef);
}

export class Engine {
  private lib: any;
  private handle: unknown;
  private fnInit: any;
  private fnDestroy: any;
  private fnComplete: any;
  private fnEmbed: any;
  private fnTranscribe: any;
  private fnLastError: any;

  /** Construct from a local bundle directory. */
  constructor(bundle: string, opts: OpenOptions = {}) {
    this.lib = loadLib(opts.ffiLib);
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

  /** Auto-detect: a value containing a separator or already on disk is a local
   * path; otherwise it is a model name downloaded from the regional public hub. */
  static async open(modelRef: string, opts: OpenOptions = {}): Promise<Engine> {
    if (isLocalRef(modelRef)) return new Engine(modelRef, opts);
    const bundle = await downloadModel(modelRef, opts);
    return new Engine(bundle, opts);
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
    if (this.handle) this.fnDestroy(this.handle);
    this.handle = null;
  }
}

export const _internal = { encoder };
