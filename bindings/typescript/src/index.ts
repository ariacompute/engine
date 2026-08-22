import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as fflate from "fflate";

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
  /** Dashboard bearer token. Required when modelRef is a model name. */
  token?: string;
  /** Dashboard base URL. Defaults to https://ariacompute.com. */
  site?: string;
  /** Explicit path to the FFI library. */
  ffiLib?: string;
}

const DEFAULT_SITE = "https://ariacompute.com";

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
    const suffix = model.slice(idx + 2);
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

function flattenSingleSubdir(dir: string): void {
  const entries = fs.readdirSync(dir).filter((e) => !e.startsWith("."));
  if (entries.length !== 1) return;
  const only = path.join(dir, entries[0]);
  if (!fs.statSync(only).isDirectory()) return;
  if (!fs.existsSync(path.join(only, "config.json"))) return;
  for (const name of fs.readdirSync(only)) {
    fs.renameSync(path.join(only, name), path.join(dir, name));
  }
  fs.rmdirSync(only);
}

async function extractZip(data: Buffer, dest: string): Promise<void> {
  if (data[0] !== 0x50 || data[1] !== 0x4b) {
    throw new Error("downloaded archive is not a valid zip");
  }
  fs.mkdirSync(dest, { recursive: true });
  const entries = fflate.unzipSync(data);
  for (const [name, content] of Object.entries(entries)) {
    const out = path.join(dest, name);
    if (name.endsWith("/")) {
      fs.mkdirSync(out, { recursive: true });
    } else {
      fs.mkdirSync(path.dirname(out), { recursive: true });
      fs.writeFileSync(out, content);
    }
  }
  flattenSingleSubdir(dest);
}

async function httpGet(url: string, token: string): Promise<{ ok: boolean; status: number; body: Buffer | any }> {
  const resp = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
  if (!resp.ok) return { ok: false, status: resp.status, body: null };
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("json")) return { ok: true, status: resp.status, body: await resp.json() };
  return { ok: true, status: resp.status, body: Buffer.from(await resp.arrayBuffer()) };
}

const encoder = new TextEncoder();

/** Download `model` from the Dashboard private source into
 * `~/.ariacompute/models/{model}` and return that directory.
 * Skips the download when a valid bundle is already cached. */
export async function downloadModel(
  model: string,
  token: string,
  site: string = DEFAULT_SITE,
): Promise<string> {
  if (!token) throw new Error("api token is required to download a model");
  const { slug, quant } = parseBundleName(model);
  const cache = cacheDir(model);
  if (fs.existsSync(cache) && isValidBundle(cache)) return cache;

  const metaUrl = `${site.replace(/\/$/, "")}/api/dashboard/models/${encodeURIComponent(
    slug,
  )}/download?quant=${encodeURIComponent(quant)}&sdk=v1.0&format=json`;

  const metaResp = await httpGet(metaUrl, token);
  if (!metaResp.ok) throw new Error(`dashboard request failed: ${metaResp.status}`);
  const meta = metaResp.body as { url?: string };
  if (!meta.url) throw new Error("dashboard meta returned empty url");

  const zipResp = await httpGet(meta.url, token);
  if (!zipResp.ok) throw new Error(`download stream failed: ${zipResp.status}`);
  const data = zipResp.body as Buffer;

  const staging = path.join(ariaHome(), "models", `.${model}.partial`);
  if (fs.existsSync(staging)) fs.rmSync(staging, { recursive: true, force: true });
  await extractZip(data, staging);
  if (!isValidBundle(staging)) {
    fs.rmSync(staging, { recursive: true, force: true });
    throw new Error("downloaded archive did not contain a valid aria-quant-bundle");
  }
  if (fs.existsSync(cache)) fs.rmSync(cache, { recursive: true, force: true });
  fs.renameSync(staging, cache);
  return cache;
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
    this.fnComplete = this.lib.func("aria_complete", "int", ["void*", "str", "str", "str", "str", "size_t"]);
    this.fnEmbed = this.lib.func("aria_embed", "int", ["void*", "str", "str", "size_t"]);
    this.fnTranscribe = this.lib.func("aria_transcribe", "int", ["void*", "void*", "size_t", "str", "str", "size_t"]);
    this.fnLastError = this.lib.func("aria_last_error", "str", []);
    this.handle = this.fnInit(bundle) as unknown;
    if (!this.handle) {
      const err = this.fnLastError();
      throw new Error(err || "init failed");
    }
  }

  /** Auto-detect: a value containing a separator or already on disk is a local
   * path; otherwise it is a model name that is downloaded (requires `token`)
   * then loaded. */
  static async open(modelRef: string, opts: OpenOptions = {}): Promise<Engine> {
    if (isLocalRef(modelRef)) return new Engine(modelRef, opts);
    if (!opts.token) {
      throw new Error(`model name '${modelRef}' requires an api token to download`);
    }
    const bundle = await downloadModel(modelRef, opts.token, opts.site ?? DEFAULT_SITE);
    return new Engine(bundle, opts);
  }

  complete(messages: Turn[], opts: GenerateOptions = {}): CompleteResult {
    const payload = JSON.stringify({ messages, opts });
    const buf = Buffer.alloc(1 << 16);
    const rc = this.fnComplete(
      this.handle,
      payload,
      "",
      "",
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
    const rc = this.fnEmbed(this.handle, text, buf, buf.length);
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
      pcm,
      pcm.length,
      "",
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
