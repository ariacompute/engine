"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports._internal = exports.Engine = void 0;
exports.parseBundleName = parseBundleName;
exports.downloadModel = downloadModel;
exports.isLocalRef = isLocalRef;
const fs = __importStar(require("node:fs"));
const os = __importStar(require("node:os"));
const path = __importStar(require("node:path"));
const fflate = __importStar(require("fflate"));
const DEFAULT_SITE = "https://ariacompute.com";
function ariaHome() {
    return process.env.ARIA_COMPUTE_HOME || path.join(os.homedir(), ".ariacompute");
}
function cacheDir(model) {
    return path.join(ariaHome(), "models", model);
}
/** Parse `slug`/`quant` from a model name such as `gemma-4-e2b-it_q4`. */
function parseBundleName(model) {
    if (!model || model.includes("/") || model.includes("\\")) {
        throw new Error(`invalid model name: ${model}`);
    }
    const idx = model.lastIndexOf("_q");
    if (idx !== -1) {
        const slug = model.slice(0, idx);
        const suffix = model.slice(idx + 2);
        const quant = suffix === "4"
            ? "int4"
            : suffix === "8"
                ? "int8"
                : suffix === "326" || suffix === "3.26"
                    ? "int326"
                    : (() => {
                        throw new Error(`unknown quant suffix _q${suffix}`);
                    })();
        if (!slug)
            throw new Error(`invalid model name: ${model}`);
        return { slug, quant };
    }
    return { slug: model, quant: "int4" };
}
function isValidBundle(dir) {
    try {
        if (!fs.statSync(path.join(dir, "weight.bin")).isFile())
            return false;
        const meta = JSON.parse(fs.readFileSync(path.join(dir, "config.json"), "utf8"));
        return meta.format === "aria-quant-bundle";
    }
    catch {
        return false;
    }
}
function flattenSingleSubdir(dir) {
    const entries = fs.readdirSync(dir).filter((e) => !e.startsWith("."));
    if (entries.length !== 1)
        return;
    const only = path.join(dir, entries[0]);
    if (!fs.statSync(only).isDirectory())
        return;
    if (!fs.existsSync(path.join(only, "config.json")))
        return;
    for (const name of fs.readdirSync(only)) {
        fs.renameSync(path.join(only, name), path.join(dir, name));
    }
    fs.rmdirSync(only);
}
async function extractZip(data, dest) {
    if (data[0] !== 0x50 || data[1] !== 0x4b) {
        throw new Error("downloaded archive is not a valid zip");
    }
    fs.mkdirSync(dest, { recursive: true });
    const entries = fflate.unzipSync(data);
    for (const [name, content] of Object.entries(entries)) {
        const out = path.join(dest, name);
        if (name.endsWith("/")) {
            fs.mkdirSync(out, { recursive: true });
        }
        else {
            fs.mkdirSync(path.dirname(out), { recursive: true });
            fs.writeFileSync(out, content);
        }
    }
    flattenSingleSubdir(dest);
}
async function httpGet(url, token) {
    const resp = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (!resp.ok)
        return { ok: false, status: resp.status, body: null };
    const ct = resp.headers.get("content-type") || "";
    if (ct.includes("json"))
        return { ok: true, status: resp.status, body: await resp.json() };
    return { ok: true, status: resp.status, body: Buffer.from(await resp.arrayBuffer()) };
}
const encoder = new TextEncoder();
/** Download `model` from the Dashboard private source into
 * `~/.ariacompute/models/{model}` and return that directory.
 * Skips the download when a valid bundle is already cached. */
async function downloadModel(model, token, site = DEFAULT_SITE) {
    if (!token)
        throw new Error("api token is required to download a model");
    const { slug, quant } = parseBundleName(model);
    const cache = cacheDir(model);
    if (fs.existsSync(cache) && isValidBundle(cache))
        return cache;
    const metaUrl = `${site.replace(/\/$/, "")}/api/dashboard/models/${encodeURIComponent(slug)}/download?quant=${encodeURIComponent(quant)}&sdk=v1.0&format=json`;
    const metaResp = await httpGet(metaUrl, token);
    if (!metaResp.ok)
        throw new Error(`dashboard request failed: ${metaResp.status}`);
    const meta = metaResp.body;
    if (!meta.url)
        throw new Error("dashboard meta returned empty url");
    const zipResp = await httpGet(meta.url, token);
    if (!zipResp.ok)
        throw new Error(`download stream failed: ${zipResp.status}`);
    const data = zipResp.body;
    const staging = path.join(ariaHome(), "models", `.${model}.partial`);
    if (fs.existsSync(staging))
        fs.rmSync(staging, { recursive: true, force: true });
    await extractZip(data, staging);
    if (!isValidBundle(staging)) {
        fs.rmSync(staging, { recursive: true, force: true });
        throw new Error("downloaded archive did not contain a valid aria-quant-bundle");
    }
    if (fs.existsSync(cache))
        fs.rmSync(cache, { recursive: true, force: true });
    fs.renameSync(staging, cache);
    return cache;
}
function loadLib(ffiLib) {
    const libPath = ffiLib || process.env.ARIA_FFI_LIB;
    if (!libPath)
        throw new Error("aria FFI lib not found; set ARIA_FFI_LIB or pass ffiLib");
    const koffiNS = require("koffi");
    return koffiNS.load(libPath);
}
function isLocalRef(modelRef) {
    return modelRef.includes("/") || modelRef.includes("\\") || fs.existsSync(modelRef);
}
class Engine {
    /** Construct from a local bundle directory. */
    constructor(bundle, opts = {}) {
        this.lib = loadLib(opts.ffiLib);
        this.fnInit = this.lib.func("aria_model_init", "void*", ["str"]);
        this.fnDestroy = this.lib.func("aria_model_destroy", "void", ["void*"]);
        this.fnComplete = this.lib.func("aria_complete", "int", ["void*", "str", "str", "str", "str", "size_t"]);
        this.fnEmbed = this.lib.func("aria_embed", "int", ["void*", "str", "str", "size_t"]);
        this.fnTranscribe = this.lib.func("aria_transcribe", "int", ["void*", "void*", "size_t", "str", "str", "size_t"]);
        this.fnLastError = this.lib.func("aria_last_error", "str", []);
        this.handle = this.fnInit(bundle);
        if (!this.handle) {
            const err = this.fnLastError();
            throw new Error(err || "init failed");
        }
    }
    /** Auto-detect: a value containing a separator or already on disk is a local
     * path; otherwise it is a model name that is downloaded (requires `token`)
     * then loaded. */
    static async open(modelRef, opts = {}) {
        if (isLocalRef(modelRef))
            return new Engine(modelRef, opts);
        if (!opts.token) {
            throw new Error(`model name '${modelRef}' requires an api token to download`);
        }
        const bundle = await downloadModel(modelRef, opts.token, opts.site ?? DEFAULT_SITE);
        return new Engine(bundle, opts);
    }
    complete(messages, opts = {}) {
        const payload = JSON.stringify({ messages, opts });
        const buf = Buffer.alloc(1 << 16);
        const rc = this.fnComplete(this.handle, payload, "", "", buf, buf.length);
        if (rc !== 0) {
            const err = this.fnLastError();
            return { success: false, response: "", error: err || "complete failed" };
        }
        try {
            const parsed = JSON.parse(buf.toString("utf8").replace(/\0+$/, ""));
            return { success: true, response: parsed.response ?? "", generation: parsed };
        }
        catch {
            return { success: true, response: buf.toString("utf8").replace(/\0+$/, "") };
        }
    }
    embed(text) {
        const buf = Buffer.alloc(1 << 20);
        const rc = this.fnEmbed(this.handle, text, buf, buf.length);
        if (rc !== 0) {
            const err = this.fnLastError();
            throw new Error(err || "embed failed");
        }
        return JSON.parse(buf.toString("utf8").replace(/\0+$/, ""));
    }
    transcribe(pcm) {
        const buf = Buffer.alloc(1 << 16);
        const rc = this.fnTranscribe(this.handle, pcm, pcm.length, "", buf, buf.length);
        if (rc !== 0) {
            const err = this.fnLastError();
            throw new Error(err || "transcribe failed");
        }
        return buf.toString("utf8").replace(/\0+$/, "");
    }
    close() {
        if (this.handle)
            this.fnDestroy(this.handle);
        this.handle = null;
    }
}
exports.Engine = Engine;
exports._internal = { encoder };
