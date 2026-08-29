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
exports._internal = exports.Engine = exports.CN_UPGRADE = exports.CN_SITE = exports.INTL_UPGRADE = exports.INTL_SITE = void 0;
exports.defaultAuthConfig = defaultAuthConfig;
exports.fillAuthUrls = fillAuthUrls;
exports.applyAuth = applyAuth;
exports.parseBundleName = parseBundleName;
exports.preferredPublicHub = preferredPublicHub;
exports.hubBearer = hubBearer;
exports.configYmlScalar = configYmlScalar;
exports.resolveHubToken = resolveHubToken;
exports.hubFileUrls = hubFileUrls;
exports.downloadModel = downloadModel;
exports.ffiLibName = ffiLibName;
exports.cachedFfiPath = cachedFfiPath;
exports.ffiAssetOs = ffiAssetOs;
exports.selectLatestStable = selectLatestStable;
exports.extractFfiArchive = extractFfiArchive;
exports.ensureFfiLib = ensureFfiLib;
exports.isLocalRef = isLocalRef;
const node_child_process_1 = require("node:child_process");
const fs = __importStar(require("node:fs"));
const os = __importStar(require("node:os"));
const path = __importStar(require("node:path"));
const promises_1 = require("node:stream/promises");
const node_stream_1 = require("node:stream");
const zlib = __importStar(require("node:zlib"));
const DEFAULT_SITE = "https://ariacompute.com";
exports.INTL_SITE = "https://ariacompute.com";
exports.INTL_UPGRADE = "https://github.com/ariacompute";
exports.CN_SITE = "https://ariacompute.cn";
exports.CN_UPGRADE = "https://gitee.com/ariacompute";
function defaultAuthConfig() {
    return {
        router: "",
        site_url: "",
        upgrade_url: "",
        compute: "auto",
        hf_token: "",
        modelscope_api_token: "",
    };
}
function gatewayRegion(url) {
    const lower = (url || "").toLowerCase();
    if (lower.includes("ariacompute.cn") || lower.includes("gitee.com/ariacompute"))
        return "cn";
    if (lower.includes("ariacompute.com") || lower.includes("github.com/ariacompute"))
        return "intl";
    return undefined;
}
function pairUrls(region) {
    return region === "cn" ? [exports.CN_SITE, exports.CN_UPGRADE] : [exports.INTL_SITE, exports.INTL_UPGRADE];
}
function fillAuthUrls(cfg) {
    const out = { ...cfg };
    const region = gatewayRegion(out.site_url) || gatewayRegion(out.upgrade_url);
    if (!region)
        return out;
    const [site, upgrade] = pairUrls(region);
    if (!out.site_url)
        out.site_url = site;
    if (!out.upgrade_url)
        out.upgrade_url = upgrade;
    return out;
}
function applyAuth(existing, updates) {
    const out = { ...existing };
    for (const [k, v] of Object.entries(updates)) {
        if (v === undefined)
            continue;
        out[k] = v;
    }
    if (!["auto", "cpu", "cuda"].includes(out.compute)) {
        throw new Error(`invalid compute: ${out.compute}`);
    }
    return fillAuthUrls(out);
}
const DEFAULT_SDK = "v1.0";
const HUB_REQUIRED = ["config.json", "weight.bin"];
const HUB_OPTIONAL = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
];
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
        let suffix = model.slice(idx + 2);
        if (suffix.endsWith("_channel") || suffix.endsWith("_group")) {
            suffix = suffix.slice(0, suffix.lastIndexOf("_"));
        }
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
function preferredPublicHub(site) {
    if (site && site.toLowerCase().includes("ariacompute.cn"))
        return "modelscope";
    return "huggingface";
}
function hubBearer(token) {
    const t = (token || "").trim();
    if (!t)
        return undefined;
    const low = t.toLowerCase();
    if (low.startsWith("sk-") || low.startsWith("bfvk-"))
        return undefined;
    return t;
}
function unquoteYaml(v) {
    const t = v.trim();
    if (t.length >= 2 && ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'")))) {
        return t.slice(1, -1);
    }
    return t;
}
function configYmlScalar(key) {
    try {
        const raw = fs.readFileSync(path.join(ariaHome(), "config.yml"), "utf8");
        for (const line of raw.split("\n")) {
            if (line.startsWith(" ") || line.startsWith("\t"))
                continue;
            const s = line.trim();
            if (!s || s.startsWith("#") || !s.includes(":"))
                continue;
            const idx = s.indexOf(":");
            if (s.slice(0, idx).trim() !== key)
                continue;
            const v = unquoteYaml(s.slice(idx + 1));
            return v || undefined;
        }
    }
    catch {
        return undefined;
    }
    return undefined;
}
function resolveHubToken(source, opts = {}) {
    const named = source === "modelscope" ? opts.modelscopeApiToken : opts.hfToken;
    const field = source === "modelscope" ? "modelscope_api_token" : "hf_token";
    for (const cand of [named, opts.token, configYmlScalar(field)]) {
        const b = hubBearer(cand);
        if (b)
            return b;
    }
    return undefined;
}
function hubPathNames(model) {
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
    if (lower.endsWith("_q326"))
        stems.push(`${core.slice(0, -5)}q3.26`);
    else if (lower.endsWith("_q3.26"))
        stems.push(`${core.slice(0, -6)}q326`);
    for (const stem of stems) {
        for (const share of ["", "_channel", "_group"]) {
            const cand = `${stem}${share}`;
            if (!names.includes(cand))
                names.push(cand);
        }
    }
    return names;
}
function hubFileUrls(source, model, file, sdk = DEFAULT_SDK) {
    const urls = [];
    for (const name of hubPathNames(model)) {
        if (source === "modelscope") {
            for (const repo of [`AriaCompute/${name}`, "AriaCompute/model"]) {
                urls.push(`https://www.modelscope.cn/models/${repo}/resolve/master/${sdk}/${name}/${file}`);
                urls.push(`https://modelscope.cn/models/${repo}/resolve/master/${sdk}/${name}/${file}`);
            }
        }
        else {
            for (const repo of [`ariacompute/${name}`, "ariacompute/model"]) {
                urls.push(`https://huggingface.co/${repo}/resolve/main/${sdk}/${name}/${file}`);
            }
        }
    }
    return urls;
}
class HubHttpError extends Error {
    constructor(status, message) {
        super(message);
        this.status = status;
    }
}
async function fetchUrlToFile(url, dest, token) {
    const headers = {};
    if (token)
        headers.Authorization = `Bearer ${token}`;
    const resp = await fetch(url, { headers });
    if (resp.status === 401 || resp.status === 403) {
        throw new HubHttpError(resp.status, `HTTP ${resp.status}`);
    }
    if (!resp.ok)
        throw new HubHttpError(resp.status, `HTTP ${resp.status}`);
    if (!resp.body)
        throw new Error("empty response body");
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    await (0, promises_1.pipeline)(node_stream_1.Readable.fromWeb(resp.body), fs.createWriteStream(dest));
}
async function fetchHubFile(source, model, file, dest, token, required) {
    let last;
    for (const url of hubFileUrls(source, model, file)) {
        try {
            await fetchUrlToFile(url, dest, token);
            return true;
        }
        catch (e) {
            last = e;
            if (e instanceof HubHttpError && (e.status === 401 || e.status === 403)) {
                const field = source === "modelscope" ? "modelscope_api_token" : "hf_token";
                throw new Error(`auth failed HTTP ${e.status}; set ${field} via aria-engine auth (do not pass a Dashboard sk-/bfvk- key as the hub token)`);
            }
        }
    }
    if (required)
        throw new Error(`${source}: missing ${file}${last ? `: ${last}` : ""}`);
    return false;
}
const encoder = new TextEncoder();
/** Download `model` from the regional public hub into
 * `~/.ariacompute/models/{model}` and return that directory.
 * Matches aria-engine download: .com → Hugging Face, .cn → ModelScope.
 * Dashboard is not used. Skips the download when a valid bundle is already cached. */
async function downloadModel(model, tokenOrOpts, site = DEFAULT_SITE) {
    const opts = tokenOrOpts && typeof tokenOrOpts === "object"
        ? tokenOrOpts
        : { token: tokenOrOpts, site };
    parseBundleName(model);
    const resolvedSite = opts.site ?? site ?? DEFAULT_SITE;
    const source = preferredPublicHub(resolvedSite);
    const hubToken = resolveHubToken(source, opts);
    const cache = cacheDir(model);
    if (fs.existsSync(cache) && isValidBundle(cache))
        return cache;
    const staging = path.join(ariaHome(), "models", `.${model}.partial`);
    try {
        if (fs.existsSync(staging))
            fs.rmSync(staging, { recursive: true, force: true });
        fs.mkdirSync(staging, { recursive: true });
        for (const file of HUB_REQUIRED) {
            await fetchHubFile(source, model, file, path.join(staging, file), hubToken, true);
        }
        for (const extra of HUB_OPTIONAL) {
            try {
                await fetchHubFile(source, model, extra, path.join(staging, extra), hubToken, false);
            }
            catch {
                /* optional tokenizer sidecar */
            }
        }
        if (!isValidBundle(staging)) {
            throw new Error(`${source} fetch completed but bundle invalid (need weight.bin + aria-quant-bundle config.json)`);
        }
        if (fs.existsSync(cache))
            fs.rmSync(cache, { recursive: true, force: true });
        fs.renameSync(staging, cache);
        return cache;
    }
    catch (e) {
        if (fs.existsSync(staging))
            fs.rmSync(staging, { recursive: true, force: true });
        throw e;
    }
}
const SDK_UA = "aria-engine-sdk/0.1.0";
function ffiLibName(platform = process.platform) {
    if (platform === "win32" || platform.toLowerCase().startsWith("win"))
        return "aria_ffi.dll";
    if (platform === "darwin")
        return "libaria_ffi.dylib";
    return "libaria_ffi.so";
}
function libDir() {
    return path.join(ariaHome(), "lib");
}
function cachedFfiPath(platform = process.platform) {
    const candidate = path.join(libDir(), ffiLibName(platform));
    return fs.existsSync(candidate) ? candidate : undefined;
}
function bundledFfiPath() {
    const candidate = path.join(__dirname, "lib", ffiLibName());
    return fs.existsSync(candidate) ? candidate : undefined;
}
function ffiAssetOs(platform = process.platform, arch = process.arch) {
    const p = platform.toLowerCase();
    const a = arch.toLowerCase();
    if (p === "linux" && (a === "x64" || a === "x86_64" || a === "amd64"))
        return "linux_x86_64";
    if (p === "linux" && (a === "arm64" || a === "aarch64"))
        return "linux_arm64";
    if (p === "darwin" || p === "macos")
        return "macos";
    if ((p === "win32" || p.startsWith("win")) && (a === "x64" || a === "x86_64" || a === "amd64")) {
        return "windows_x86_64";
    }
    throw new Error(`unsupported platform ${platform}/${arch} for libaria_ffi`);
}
function stripV(tag) {
    const t = tag.trim();
    return t.startsWith("v") || t.startsWith("V") ? t.slice(1) : t;
}
function parseSemver(tag) {
    const core = stripV(tag).split("-", 1)[0].split("+", 1)[0];
    const parts = core.split(".");
    if (!parts.length || !/^\d+$/.test(parts[0]))
        return undefined;
    return [
        Number(parts[0]),
        parts[1] && /^\d+$/.test(parts[1]) ? Number(parts[1]) : 0,
        parts[2] && /^\d+$/.test(parts[2]) ? Number(parts[2]) : 0,
    ];
}
function selectLatestStable(releases) {
    let bestTag;
    let bestKey = [-1, -1, -1];
    for (const rel of releases) {
        if (rel.draft || rel.prerelease)
            continue;
        const tag = String(rel.tag_name || rel.tag || "");
        const parsed = parseSemver(tag);
        if (!parsed)
            continue;
        if (parsed[0] > bestKey[0] ||
            (parsed[0] === bestKey[0] && parsed[1] > bestKey[1]) ||
            (parsed[0] === bestKey[0] && parsed[1] === bestKey[1] && parsed[2] > bestKey[2])) {
            bestKey = parsed;
            bestTag = tag;
        }
    }
    if (!bestTag)
        throw new Error("no stable release found for libaria_ffi");
    return stripV(bestTag);
}
function upgradeOrg(site) {
    const cfg = configYmlScalar("upgrade_url");
    if (cfg)
        return cfg.replace(/\/$/, "");
    const hint = (site || configYmlScalar("site_url") || DEFAULT_SITE).toLowerCase();
    if (hint.includes("ariacompute.cn") || hint.includes("gitee.com")) {
        return "https://gitee.com/ariacompute";
    }
    return "https://github.com/ariacompute";
}
function releasesApiUrl(org) {
    const owner = org.replace(/\/$/, "").split("/").pop() || "ariacompute";
    if (org.toLowerCase().includes("gitee.com")) {
        return `https://gitee.com/api/v5/repos/${owner}/engine/releases?per_page=30`;
    }
    return `https://api.github.com/repos/${owner}/engine/releases?per_page=30`;
}
function extractFfiArchive(archive, destDir, want = ffiLibName()) {
    const tar = zlib.gunzipSync(fs.readFileSync(archive));
    let offset = 0;
    while (offset + 512 <= tar.length) {
        const header = tar.subarray(offset, offset + 512);
        if (header.every((b) => b === 0))
            break;
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
            }
            catch {
                /* windows */
            }
            return dest;
        }
        offset += Math.ceil(size / 512) * 512;
    }
    throw new Error(`${want} not found in ${archive}`);
}
function httpGetBytesSync(url) {
    const script = 'fetch(process.env.ARIA_FFI_URL,{headers:{"User-Agent":"aria-engine-sdk/0.1.0"},redirect:"follow"}).then(async r=>{if(!r.ok){process.stderr.write("HTTP "+r.status);process.exit(1)}process.stdout.write(Buffer.from(await r.arrayBuffer()))}).catch(e=>{process.stderr.write(String(e&&e.message||e));process.exit(1)})';
    const r = (0, node_child_process_1.spawnSync)(process.execPath, ["-e", script], {
        env: { ...process.env, ARIA_FFI_URL: url },
        encoding: "buffer",
        maxBuffer: 256 * 1024 * 1024,
    });
    if (r.status !== 0) {
        throw new Error(`download failed: ${(r.stderr && r.stderr.toString()) || "exit " + r.status}`);
    }
    return r.stdout;
}
async function httpGetBytes(url) {
    const resp = await fetch(url, { headers: { "User-Agent": SDK_UA }, redirect: "follow" });
    if (!resp.ok)
        throw new Error(`HTTP ${resp.status} ${url}`);
    return Buffer.from(await resp.arrayBuffer());
}
function installFfiFromReleases(raw, archiveBytes) {
    let releases;
    try {
        releases = JSON.parse(raw.toString("utf8"));
    }
    catch {
        throw new Error("invalid releases JSON");
    }
    if (!Array.isArray(releases))
        throw new Error("unexpected releases payload");
    const ver = selectLatestStable(releases);
    const assetName = `libaria_ffi_${ver}_${ffiAssetOs()}.tar.gz`;
    let url;
    for (const rel of releases) {
        const tag = String(rel.tag_name || rel.tag || "");
        if (stripV(tag) !== ver)
            continue;
        const assets = rel.assets || [];
        for (const asset of assets) {
            if (asset.name === assetName) {
                url = asset.browser_download_url || asset.direct_asset_url;
                break;
            }
        }
        if (url)
            break;
    }
    if (!url)
        throw new Error(`release asset not found: ${assetName}`);
    const staging = path.join(ariaHome(), "tmp", `ffi-${ver}`);
    fs.rmSync(staging, { recursive: true, force: true });
    fs.mkdirSync(staging, { recursive: true });
    const archive = path.join(staging, assetName);
    try {
        fs.writeFileSync(archive, archiveBytes(url));
        return extractFfiArchive(archive, libDir(), ffiLibName());
    }
    finally {
        fs.rmSync(staging, { recursive: true, force: true });
    }
}
/** Return a path to libaria_ffi, downloading the latest stable Release if needed. */
async function ensureFfiLib(site) {
    const env = process.env.ARIA_FFI_LIB;
    if (env && fs.existsSync(env))
        return env;
    const bundled = bundledFfiPath();
    if (bundled)
        return bundled;
    const cached = cachedFfiPath();
    if (cached)
        return cached;
    const org = upgradeOrg(site);
    const raw = await httpGetBytes(releasesApiUrl(org));
    return installFfiFromReleases(raw, (assetUrl) => httpGetBytesSync(assetUrl));
}
function ensureFfiLibSync(site) {
    const env = process.env.ARIA_FFI_LIB;
    if (env && fs.existsSync(env))
        return env;
    const bundled = bundledFfiPath();
    if (bundled)
        return bundled;
    const cached = cachedFfiPath();
    if (cached)
        return cached;
    const org = upgradeOrg(site);
    const raw = httpGetBytesSync(releasesApiUrl(org));
    return installFfiFromReleases(raw, httpGetBytesSync);
}
function loadLib(ffiLib, site) {
    const libPath = ffiLib ||
        process.env.ARIA_FFI_LIB ||
        bundledFfiPath() ||
        cachedFfiPath() ||
        ensureFfiLibSync(site);
    if (!libPath)
        throw new Error("Cannot locate libaria_ffi");
    const koffiNS = require("koffi");
    return koffiNS.load(libPath);
}
function isLocalRef(modelRef) {
    return modelRef.includes("/") || modelRef.includes("\\") || fs.existsSync(modelRef);
}
class Engine {
    /** Empty construct, or a local bundle directory. */
    constructor(bundle, opts = {}) {
        this.lib = null;
        this.handle = null;
        this.cfg = defaultAuthConfig();
        this.opts = {};
        this.opts = opts;
        if (opts.site || opts.hfToken || opts.modelscopeApiToken) {
            this.auth({
                site_url: opts.site,
                hf_token: opts.hfToken,
                modelscope_api_token: opts.modelscopeApiToken,
            });
        }
        if (bundle)
            this.bindAndInit(bundle, opts.ffiLib);
    }
    /** Set Config / Run fields on this instance only. Does not write config.yml. */
    auth(updates) {
        this.cfg = applyAuth(this.cfg, updates);
        return this;
    }
    authStatus() {
        return { ...this.cfg };
    }
    /** Reset instance defaults. Does not delete ~/.ariacompute/config.yml. */
    authClear() {
        this.cfg = defaultAuthConfig();
        return this;
    }
    bindAndInit(bundle, ffiLib) {
        this.lib = loadLib(ffiLib || this.opts.ffiLib, this.cfg.site_url || this.opts.site);
        this.fnInit = this.lib.func("aria_model_init", "void*", ["str"]);
        this.fnDestroy = this.lib.func("aria_model_destroy", "void", ["void*"]);
        this.fnComplete = this.lib.func("aria_complete", "int", ["void*", "str", "str", "str", "void*", "size_t"]);
        this.fnEmbed = this.lib.func("aria_embed", "int", ["void*", "str", "void*", "size_t"]);
        this.fnTranscribe = this.lib.func("aria_transcribe", "int", ["void*", "void*", "size_t", "str", "void*", "size_t"]);
        this.fnLastError = this.lib.func("aria_last_error", "str", []);
        this.handle = this.fnInit(bundle);
        if (!this.handle) {
            const err = this.fnLastError();
            throw new Error(err || "init failed");
        }
    }
    /** Download (if needed) and load a model using instance auth. */
    async open(modelRef) {
        await ensureFfiLib(this.cfg.site_url || this.opts.site);
        const openOpts = {
            ...this.opts,
            site: this.cfg.site_url || this.opts.site,
            hfToken: this.cfg.hf_token || this.opts.hfToken,
            modelscopeApiToken: this.cfg.modelscope_api_token || this.opts.modelscopeApiToken,
        };
        const bundle = isLocalRef(modelRef) ? modelRef : await downloadModel(modelRef, openOpts);
        if (this.handle)
            this.close();
        this.bindAndInit(bundle, this.opts.ffiLib);
        return this;
    }
    /** Auto-detect: a value containing a separator or already on disk is a local
     * path; otherwise it is a model name downloaded from the regional public hub. */
    static async open(modelRef, opts = {}) {
        const eng = new Engine(undefined, opts);
        await eng.open(modelRef);
        return eng;
    }
    complete(messages, opts = {}) {
        const messagesJson = JSON.stringify(messages);
        const optionsJson = JSON.stringify(opts || { max_tokens: 16 });
        const toolsJson = JSON.stringify([]);
        const buf = Buffer.alloc(1 << 16);
        const rc = this.fnComplete(this.handle, messagesJson, optionsJson, toolsJson, buf, buf.length);
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
        const rc = this.fnEmbed(this.handle, JSON.stringify({ input: text }), buf, buf.length);
        if (rc !== 0) {
            const err = this.fnLastError();
            throw new Error(err || "embed failed");
        }
        return JSON.parse(buf.toString("utf8").replace(/\0+$/, ""));
    }
    transcribe(pcm) {
        const buf = Buffer.alloc(1 << 16);
        const rc = this.fnTranscribe(this.handle, Buffer.from(pcm), pcm.length, null, buf, buf.length);
        if (rc !== 0) {
            const err = this.fnLastError();
            throw new Error(err || "transcribe failed");
        }
        return buf.toString("utf8").replace(/\0+$/, "");
    }
    close() {
        if (this.handle && this.fnDestroy)
            this.fnDestroy(this.handle);
        this.handle = null;
    }
}
exports.Engine = Engine;
exports._internal = { encoder };
