export interface GenerateOptions {
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stream?: boolean;
    stop?: string[];
    response_format?: {
        type: "text" | "json_object" | "json_schema";
        json_schema?: unknown;
    };
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
export declare const INTL_CLOUD = "https://gateway.ariacompute.com";
export declare const INTL_SITE = "https://ariacompute.com";
export declare const INTL_UPGRADE = "https://github.com/ariacompute";
export declare const CN_CLOUD = "https://gateway.ariacompute.cn";
export declare const CN_SITE = "https://ariacompute.cn";
export declare const CN_UPGRADE = "https://gitee.com/ariacompute";
export interface AuthConfig {
    cloud_api_key: string;
    cloud_url: string;
    site_url: string;
    upgrade_url: string;
    hybrid_mode: string;
    hybrid_execution: string;
    hybrid_semantic: boolean;
    hybrid_semantic_timeout_ms: number;
    hybrid_semantic_cache_size: number;
    compute: string;
    hf_token: string;
    modelscope_api_token: string;
}
export declare function defaultAuthConfig(): AuthConfig;
export declare function fillAuthUrls(cfg: AuthConfig): AuthConfig;
/** Replace `authHooks.probeDashboard` in tests to avoid a real Dashboard probe. */
export declare const authHooks: {
    probeDashboard: (siteUrl: string, apiKey: string) => boolean;
};
export declare function detectGatewayPair(apiKey: string): [string, string, string];
export declare function applyAuth(existing: AuthConfig, updates: Partial<AuthConfig>): AuthConfig;
/** Parse `slug`/`quant` from a model name such as `gemma-4-e2b-it_q4`. */
export declare function parseBundleName(model: string): {
    slug: string;
    quant: string;
};
export declare function preferredPublicHub(site?: string): "huggingface" | "modelscope";
export declare function hubBearer(token?: string): string | undefined;
export declare function configYmlScalar(key: string): string | undefined;
export declare function resolveHubToken(source: "huggingface" | "modelscope", opts?: Pick<OpenOptions, "token" | "hfToken" | "modelscopeApiToken">): string | undefined;
export declare function hubFileUrls(source: "huggingface" | "modelscope", model: string, file: string, sdk?: string): string[];
/** Download `model` from the regional public hub into
 * `~/.ariacompute/models/{model}` and return that directory.
 * Matches aria-engine download: .com → Hugging Face, .cn → ModelScope.
 * Dashboard is not used. Skips the download when a valid bundle is already cached. */
export declare function downloadModel(model: string, tokenOrOpts?: string | OpenOptions, site?: string): Promise<string>;
export declare function ffiLibName(platform?: string): string;
export declare function cachedFfiPath(platform?: string): string | undefined;
export declare function ffiAssetOs(platform?: string, arch?: string): string;
export declare function selectLatestStable(releases: Array<Record<string, unknown>>): string;
export declare function extractFfiArchive(archive: string, destDir: string, want?: string): string;
/** Return a path to libaria_ffi, downloading the latest stable Release if needed. */
export declare function ensureFfiLib(site?: string): Promise<string>;
export declare function isLocalRef(modelRef: string): boolean;
export declare class Engine {
    private lib;
    private handle;
    private fnInit;
    private fnDestroy;
    private fnComplete;
    private fnEmbed;
    private fnTranscribe;
    private fnLastError;
    private cfg;
    private opts;
    /** Empty construct, or a local bundle directory. */
    constructor(bundle?: string, opts?: OpenOptions);
    /** Set Config / Run fields on this instance only. Does not write config.yml. */
    auth(updates: Partial<AuthConfig>): this;
    authStatus(): AuthConfig;
    /** Reset instance defaults. Does not delete ~/.ariacompute/config.yml. */
    authClear(): this;
    private bindAndInit;
    /** Download (if needed) and load a model using instance auth. */
    open(modelRef: string): Promise<this>;
    /** Auto-detect: a value containing a separator or already on disk is a local
     * path; otherwise it is a model name downloaded from the regional public hub. */
    static open(modelRef: string, opts?: OpenOptions): Promise<Engine>;
    complete(messages: Turn[], opts?: GenerateOptions): CompleteResult;
    embed(text: string): number[];
    transcribe(pcm: Uint8Array): string;
    close(): void;
}
export declare const _internal: {
    encoder: TextEncoder;
};
