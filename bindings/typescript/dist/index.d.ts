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
    /** Optional Hugging Face / ModelScope hub token. Dashboard sk-/bfvk- keys are ignored. */
    token?: string;
    /** Site used to pick the regional hub. Defaults to https://ariacompute.com (.com → HF, .cn → ModelScope). */
    site?: string;
    /** Explicit path to the FFI library. */
    ffiLib?: string;
}
/** Parse `slug`/`quant` from a model name such as `gemma-4-e2b-it_q4`. */
export declare function parseBundleName(model: string): {
    slug: string;
    quant: string;
};
export declare function preferredPublicHub(site?: string): "huggingface" | "modelscope";
export declare function hubBearer(token?: string): string | undefined;
export declare function hubFileUrls(source: "huggingface" | "modelscope", model: string, file: string, sdk?: string): string[];
/** Download `model` from the regional public hub into
 * `~/.ariacompute/models/{model}` and return that directory.
 * Matches aria-engine download: .com → Hugging Face, .cn → ModelScope.
 * Dashboard is not used. Skips the download when a valid bundle is already cached. */
export declare function downloadModel(model: string, token?: string, site?: string): Promise<string>;
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
    /** Construct from a local bundle directory. */
    constructor(bundle: string, opts?: OpenOptions);
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
