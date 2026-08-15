"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Engine = void 0;
const koffi_1 = __importDefault(require("koffi"));
function loadLib() {
    const path = process.env.ARIA_FFI_LIB;
    if (!path)
        throw new Error("Set ARIA_FFI_LIB");
    const lib = koffi_1.default.load(path);
    return {
        init: lib.func("aria_model_init", "void *", ["str"]),
        destroy: lib.func("aria_model_destroy", "void", ["void *"]),
        complete: lib.func("aria_complete", "int", [
            "void *",
            "str",
            "str",
            "str",
            "uint8 *",
            "size_t",
        ]),
        embed: lib.func("aria_embed", "int", ["void *", "str", "uint8 *", "size_t"]),
        transcribe: lib.func("aria_transcribe", "int", [
            "void *",
            "uint8 *",
            "size_t",
            "str",
            "uint8 *",
            "size_t",
        ]),
        lastError: lib.func("aria_last_error", "str", []),
    };
}
function parseOut(buf) {
    const text = buf.toString("utf8").replace(/\0.*$/s, "");
    return JSON.parse(text);
}
class Engine {
    constructor(bundlePath) {
        this.lib = loadLib();
        this.handle = this.lib.init(bundlePath);
        if (!this.handle)
            throw new Error(this.lib.lastError() || "init failed");
    }
    close() {
        if (this.handle) {
            this.lib.destroy(this.handle);
            this.handle = null;
        }
    }
    complete(messages, options = { max_tokens: 16 }, tools = []) {
        const out = Buffer.alloc(256 * 1024);
        const rc = this.lib.complete(this.handle, JSON.stringify(messages), JSON.stringify(options), JSON.stringify(tools), out, out.length);
        if (rc !== 0)
            throw new Error(this.lib.lastError() || "complete failed");
        return parseOut(out);
    }
    embed(input) {
        const out = Buffer.alloc(256 * 1024);
        const rc = this.lib.embed(this.handle, JSON.stringify({ input }), out, out.length);
        if (rc !== 0)
            throw new Error(this.lib.lastError() || "embed failed");
        return parseOut(out);
    }
    transcribe(pcm) {
        const out = Buffer.alloc(64 * 1024);
        const rc = this.lib.transcribe(this.handle, pcm, pcm.length, null, out, out.length);
        if (rc !== 0)
            throw new Error(this.lib.lastError() || "transcribe failed");
        return parseOut(out);
    }
}
exports.Engine = Engine;
