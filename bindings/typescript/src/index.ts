import koffi from "koffi";

export type ChatMessage = { role: string; content: string };

function loadLib() {
  const path = process.env.ARIA_FFI_LIB;
  if (!path) throw new Error("Set ARIA_FFI_LIB");
  const lib = koffi.load(path);
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

function parseOut(buf: Buffer) {
  const text = buf.toString("utf8").replace(/\0.*$/s, "");
  return JSON.parse(text);
}

export class Engine {
  private lib = loadLib();
  private handle: any;

  constructor(bundlePath: string) {
    this.handle = this.lib.init(bundlePath);
    if (!this.handle) throw new Error(this.lib.lastError() || "init failed");
  }

  close() {
    if (this.handle) {
      this.lib.destroy(this.handle);
      this.handle = null;
    }
  }

  complete(
    messages: ChatMessage[],
    options: Record<string, unknown> = { max_tokens: 16 },
    tools: unknown[] = []
  ) {
    const out = Buffer.alloc(256 * 1024);
    const rc = this.lib.complete(
      this.handle,
      JSON.stringify(messages),
      JSON.stringify(options),
      JSON.stringify(tools),
      out,
      out.length
    );
    if (rc !== 0) throw new Error(this.lib.lastError() || "complete failed");
    return parseOut(out);
  }

  embed(input: string) {
    const out = Buffer.alloc(256 * 1024);
    const rc = this.lib.embed(
      this.handle,
      JSON.stringify({ input }),
      out,
      out.length
    );
    if (rc !== 0) throw new Error(this.lib.lastError() || "embed failed");
    return parseOut(out);
  }

  transcribe(pcm: Buffer) {
    const out = Buffer.alloc(64 * 1024);
    const rc = this.lib.transcribe(
      this.handle,
      pcm,
      pcm.length,
      null,
      out,
      out.length
    );
    if (rc !== 0) throw new Error(this.lib.lastError() || "transcribe failed");
    return parseOut(out);
  }
}
