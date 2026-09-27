import * as fs from "fs";
import * as os from "os";
import * as path from "path";

export type Track = "encoder" | "decoder";

function ariaHome(): string {
  return process.env.ARIA_COMPUTE_HOME || path.join(os.homedir(), ".ariacompute");
}

function ffiCandidates(): string[] {
  const env = process.env.ARIA_FFI_LIB;
  if (env) return [env];
  const lib = path.join(ariaHome(), "lib");
  if (process.platform === "win32") return [path.join(lib, "aria-engine_ffi.dll")];
  if (process.platform === "darwin") return [path.join(lib, "libaria-engine_ffi.dylib")];
  return [path.join(lib, "libaria-engine_ffi.so")];
}

/** Load koffi binding when native lib is present; otherwise throw with install hint. */
export function loadFfi(): any {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const koffi = require("koffi");
  const candidates = ffiCandidates();
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) {
    throw new Error(
      `libaria-engine_ffi not found; run aria-engine upgrade or set ARIA_FFI_LIB (tried ${candidates.join(", ")})`
    );
  }
  const lib = koffi.load(found);
  return {
    path: found,
    aria_last_error: lib.func("aria_last_error", "str", []),
    aria_model_init: lib.func("aria_model_init", "void *", ["str", "str"]),
    aria_model_destroy: lib.func("aria_model_destroy", "void", ["void *"]),
    aria_systemone: lib.func("aria_systemone", "int", ["void *", "str", "char *", "size_t"]),
  };
}

export class AriaEngine {
  private ffi: any;
  private handle: any;

  constructor(checkpoint: string, track: Track = "encoder") {
    this.ffi = loadFfi();
    this.handle = this.ffi.aria_model_init(checkpoint, track);
    if (!this.handle) {
      throw new Error(this.ffi.aria_last_error() || "aria_model_init failed");
    }
  }

  systemone(request: object): object {
    const buf = Buffer.alloc(1 << 20);
    const rc = this.ffi.aria_systemone(this.handle, JSON.stringify(request), buf, buf.length);
    if (rc !== 0) {
      throw new Error(this.ffi.aria_last_error() || `aria_systemone rc=${rc}`);
    }
    const text = buf.toString("utf8").replace(/\0.*$/, "");
    return JSON.parse(text);
  }

  destroy(): void {
    if (this.handle) {
      this.ffi.aria_model_destroy(this.handle);
      this.handle = null;
    }
  }
}

export default AriaEngine;
