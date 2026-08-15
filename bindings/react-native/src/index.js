/**
 * React Native surface — native module bridges to libaria_ffi (Turbo Module / JSI).
 * JS API mirrors @ariacompute/engine-ts for docs samples.
 */
export class AriaEngine {
  constructor(bundlePath) {
    this.bundlePath = bundlePath;
    // NativeModules.AriaEngine.init(bundlePath) when linked.
  }

  async complete(messages, options = { max_tokens: 16 }, tools = []) {
    // return NativeModules.AriaEngine.complete(...)
    return {
      success: true,
      response: "",
      function_calls: [],
      note: "Link native module to libaria_ffi",
    };
  }

  async close() {}
}
