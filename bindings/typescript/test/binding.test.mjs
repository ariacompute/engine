import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

test("complete_ok", { skip: !process.env.ARIA_FFI_LIB || !process.env.ARIA_BUNDLE }, async () => {
  // Prefer compiled dist; fallback skip if not built
  let Engine;
  try {
    ({ Engine } = require("../dist/index.js"));
  } catch {
    test.skip("build typescript first");
    return;
  }
  const eng = new Engine(process.env.ARIA_BUNDLE);
  try {
    const out = eng.complete([{ role: "user", content: "hi" }], { max_tokens: 2 });
    assert.equal(out.success, true);
    assert.ok(out.response);
  } finally {
    eng.close();
  }
});
