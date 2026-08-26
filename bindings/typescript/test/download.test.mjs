import test from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

function loadDownload() {
  try {
    return require("../dist/index.js");
  } catch {
    return null;
  }
}

function makeValidBundle(dir) {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "weight.bin"), "x");
  fs.writeFileSync(
    path.join(dir, "config.json"),
    JSON.stringify({ format: "aria-quant-bundle" }),
  );
}

test("parseBundleName variants", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { parseBundleName } = mod;
  assert.deepEqual(parseBundleName("gemma-4-e2b-it_q4"), { slug: "gemma-4-e2b-it", quant: "int4" });
  assert.deepEqual(parseBundleName("foo_q8"), { slug: "foo", quant: "int8" });
  assert.deepEqual(parseBundleName("foo_q326"), { slug: "foo", quant: "int326" });
  assert.deepEqual(parseBundleName("foo_q326_channel"), { slug: "foo", quant: "int326" });
  assert.deepEqual(parseBundleName("foo_q3.26"), { slug: "foo", quant: "int326" });
  assert.deepEqual(parseBundleName("foo"), { slug: "foo", quant: "int4" });
  assert.throws(() => parseBundleName("foo/bar"));
  assert.throws(() => parseBundleName("foo_q9"));
});

test("isLocalRef detection", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { isLocalRef } = mod;
  assert.equal(isLocalRef("/abs/path"), true);
  assert.equal(isLocalRef("C:\\win"), true);
  assert.equal(isLocalRef("model_name"), false);
});

test("preferred hub follows site TLD", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { preferredPublicHub } = mod;
  assert.equal(preferredPublicHub("https://ariacompute.com"), "huggingface");
  assert.equal(preferredPublicHub("https://ariacompute.cn"), "modelscope");
  assert.equal(preferredPublicHub(), "huggingface");
});

test("dashboard token is not sent to hub", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { hubBearer } = mod;
  assert.equal(hubBearer("sk-bf-95076ed1-8c1a-4efa-b33c-f52c1d7f9f24"), undefined);
  assert.equal(hubBearer("bfvk-test"), undefined);
  assert.equal(hubBearer("hf_abc"), "hf_abc");
});

test("hub URLs follow upload layout", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { hubFileUrls } = mod;
  const hf = hubFileUrls("huggingface", "gemma-4-e2b-it_q4", "config.json");
  assert.ok(
    hf.some((u) =>
      u.includes("/ariacompute/gemma-4-e2b-it_q4/resolve/main/v1.0/gemma-4-e2b-it_q4/config.json"),
    ),
  );
  const ms = hubFileUrls("modelscope", "gemma-4-e2b-it_q4", "weight.bin");
  assert.ok(ms.some((u) => u.includes("/v1.0/gemma-4-e2b-it_q4/weight.bin")));
  assert.ok([...hf, ...ms].every((u) => !u.includes("/api/dashboard/")));
});

test("downloadModel cached bundle skips network", async () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { downloadModel } = mod;
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "aria-ts-"));
  const prev = process.env.ARIA_COMPUTE_HOME;
  process.env.ARIA_COMPUTE_HOME = home;
  try {
    const cache = path.join(home, "models", "foo_q4");
    makeValidBundle(cache);
    const result = await downloadModel("foo_q4", "tok");
    assert.equal(result, cache);
    const noToken = await downloadModel("foo_q4");
    assert.equal(noToken, cache);
  } finally {
    process.env.ARIA_COMPUTE_HOME = prev;
  }
});
