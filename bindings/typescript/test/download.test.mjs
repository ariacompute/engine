import test from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const { parseBundleName, downloadModel, isLocalRef } = await import(
  "../src/index.ts"
).catch(async () => {
  return await import("../dist/index.js");
});

function makeValidBundle(dir) {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "weight.bin"), "x");
  fs.writeFileSync(
    path.join(dir, "config.json"),
    JSON.stringify({ format: "aria-quant-bundle" }),
  );
}

test("parseBundleName variants", () => {
  assert.deepEqual(parseBundleName("gemma-4-e2b-it_q4"), { slug: "gemma-4-e2b-it", quant: "int4" });
  assert.deepEqual(parseBundleName("foo_q8"), { slug: "foo", quant: "int8" });
  assert.deepEqual(parseBundleName("foo_q326"), { slug: "foo", quant: "int326" });
  assert.deepEqual(parseBundleName("foo_q3.26"), { slug: "foo", quant: "int326" });
  assert.deepEqual(parseBundleName("foo"), { slug: "foo", quant: "int4" });
  assert.throws(() => parseBundleName("foo/bar"));
  assert.throws(() => parseBundleName("foo_q9"));
});

test("isLocalRef detection", () => {
  assert.equal(isLocalRef("/abs/path"), true);
  assert.equal(isLocalRef("C:\\win"), true);
  assert.equal(isLocalRef("model_name"), false);
});

test("downloadModel missing token throws", async () => {
  await assert.rejects(() => downloadModel("foo_q4", ""), /token is required/);
});

test("downloadModel cached bundle skips network", async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "aria-ts-"));
  const prev = process.env.ARIA_COMPUTE_HOME;
  process.env.ARIA_COMPUTE_HOME = home;
  try {
    const cache = path.join(home, "models", "foo_q4");
    makeValidBundle(cache);
    const result = await downloadModel("foo_q4", "tok");
    assert.equal(result, cache);
  } finally {
    process.env.ARIA_COMPUTE_HOME = prev;
  }
});

test("downloadModel invalid meta throws", async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "aria-ts-"));
  const prev = process.env.ARIA_COMPUTE_HOME;
  process.env.ARIA_COMPUTE_HOME = home;
  try {
    await assert.rejects(
      () => downloadModel("foo_q4", "tok", "http://127.0.0.1:9"),
      /dashboard request failed|fetch failed/,
    );
  } finally {
    process.env.ARIA_COMPUTE_HOME = prev;
  }
});
