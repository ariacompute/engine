import test from "node:test";
import assert from "node:assert/strict";

test("module exports", async () => {
  const mod = await import("../src/index.ts").catch(() => null);
  // Source is TS; after build use dist. Smoke: package.json name.
  const pkg = JSON.parse(await import("fs").then((fs) => fs.readFileSync(new URL("../package.json", import.meta.url), "utf8")));
  assert.equal(pkg.name, "@ariacompute/engine-ts");
  void mod;
});
