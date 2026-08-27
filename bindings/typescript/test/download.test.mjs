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

test("resolveHubToken uses named fields and config.yml", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { resolveHubToken } = mod;
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "aria-ts-"));
  const prev = process.env.ARIA_COMPUTE_HOME;
  process.env.ARIA_COMPUTE_HOME = home;
  try {
    fs.writeFileSync(
      path.join(home, "config.yml"),
      'hf_token: hf_from_yml\nmodelscope_api_token: "ms_from_yml"\n',
    );
    assert.equal(
      resolveHubToken("huggingface", { hfToken: "hf_named", token: "hf_generic" }),
      "hf_named",
    );
    assert.equal(
      resolveHubToken("modelscope", { modelscopeApiToken: "ms_named" }),
      "ms_named",
    );
    assert.equal(resolveHubToken("huggingface", {}), "hf_from_yml");
    assert.equal(resolveHubToken("modelscope", {}), "ms_from_yml");
    assert.equal(
      resolveHubToken("huggingface", { token: "sk-bf-not-hub" }),
      "hf_from_yml",
    );
  } finally {
    process.env.ARIA_COMPUTE_HOME = prev;
  }
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

test("ffiAssetOs matches upgrade suffixes", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { ffiAssetOs } = mod;
  assert.equal(ffiAssetOs("linux", "x64"), "linux_x86_64");
  assert.equal(ffiAssetOs("linux", "arm64"), "linux_arm64");
  assert.equal(ffiAssetOs("darwin", "arm64"), "macos");
  assert.equal(ffiAssetOs("win32", "x64"), "windows_x86_64");
  assert.throws(() => ffiAssetOs("linux", "ppc64"));
});

test("selectLatestStable skips draft and prerelease", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { selectLatestStable } = mod;
  assert.equal(
    selectLatestStable([
      { tag_name: "v0.7.1", draft: false, prerelease: false },
      { tag_name: "v0.8.0-rc1", draft: false, prerelease: true },
      { tag_name: "v0.7.2", draft: false, prerelease: false },
      { tag_name: "v0.9.0", draft: true, prerelease: false },
    ]),
    "0.7.2",
  );
});

test("extractFfiArchive and cached ensureFfiLib skip network", async () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { extractFfiArchive, ensureFfiLib, ffiLibName } = mod;
  if (process.platform !== "linux") {
    test.skip("linux .so fixture");
    return;
  }
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "aria-ffi-"));
  const prevHome = process.env.ARIA_COMPUTE_HOME;
  const prevLib = process.env.ARIA_FFI_LIB;
  process.env.ARIA_COMPUTE_HOME = home;
  delete process.env.ARIA_FFI_LIB;
  try {
    const src = fs.mkdtempSync(path.join(os.tmpdir(), "aria-ffi-src-"));
    const want = ffiLibName();
    fs.writeFileSync(path.join(src, want), "dummy-ffi");
    const archive = path.join(home, "lib.tar.gz");
    const { spawnSync } = await import("node:child_process");
    const r = spawnSync("tar", ["-czf", archive, "-C", src, want], { encoding: "utf8" });
    if (r.status !== 0) {
      test.skip("tar not available");
      return;
    }
    const destDir = path.join(home, "lib");
    const got = extractFfiArchive(archive, destDir, want);
    assert.equal(path.basename(got), want);
    assert.equal(fs.readFileSync(got, "utf8"), "dummy-ffi");
    const cached = await ensureFfiLib();
    assert.equal(cached, got);
  } finally {
    process.env.ARIA_COMPUTE_HOME = prevHome;
    if (prevLib === undefined) delete process.env.ARIA_FFI_LIB;
    else process.env.ARIA_FFI_LIB = prevLib;
  }
});

test("auth instance all fields roundtrip", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { Engine, CN_SITE, CN_CLOUD, CN_UPGRADE } = mod;
  const eng = new Engine();
  eng.auth({
    cloud_api_key: "sk-test",
    cloud_url: CN_CLOUD,
    site_url: CN_SITE,
    upgrade_url: CN_UPGRADE,
    hybrid_mode: "cost",
    hybrid_execution: "device",
    hybrid_semantic: false,
    hybrid_semantic_timeout_ms: 250,
    hybrid_semantic_cache_size: 16,
    compute: "cpu",
    hf_token: "hf_abc",
    modelscope_api_token: "ms_xyz",
  });
  const st = eng.authStatus();
  assert.equal(st.cloud_api_key, "sk-test");
  assert.equal(st.hybrid_mode, "cost");
  assert.equal(st.hybrid_execution, "device");
  assert.equal(st.hybrid_semantic, false);
  assert.equal(st.hybrid_semantic_timeout_ms, 250);
  assert.equal(st.hybrid_semantic_cache_size, 16);
  assert.equal(st.compute, "cpu");
  assert.equal(st.hf_token, "hf_abc");
  assert.equal(st.modelscope_api_token, "ms_xyz");
  assert.equal(st.site_url, CN_SITE);
});

test("auth instance partial merge", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { Engine } = mod;
  const eng = new Engine();
  eng.auth({ hf_token: "hf_one", hybrid_mode: "intelligence" });
  eng.auth({ compute: "cuda" });
  const st = eng.authStatus();
  assert.equal(st.hf_token, "hf_one");
  assert.equal(st.hybrid_mode, "intelligence");
  assert.equal(st.compute, "cuda");
});

test("auth invalid enum leaves state", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { Engine } = mod;
  const eng = new Engine();
  eng.auth({ hybrid_mode: "cost" });
  assert.throws(() => eng.auth({ hybrid_mode: "nope" }));
  assert.equal(eng.authStatus().hybrid_mode, "cost");
});

test("auth clear resets instance", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { Engine } = mod;
  const eng = new Engine();
  eng.auth({ hf_token: "hf_x", hybrid_mode: "cost" });
  eng.authClear();
  const st = eng.authStatus();
  assert.equal(st.hf_token, "");
  assert.equal(st.hybrid_mode, "balance");
});

test("auth fills urls from site tld", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { Engine, CN_CLOUD, CN_UPGRADE } = mod;
  const eng = new Engine();
  eng.auth({ site_url: "https://ariacompute.cn" });
  const st = eng.authStatus();
  assert.equal(st.cloud_url, CN_CLOUD);
  assert.equal(st.upgrade_url, CN_UPGRADE);
});

test("auth does not write config.yml", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const { Engine } = mod;
  const prev = process.env.ARIA_COMPUTE_HOME;
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "aria-auth-"));
  process.env.ARIA_COMPUTE_HOME = home;
  try {
    const eng = new Engine();
    eng.auth({
      cloud_api_key: "sk-test",
      site_url: "https://ariacompute.com",
      hf_token: "hf_x",
    });
    assert.equal(fs.existsSync(path.join(home, "config.yml")), false);
  } finally {
    if (prev === undefined) delete process.env.ARIA_COMPUTE_HOME;
    else process.env.ARIA_COMPUTE_HOME = prev;
  }
});

test("auth detect urls from key mocked", () => {
  const mod = loadDownload();
  if (!mod) {
    test.skip("build typescript first");
    return;
  }
  const prev = mod.authHooks.probeDashboard;
  mod.authHooks.probeDashboard = (site) => String(site).includes("ariacompute.cn");
  try {
    const eng = new mod.Engine();
    eng.auth({ cloud_api_key: "sk-region" });
    const st = eng.authStatus();
    assert.equal(st.site_url, mod.CN_SITE);
    assert.equal(st.cloud_url, mod.CN_CLOUD);
    assert.equal(st.upgrade_url, mod.CN_UPGRADE);
  } finally {
    mod.authHooks.probeDashboard = prev;
  }
});
