# @ariacompute/engine-ts

```bash
npm i @ariacompute/engine-ts
```

`libaria_ffi` is resolved as `ARIA_FFI_LIB` → package-bundled `dist/lib/` → `~/.ariacompute/lib/` (same as `aria-engine upgrade`). If none exist, `Engine.open` downloads the latest stable `libaria_ffi_{ver}_{os}.tar.gz` from regional Releases. `ARIA_FFI_LIB` remains an optional override.

```ts
import { Engine } from "@ariacompute/engine-ts";
const eng = new Engine("/path/to/bundle");
console.log(eng.complete([{ role: "user", content: "hi" }]));
eng.close();
```

## Auto-download by model name

`Engine.open(modelRef, { token, site })` accepts a local bundle path **or** an
Aria model name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or already on
disk is loaded directly; otherwise the SDK downloads it from the regional public
hub (same as `aria-engine download`: `.com` → Hugging Face, `.cn` → ModelScope;
`site` defaults to `https://ariacompute.com`) into `~/.ariacompute/models/{model}`
and then loads it. Dashboard is not used. A Dashboard `sk-` / `bfvk-` token is
ignored for hub auth. Token is optional for public models. A valid cached bundle
is reused without re-downloading.

Gated files: call `eng.auth({ hf_token })` (`.com`) or `eng.auth({ modelscope_api_token, site_url })` (`.cn`) **before** `open`. Same keys as `aria-engine auth`. Instance-only; does not write `config.yml`. If omitted, the SDK reads `~/.ariacompute/config.yml`.

```ts
const eng = await Engine.open("gemma-4-e2b-it_q4");

const gated = new Engine();
gated.auth({ hf_token: "hf_..." });
await gated.open("gemma-4-e2b-it_q4");
```
