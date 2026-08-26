# @ariacompute/engine-ts

```bash
export ARIA_FFI_LIB=/path/to/libaria_ffi.so
npm i @ariacompute/engine-ts
```

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

```ts
const eng = await Engine.open("gemma-4-e2b-it_q4");
```
