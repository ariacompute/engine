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
disk is loaded directly; otherwise the SDK downloads it from the Dashboard
private source (requires `token`; `site` defaults to `https://ariacompute.com`)
into `~/.ariacompute/models/{model}` and then loads it. A valid cached bundle is
reused without re-downloading.

```ts
const eng = await Engine.open("gemma-4-e2b-it_q4", { token: "API_TOKEN" });
```
