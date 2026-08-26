# @ariacompute/engine-rn

```js
import { AriaEngine } from '@ariacompute/engine-rn';
const eng = new AriaEngine('/path/to/bundle');
console.log(await eng.complete([{ role: 'user', content: 'hi' }]));
```

npm publish name: `@ariacompute/engine-rn`. Device-farm CI: `.github/workflows/bindings-mobile.yml`.

`AriaEngine.open` installs `libaria_ffi` into `~/.ariacompute/lib/` when it is not already on `ARIA_FFI_LIB` or in that cache (same Releases asset as `aria-engine upgrade`).

## Auto-download by model name

`AriaEngine.open(modelRef, { token, site })` accepts a local bundle path **or**
an Aria model name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or already
on disk is loaded directly; otherwise the SDK downloads it from the regional
public hub (same as `aria-engine download`: `.com` → Hugging Face, `.cn` →
ModelScope; `site` defaults to `https://ariacompute.com`) into
`~/.ariacompute/models/{model}` and then loads it. Dashboard is not used. A
Dashboard `sk-` / `bfvk-` token is ignored for hub auth. Token is optional for
public models. Gated files: pass `hfToken` / `modelscopeApiToken` (same as
`aria-engine auth`); if omitted, reads `~/.ariacompute/config.yml`. A valid
cached bundle is reused without re-downloading.

```js
const eng = await AriaEngine.open("gemma-4-e2b-it_q4");
const gated = await AriaEngine.open("gemma-4-e2b-it_q4", { hfToken: "hf_..." });
```
