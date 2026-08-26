# @ariacompute/engine-rn

```js
import { AriaEngine } from '@ariacompute/engine-rn';
const eng = new AriaEngine('/path/to/bundle');
console.log(await eng.complete([{ role: 'user', content: 'hi' }]));
```

npm publish name: `@ariacompute/engine-rn`. Device-farm CI: `.github/workflows/bindings-mobile.yml`.

## Auto-download by model name

`AriaEngine.open(modelRef, { token, site })` accepts a local bundle path **or**
an Aria model name (e.g. `gemma-4-e2b-it_q4`). A value containing `/` or already
on disk is loaded directly; otherwise the SDK downloads it from the regional
public hub (same as `aria-engine download`: `.com` → Hugging Face, `.cn` →
ModelScope; `site` defaults to `https://ariacompute.com`) into
`~/.ariacompute/models/{model}` and then loads it. Dashboard is not used. A
Dashboard `sk-` / `bfvk-` token is ignored for hub auth. Token is optional for
public models. A valid cached bundle is reused without re-downloading.

```js
const eng = await AriaEngine.open("gemma-4-e2b-it_q4");
```
