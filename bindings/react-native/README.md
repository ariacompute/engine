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
on disk is loaded directly; otherwise the SDK downloads it from the Dashboard
private source (requires `token`; `site` defaults to `https://ariacompute.com`)
into `~/.ariacompute/models/{model}` and then loads it. A valid cached bundle is
reused without re-downloading. The zip streaming + extraction is delegated to the
native `AriaEngineModule` bridge.

```js
const eng = await AriaEngine.open("gemma-4-e2b-it_q4", { token: "API_TOKEN" });
```
