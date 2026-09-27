# @ariacompute/engine-ts

```ts
import { AriaEngine } from "@ariacompute/engine-ts";
const eng = new AriaEngine("/path/to/checkpoint", "encoder");
const resp = eng.systemone({ state: {}, questions: { q1: { type: "noul", instructions: "ok?", criteria: {} } } });
eng.destroy();
```

Requires `libaria-engine_ffi` via `aria-engine upgrade` or `ARIA_FFI_LIB`.
