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
