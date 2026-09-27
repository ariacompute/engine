# Aria Engine

Pure Rust inference runtime for **AFM**:

| Track | Crate | Model |
| --- | --- | --- |
| Encoder (`afm_de`) | `ariacompute-de` | Laya-layout ModernBERT DecisionModel |
| Decoder (`afm_dd`) | `ariacompute-dd` | MiniCPM5-2B ± PEFT LoRA (SemIf direct) |

Surfaces: **`aria-engine` CLI** · **`POST /v1/systemone`** · **`libaria_ffi` C ABI** · language SDKs under `bindings/`.

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo build --release -p aria-cli -p ariacompute-ffi
```

Optional CUDA feature: `cargo build -p aria-cli --features cuda` (propagates to de/dd).

## CLI

```bash
aria-engine setup                 # site_url + upgrade_url (GitHub/Gitee org)
aria-engine download afm-de       # → ~/.ariacompute/models/afm-de
aria-engine list | check | clean
aria-engine upgrade [version]     # CLI + libaria-engine_ffi from Releases
aria-engine serve --track decoder --checkpoint ~/.ariacompute/models/afm-dd
aria-engine decide --track encoder --checkpoint ./ckpt --logits '[0.1,2.0]' < record.json
aria-engine -v
```

Decoder HTTP default bind: `127.0.0.1:8011`. Encoder default: `127.0.0.1:8010`.

## FFI

Header: [`ffi/include/aria.h`](ffi/include/aria.h) — `aria_model_init`, `aria_systemone`, `aria_model_destroy`.

Release assets: `aria-engine_${VERSION}_${OS}.tar.gz|zip`, `libaria-engine_ffi_${VERSION}_${OS}.tar.gz`.

## Language SDKs

| Binding | Package |
| --- | --- |
| Rust | `ariacompute-engine` (`cargo add`) |
| TypeScript | `@ariacompute/engine-ts` |
| React Native | `@ariacompute/engine-rn` |
| Python | `aria-engine` (PyPI) |
| Flutter | `package:aria_engine` (pub.dev) |
| Kotlin | `com.ariacompute:engine` (Maven Central) |
| Swift | CocoaPods `AriaEngine` |
| Go | `bindings/go` (cgo) |

## Publish workflows

On GitHub Release: `release.yml` (CLI+FFI assets) plus `publish-cargo.yml`, `publish-npm.yml`, `publish-pub.yml`, `publish-pypi.yml`, `publish-maven.yml`, `publish-cocoapods.yml`.

## License

MIT.

## Engineering Conventions

This repository follows the Harness Engineering philosophy:

- [`AGENTS.md`](AGENTS.md): Agent engineering context entry and directory index
- [`requirements.md`](requirements.md): Requirements spec (feature boundaries/exceptions/acceptance criteria, human-review-gated)
- [`task.md`](task.md): Implementation task checklist
