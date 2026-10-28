# AGENTS.md — Aria Engine (AFM-D)

## Product

Greenfield **AFM-D typed-decision** runtime (Encoder + Decoder). Do **not** revive Aria chat graph/kernel/session from `deprecated/engine`. Reuse only product shell patterns: CLI/`upgrade`, FFI packaging, `release.yml`, `bindings/*`, publish workflows.

## Workspace crates

- `ariacompute-core` — System One types, packing, config (`~/.ariacompute`), gateway
- `ariacompute-de` — Encoder checkpoint + packing + score/logits
- `ariacompute-dd` — SemIf row + decoder map + scorer shell
- `aria-cli` — bin `aria-engine`（setup / download / upgrade / serve / decide）
- `ariacompute-ffi` — cdylib `aria_ffi` → release name `libaria-engine_ffi`
- `bindings/rust` — crates.io `ariacompute-engine`

## Trust stores (eMudhra/emSign CA key compromise, 2026-10)

- The `aria-engine` / `aria-cli` binaries do outbound TLS for hub downloads via
  `reqwest` and `hf-hub` (both `rustls-tls`), which use **`webpki-roots`** — pinned
  `>=1.0.10` in `cli/Cargo.toml` (removes the compromised emSign roots). The Docker
  build already installs `ca-certificates` (system store), but the binaries embed
  webpki-roots, so the lock bump is the decisive fix.
- The language bindings (Go / Python / TS / RN) are **FFI-only** (cgo / ctypes / koffi)
  and perform no outbound TLS themselves, so their runtimes' trust stores are out of
  scope; the actual network path is the Rust side above.

## Contracts (do not drift)

- Encoder budgets: `max_len=1024`, `head_max_len=512`, `OPTION_DESC_MAX=96`
- Decoder: MiniCPM5-2B @ `12a3808a956f869c767195e9266b59c4d21d92e2`, options 2–16, port **8011**
- System One: `POST /v1/systemone` body `{ state, questions }` → `{ answers, model }`
- Upgrade assets must match `release.yml` naming
- `engine.yml` 五字段：`site_url` / `upgrade_url` / `compute` / `hf_token` / `modelscope_api_token`（**无** router）

## Commands

```bash
cargo test --workspace
./scripts/prepare-binding-fixture.sh
./scripts/run-binding-tests.sh
```

## Out of scope

Training / RLCD in Rust; replacing Python `model/afm-d` train/eval/bench.
