fn main() {
    println!("cargo:rerun-if-env-changed=ARIA_ENGINE_VERSION");
    let raw = std::env::var("ARIA_ENGINE_VERSION").unwrap_or_else(|_| {
        std::env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION set by cargo")
    });
    // Release tags are often `v1.2.3`; embed without the leading `v`.
    let version = raw.strip_prefix('v').unwrap_or(raw.as_str());
    println!("cargo:rustc-env=ARIA_ENGINE_VERSION={version}");
}
