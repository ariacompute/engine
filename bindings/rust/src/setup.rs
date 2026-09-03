//! Instance-level Engine setup (in-memory; does not write engine.yml).

use thiserror::Error;

pub const INTL_SITE: &str = "https://ariacompute.com";
pub const INTL_UPGRADE: &str = "https://github.com/ariacompute";
pub const CN_SITE: &str = "https://ariacompute.cn";
pub const CN_UPGRADE: &str = "https://gitee.com/ariacompute";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SetupError {
    #[error("invalid compute: {0}")]
    InvalidCompute(String),
    #[error("{0}")]
    InvalidKey(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupConfig {
    // --- Local (router registration) ---
    pub router: String,
    pub router_api_key: String,
    // --- OAuth (Aria Compute) ---
    pub serve_site: String,
    pub serve_api_key: String,
    // --- Hub / compute ---
    pub site_url: String,
    pub upgrade_url: String,
    pub compute: String,
    pub hf_token: String,
    pub modelscope_api_token: String,
}

impl Default for SetupConfig {
    fn default() -> Self {
        Self {
            router: String::new(),
            router_api_key: String::new(),
            serve_site: String::new(),
            serve_api_key: String::new(),
            site_url: String::new(),
            upgrade_url: String::new(),
            compute: "auto".into(),
            hf_token: String::new(),
            modelscope_api_token: String::new(),
        }
    }
}

/// Partial merge. `None` fields are omitted.
#[derive(Debug, Clone, Default)]
pub struct SetupUpdates {
    // Local
    pub router: Option<String>,
    pub router_api_key: Option<String>,
    // OAuth
    pub serve_site: Option<String>,
    pub serve_api_key: Option<String>,
    // Hub / compute
    pub site_url: Option<String>,
    pub upgrade_url: Option<String>,
    pub compute: Option<String>,
    pub hf_token: Option<String>,
    pub modelscope_api_token: Option<String>,
}

fn gateway_region(url: &str) -> Option<&'static str> {
    let lower = url.to_ascii_lowercase();
    if lower.contains("ariacompute.cn") || lower.contains("gitee.com/ariacompute") {
        Some("cn")
    } else if lower.contains("ariacompute.com") || lower.contains("github.com/ariacompute") {
        Some("intl")
    } else {
        None
    }
}

fn pair_urls(region: &str) -> (&'static str, &'static str) {
    if region == "cn" {
        (CN_SITE, CN_UPGRADE)
    } else {
        (INTL_SITE, INTL_UPGRADE)
    }
}

/// Fill missing site/upgrade URLs from a provided TLD.
pub fn fill_setup_urls(mut cfg: SetupConfig) -> SetupConfig {
    let region = gateway_region(&cfg.site_url).or_else(|| gateway_region(&cfg.upgrade_url));
    let Some(region) = region else {
        return cfg;
    };
    let (site, upgrade) = pair_urls(region);
    if cfg.site_url.is_empty() {
        cfg.site_url = site.into();
    }
    if cfg.upgrade_url.is_empty() {
        cfg.upgrade_url = upgrade.into();
    }
    cfg
}

fn validate_router_api_key(key: &str) -> Result<(), SetupError> {
    let t = key.trim();
    if t.is_empty() {
        return Ok(());
    }
    if t.starts_with("bfvk-") {
        return Err(SetupError::InvalidKey(
            "OAuth key detected (bfvk-); use serve_api_key / [2/2] OAuth (Aria Compute)".into(),
        ));
    }
    Ok(())
}

fn validate_serve_api_key(key: &str) -> Result<(), SetupError> {
    let t = key.trim();
    if t.is_empty() {
        return Ok(());
    }
    if t.starts_with("sk-aria_") {
        return Err(SetupError::InvalidKey(
            "Local router key detected (sk-aria_); use router_api_key / [1/2] Local".into(),
        ));
    }
    if !t.starts_with("bfvk-") {
        return Err(SetupError::InvalidKey(
            "serve_api_key must start with bfvk-".into(),
        ));
    }
    Ok(())
}

/// Merge `updates` into `existing`. Validates; does not mutate `existing`.
pub fn apply_setup(existing: &SetupConfig, updates: &SetupUpdates) -> Result<SetupConfig, SetupError> {
    let mut out = existing.clone();
    if let Some(v) = &updates.router {
        out.router = v.clone();
    }
    if let Some(v) = &updates.router_api_key {
        validate_router_api_key(v)?;
        out.router_api_key = v.clone();
    }
    if let Some(v) = &updates.serve_site {
        out.serve_site = v.clone();
    }
    if let Some(v) = &updates.serve_api_key {
        validate_serve_api_key(v)?;
        out.serve_api_key = v.clone();
    }
    if let Some(v) = &updates.site_url {
        out.site_url = v.clone();
    }
    if let Some(v) = &updates.upgrade_url {
        out.upgrade_url = v.clone();
    }
    if let Some(v) = &updates.compute {
        out.compute = v.clone();
    }
    if let Some(v) = &updates.hf_token {
        out.hf_token = v.clone();
    }
    if let Some(v) = &updates.modelscope_api_token {
        out.modelscope_api_token = v.clone();
    }
    match out.compute.as_str() {
        "auto" | "cpu" | "cuda" => {}
        other => return Err(SetupError::InvalidCompute(other.into())),
    }
    validate_router_api_key(&out.router_api_key)?;
    validate_serve_api_key(&out.serve_api_key)?;
    Ok(fill_setup_urls(out))
}
