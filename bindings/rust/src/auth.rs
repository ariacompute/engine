//! Instance-level Engine auth (in-memory; does not write config.yml).

use thiserror::Error;

pub const INTL_SITE: &str = "https://ariacompute.com";
pub const INTL_UPGRADE: &str = "https://github.com/ariacompute";
pub const CN_SITE: &str = "https://ariacompute.cn";
pub const CN_UPGRADE: &str = "https://gitee.com/ariacompute";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AuthError {
    #[error("invalid compute: {0}")]
    InvalidCompute(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthConfig {
    pub router: String,
    pub site_url: String,
    pub upgrade_url: String,
    pub compute: String,
    pub hf_token: String,
    pub modelscope_api_token: String,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            router: String::new(),
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
pub struct AuthUpdates {
    pub router: Option<String>,
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
pub fn fill_auth_urls(mut cfg: AuthConfig) -> AuthConfig {
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

/// Merge `updates` into `existing`. Validates; does not mutate `existing`.
pub fn apply_auth(existing: &AuthConfig, updates: &AuthUpdates) -> Result<AuthConfig, AuthError> {
    let mut out = existing.clone();
    if let Some(v) = &updates.router {
        out.router = v.clone();
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
        other => return Err(AuthError::InvalidCompute(other.into())),
    }
    Ok(fill_auth_urls(out))
}
