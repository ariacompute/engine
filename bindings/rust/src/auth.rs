//! Instance-level Engine auth (in-memory; does not write config.yml).

use std::sync::Mutex;
use std::time::Duration;
use thiserror::Error;

pub const INTL_CLOUD: &str = "https://gateway.ariacompute.com";
pub const INTL_SITE: &str = "https://ariacompute.com";
pub const INTL_UPGRADE: &str = "https://github.com/ariacompute";
pub const CN_CLOUD: &str = "https://gateway.ariacompute.cn";
pub const CN_SITE: &str = "https://ariacompute.cn";
pub const CN_UPGRADE: &str = "https://gitee.com/ariacompute";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AuthError {
    #[error("invalid hybrid_mode: {0}")]
    InvalidHybridMode(String),
    #[error("invalid hybrid_execution: {0}")]
    InvalidHybridExecution(String),
    #[error("invalid compute: {0}")]
    InvalidCompute(String),
    #[error("hybrid_semantic_timeout_ms / cache_size must be positive integers")]
    InvalidTimeoutOrCache,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthConfig {
    pub cloud_api_key: String,
    pub cloud_url: String,
    pub site_url: String,
    pub upgrade_url: String,
    pub hybrid_mode: String,
    pub hybrid_execution: String,
    pub hybrid_semantic: bool,
    pub hybrid_semantic_timeout_ms: i32,
    pub hybrid_semantic_cache_size: i32,
    pub compute: String,
    pub hf_token: String,
    pub modelscope_api_token: String,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            cloud_api_key: String::new(),
            cloud_url: String::new(),
            site_url: String::new(),
            upgrade_url: String::new(),
            hybrid_mode: "balance".into(),
            hybrid_execution: "hybrid".into(),
            hybrid_semantic: true,
            hybrid_semantic_timeout_ms: 800,
            hybrid_semantic_cache_size: 512,
            compute: "auto".into(),
            hf_token: String::new(),
            modelscope_api_token: String::new(),
        }
    }
}

/// Partial merge. `None` fields are omitted.
#[derive(Debug, Clone, Default)]
pub struct AuthUpdates {
    pub cloud_api_key: Option<String>,
    pub cloud_url: Option<String>,
    pub site_url: Option<String>,
    pub upgrade_url: Option<String>,
    pub hybrid_mode: Option<String>,
    pub hybrid_execution: Option<String>,
    pub hybrid_semantic: Option<bool>,
    pub hybrid_semantic_timeout_ms: Option<i32>,
    pub hybrid_semantic_cache_size: Option<i32>,
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

fn pair_urls(region: &str) -> (&'static str, &'static str, &'static str) {
    if region == "cn" {
        (CN_CLOUD, CN_SITE, CN_UPGRADE)
    } else {
        (INTL_CLOUD, INTL_SITE, INTL_UPGRADE)
    }
}

/// Fill missing cloud/site/upgrade URLs from a provided TLD.
pub fn fill_auth_urls(mut cfg: AuthConfig) -> AuthConfig {
    let region = gateway_region(&cfg.site_url)
        .or_else(|| gateway_region(&cfg.cloud_url))
        .or_else(|| gateway_region(&cfg.upgrade_url));
    let Some(region) = region else {
        return cfg;
    };
    let (cloud, site, upgrade) = pair_urls(region);
    if cfg.cloud_url.is_empty() {
        cfg.cloud_url = cloud.into();
    }
    if cfg.site_url.is_empty() {
        cfg.site_url = site.into();
    }
    if cfg.upgrade_url.is_empty() {
        cfg.upgrade_url = upgrade.into();
    }
    cfg
}

fn locale_prefers_cn() -> bool {
    let lang = format!(
        "{}{}",
        std::env::var("LANG").unwrap_or_default(),
        std::env::var("LC_ALL").unwrap_or_default()
    )
    .to_ascii_lowercase();
    lang.contains("zh") || lang.contains(".cn") || lang.starts_with("cn")
}

type ProbeFn = fn(&str, &str) -> bool;

fn default_probe_dashboard(site_url: &str, api_key: &str) -> bool {
    let url = format!(
        "{}/api/dashboard/models",
        site_url.trim_end_matches('/')
    );
    let resp = ureq::get(&url)
        .set("User-Agent", "aria-engine-sdk/0.1.0")
        .set("Authorization", &format!("Bearer {api_key}"))
        .timeout(Duration::from_secs(10))
        .call();
    match resp {
        Ok(r) => (200..300).contains(&r.status()),
        Err(ureq::Error::Status(code, _)) => (200..300).contains(&code),
        Err(_) => false,
    }
}

static PROBE_DASHBOARD: Mutex<ProbeFn> = Mutex::new(default_probe_dashboard);

#[cfg(test)]
pub fn set_probe_dashboard(f: ProbeFn) {
    *PROBE_DASHBOARD.lock().unwrap() = f;
}

#[cfg(test)]
pub fn reset_probe_dashboard() {
    *PROBE_DASHBOARD.lock().unwrap() = default_probe_dashboard;
}

fn probe_dashboard(site_url: &str, api_key: &str) -> bool {
    let f = *PROBE_DASHBOARD.lock().unwrap();
    f(site_url, api_key)
}

/// Match CLI detect: probe Dashboard with the key, else locale fallback.
pub fn detect_gateway_pair(api_key: &str) -> (&'static str, &'static str, &'static str) {
    let key = api_key.trim();
    let (first, second) = if locale_prefers_cn() {
        ("cn", "intl")
    } else {
        ("intl", "cn")
    };
    for region in [first, second] {
        let (cloud, site, upgrade) = pair_urls(region);
        if !key.is_empty() && probe_dashboard(site, key) {
            return (cloud, site, upgrade);
        }
    }
    pair_urls(first)
}

/// Merge `updates` into `existing`. Validates; does not mutate `existing`.
pub fn apply_auth(existing: &AuthConfig, updates: &AuthUpdates) -> Result<AuthConfig, AuthError> {
    let mut out = existing.clone();
    if let Some(v) = &updates.cloud_api_key {
        out.cloud_api_key = v.clone();
    }
    if let Some(v) = &updates.cloud_url {
        out.cloud_url = v.clone();
    }
    if let Some(v) = &updates.site_url {
        out.site_url = v.clone();
    }
    if let Some(v) = &updates.upgrade_url {
        out.upgrade_url = v.clone();
    }
    if let Some(v) = &updates.hybrid_mode {
        out.hybrid_mode = v.clone();
    }
    if let Some(v) = &updates.hybrid_execution {
        out.hybrid_execution = v.clone();
    }
    if let Some(v) = updates.hybrid_semantic {
        out.hybrid_semantic = v;
    }
    if let Some(v) = updates.hybrid_semantic_timeout_ms {
        out.hybrid_semantic_timeout_ms = v;
    }
    if let Some(v) = updates.hybrid_semantic_cache_size {
        out.hybrid_semantic_cache_size = v;
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
    match out.hybrid_mode.as_str() {
        "cost" | "balance" | "intelligence" => {}
        other => return Err(AuthError::InvalidHybridMode(other.into())),
    }
    match out.hybrid_execution.as_str() {
        "hybrid" | "device" | "cloud" => {}
        other => return Err(AuthError::InvalidHybridExecution(other.into())),
    }
    match out.compute.as_str() {
        "auto" | "cpu" | "cuda" => {}
        other => return Err(AuthError::InvalidCompute(other.into())),
    }
    if out.hybrid_semantic_timeout_ms <= 0 || out.hybrid_semantic_cache_size <= 0 {
        return Err(AuthError::InvalidTimeoutOrCache);
    }
    out = fill_auth_urls(out);
    if !out.cloud_api_key.is_empty()
        && (out.cloud_url.is_empty() || out.site_url.is_empty() || out.upgrade_url.is_empty())
    {
        let (cloud, site, upgrade) = detect_gateway_pair(&out.cloud_api_key);
        if out.cloud_url.is_empty() {
            out.cloud_url = cloud.into();
        }
        if out.site_url.is_empty() {
            out.site_url = site.into();
        }
        if out.upgrade_url.is_empty() {
            out.upgrade_url = upgrade.into();
        }
    }
    Ok(out)
}
