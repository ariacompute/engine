//! Detect gateway + site URLs from API key region, locale, and connectivity.

use std::time::{Duration, Instant};

const INTL_CLOUD: &str = "https://gateway.ariacompute.com";
const INTL_SITE: &str = "https://ariacompute.com";
const INTL_UPGRADE: &str = "https://github.com/ariacompute";
const CN_CLOUD: &str = "https://gateway.ariacompute.cn";
const CN_SITE: &str = "https://ariacompute.cn";
const CN_UPGRADE: &str = "https://gitee.com/ariacompute";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayPair {
    pub cloud_url: &'static str,
    pub site_url: &'static str,
}

impl GatewayPair {
    pub const INTL: Self = Self {
        cloud_url: INTL_CLOUD,
        site_url: INTL_SITE,
    };
    pub const CN: Self = Self {
        cloud_url: CN_CLOUD,
        site_url: CN_SITE,
    };

    /// Org root for Releases (`…/engine` appended by `upgrade`).
    pub fn upgrade_url(self) -> &'static str {
        if self == Self::CN {
            CN_UPGRADE
        } else {
            INTL_UPGRADE
        }
    }

    /// Map a URL to the matching `.com` / `.cn` pair (cloud + site always aligned).
    pub fn from_url(url: &str) -> Option<Self> {
        let lower = url.to_ascii_lowercase();
        if lower.contains("ariacompute.cn") {
            Some(Self::CN)
        } else if lower.contains("ariacompute.com") {
            Some(Self::INTL)
        } else if lower.contains("gitee.com/ariacompute") {
            Some(Self::CN)
        } else if lower.contains("github.com/ariacompute") {
            Some(Self::INTL)
        } else {
            None
        }
    }

    pub fn matches_urls(self, cloud_url: &str, site_url: &str) -> bool {
        cloud_url.trim_end_matches('/') == self.cloud_url
            && site_url.trim_end_matches('/') == self.site_url
    }
}

/// Prefer the region whose Dashboard accepts `api_key`. Falls back to locale +
/// site reachability. Always returns a matched cloud/site pair (both `.com` or both `.cn`).
pub async fn detect_gateway_and_site(api_key: &str) -> GatewayPair {
    if let Some(pair) = detect_by_api_key(api_key).await {
        return pair;
    }
    detect_by_locale_and_probe().await
}

/// If config URLs are mismatched or the key is rejected by the configured site,
/// rewrite to the pair that accepts the key (or the consistent locale fallback).
/// Returns `true` when `cloud_url` / `site_url` / `upgrade_url` changed.
pub async fn reconcile_config_urls(cfg: &mut crate::config::AriaConfig) -> bool {
    if cfg.cloud_api_key.trim().is_empty() {
        return false;
    }
    let before_cloud = cfg.cloud_url.clone();
    let before_site = cfg.site_url.clone();
    let before_upgrade = cfg.upgrade_url.clone();

    let site_pair = GatewayPair::from_url(&cfg.site_url);
    let cloud_pair = GatewayPair::from_url(&cfg.cloud_url);
    let matched = match (site_pair, cloud_pair) {
        (Some(s), Some(c)) if s == c => Some(s),
        _ => None,
    };

    if let Some(pair) = matched {
        let (ok, _) = probe_key_on_site(pair.site_url, cfg.cloud_api_key.trim()).await;
        if ok {
            cfg.cloud_url = pair.cloud_url.to_string();
            cfg.site_url = pair.site_url.to_string();
            cfg.upgrade_url = pair.upgrade_url().to_string();
            return cfg.cloud_url != before_cloud
                || cfg.site_url != before_site
                || cfg.upgrade_url != before_upgrade;
        }
    }

    let pair = detect_gateway_and_site(&cfg.cloud_api_key).await;
    cfg.cloud_url = pair.cloud_url.to_string();
    cfg.site_url = pair.site_url.to_string();
    cfg.upgrade_url = pair.upgrade_url().to_string();
    cfg.cloud_url != before_cloud
        || cfg.site_url != before_site
        || cfg.upgrade_url != before_upgrade
}

async fn detect_by_api_key(api_key: &str) -> Option<GatewayPair> {
    let key = api_key.trim();
    if key.is_empty() {
        return None;
    }
    let prefer_cn = locale_prefers_cn();
    let (first, second) = if prefer_cn {
        (GatewayPair::CN, GatewayPair::INTL)
    } else {
        (GatewayPair::INTL, GatewayPair::CN)
    };

    let (first_ok, first_ms) = probe_key_on_site(first.site_url, key).await;
    let (second_ok, second_ms) = probe_key_on_site(second.site_url, key).await;

    match (first_ok, second_ok) {
        (true, false) => Some(first),
        (false, true) => Some(second),
        (true, true) => {
            // Key valid on both (unusual); keep locale preference unless other is clearly faster.
            if second_ms + 50 < first_ms {
                Some(second)
            } else {
                Some(first)
            }
        }
        (false, false) => None,
    }
}

async fn detect_by_locale_and_probe() -> GatewayPair {
    let prefer_cn = locale_prefers_cn();
    let preferred = if prefer_cn {
        GatewayPair::CN
    } else {
        GatewayPair::INTL
    };
    let other = if prefer_cn {
        GatewayPair::INTL
    } else {
        GatewayPair::CN
    };

    let (pref_ok, pref_ms) = probe_url(preferred.site_url).await;
    let (other_ok, other_ms) = probe_url(other.site_url).await;

    match (pref_ok, other_ok) {
        (true, false) => preferred,
        (false, true) => other,
        (true, true) => {
            if other_ms + 50 < pref_ms {
                other
            } else {
                preferred
            }
        }
        (false, false) => preferred,
    }
}

fn locale_prefers_cn() -> bool {
    for key in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Ok(v) = std::env::var(key) {
            let lower = v.to_ascii_lowercase();
            if lower.starts_with("zh") || lower.contains("zh_cn") || lower.contains("zh-cn") {
                return true;
            }
        }
    }
    false
}

/// `Ok(true)` = key accepted (2xx). `Ok(false)` = reachable but unauthorized / other.
async fn probe_key_on_site(site_url: &str, api_key: &str) -> (bool, u128) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::limited(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return (false, u128::MAX),
    };
    let url = format!("{}/api/dashboard/models", site_url.trim_end_matches('/'));
    let start = Instant::now();
    let accepted = match client
        .get(&url)
        .bearer_auth(api_key)
        .header("Accept", "application/json")
        .send()
        .await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    };
    (accepted, start.elapsed().as_millis())
}

async fn probe_url(url: &str) -> (bool, u128) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .redirect(reqwest::redirect::Policy::limited(3))
        .build()
    {
        Ok(c) => c,
        Err(_) => return (false, u128::MAX),
    };
    let start = Instant::now();
    let ok = match client.head(url).send().await {
        Ok(resp) => resp.status().as_u16() < 500,
        Err(_) => match client.get(url).send().await {
            Ok(resp) => resp.status().as_u16() < 500,
            Err(_) => false,
        },
    };
    (ok, start.elapsed().as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_are_stable_and_matched() {
        assert!(GatewayPair::INTL.cloud_url.contains(".com"));
        assert!(GatewayPair::INTL.site_url.contains(".com"));
        assert!(GatewayPair::CN.cloud_url.contains(".cn"));
        assert!(GatewayPair::CN.site_url.contains(".cn"));
        assert_eq!(GatewayPair::INTL.upgrade_url(), INTL_UPGRADE);
        assert_eq!(GatewayPair::CN.upgrade_url(), CN_UPGRADE);
    }

    #[test]
    fn from_url_maps_tld_to_pair() {
        assert_eq!(
            GatewayPair::from_url("https://gateway.ariacompute.cn/v1"),
            Some(GatewayPair::CN)
        );
        assert_eq!(
            GatewayPair::from_url("https://ariacompute.com"),
            Some(GatewayPair::INTL)
        );
        assert_eq!(
            GatewayPair::from_url("https://github.com/ariacompute"),
            Some(GatewayPair::INTL)
        );
        assert_eq!(
            GatewayPair::from_url("https://gitee.com/ariacompute"),
            Some(GatewayPair::CN)
        );
        assert_eq!(GatewayPair::from_url("https://example.com"), None);
    }

    #[test]
    fn matches_urls_requires_both() {
        assert!(GatewayPair::CN.matches_urls(CN_CLOUD, CN_SITE));
        assert!(!GatewayPair::CN.matches_urls(INTL_CLOUD, CN_SITE));
    }
}
