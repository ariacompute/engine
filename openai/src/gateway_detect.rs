//! Detect gateway + site URLs from locale and short connectivity probes.

use std::time::{Duration, Instant};

const INTL_CLOUD: &str = "https://gateway.ariacompute.com";
const INTL_SITE: &str = "https://ariacompute.com";
const CN_CLOUD: &str = "https://gateway.ariacompute.cn";
const CN_SITE: &str = "https://ariacompute.cn";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayPair {
    pub cloud_url: &'static str,
    pub site_url: &'static str,
}

/// Prefer `.cn` when locale looks Chinese; otherwise `.com`. Then probe both
/// and keep the faster reachable pair (falling back to locale preference).
pub async fn detect_gateway_and_site() -> GatewayPair {
    let prefer_cn = locale_prefers_cn();
    let preferred = if prefer_cn {
        GatewayPair {
            cloud_url: CN_CLOUD,
            site_url: CN_SITE,
        }
    } else {
        GatewayPair {
            cloud_url: INTL_CLOUD,
            site_url: INTL_SITE,
        }
    };
    let other = if prefer_cn {
        GatewayPair {
            cloud_url: INTL_CLOUD,
            site_url: INTL_SITE,
        }
    } else {
        GatewayPair {
            cloud_url: CN_CLOUD,
            site_url: CN_SITE,
        }
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
    fn pairs_are_stable() {
        assert!(INTL_CLOUD.contains(".com"));
        assert!(CN_CLOUD.contains(".cn"));
    }
}
