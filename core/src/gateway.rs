//! Regional gateway / upgrade URL pair (.com vs .cn).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayPair {
    Intl,
    Cn,
}

impl GatewayPair {
    pub const INTL: Self = Self::Intl;
    pub const CN: Self = Self::Cn;

    pub fn site_url(self) -> &'static str {
        match self {
            Self::Intl => "https://ariacompute.com",
            Self::Cn => "https://ariacompute.cn",
        }
    }

    pub fn upgrade_url(self) -> &'static str {
        match self {
            Self::Intl => "https://github.com/ariacompute",
            Self::Cn => "https://gitee.com/ariacompute",
        }
    }

    pub fn from_url(url: &str) -> Self {
        let lower = url.to_ascii_lowercase();
        if lower.contains(".cn") || lower.contains("gitee.com") {
            Self::Cn
        } else {
            Self::Intl
        }
    }

    /// Prefer CN when locale looks Chinese; otherwise INTL.
    pub fn detect_default() -> Self {
        if let Ok(lang) = std::env::var("LANG") {
            if lang.to_ascii_lowercase().starts_with("zh") {
                return Self::Cn;
            }
        }
        Self::Intl
    }

    pub fn is_cn(self) -> bool {
        self == Self::Cn
    }
}

/// Public model hub selected from `site_url`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicHub {
    HuggingFace,
    ModelScope,
}

impl PublicHub {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::ModelScope => "modelscope",
        }
    }
}

/// `.cn` (or gitee) sites → ModelScope; otherwise Hugging Face.
pub fn preferred_hub(site_url: &str) -> PublicHub {
    match GatewayPair::from_url(site_url) {
        GatewayPair::Cn => PublicHub::ModelScope,
        GatewayPair::Intl => PublicHub::HuggingFace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn upgrade_urls() {
        assert_eq!(
            GatewayPair::INTL.upgrade_url(),
            "https://github.com/ariacompute"
        );
        assert_eq!(
            GatewayPair::CN.upgrade_url(),
            "https://gitee.com/ariacompute"
        );
    }

    #[test]
    fn preferred_hub_by_site() {
        assert_eq!(
            preferred_hub("https://ariacompute.com"),
            PublicHub::HuggingFace
        );
        assert_eq!(
            preferred_hub("https://ariacompute.cn"),
            PublicHub::ModelScope
        );
        assert_eq!(preferred_hub(""), PublicHub::HuggingFace);
    }

    #[test]
    fn site_urls_and_flags() {
        assert_eq!(GatewayPair::INTL.site_url(), "https://ariacompute.com");
        assert_eq!(GatewayPair::CN.site_url(), "https://ariacompute.cn");
        assert!(GatewayPair::CN.is_cn());
        assert!(!GatewayPair::INTL.is_cn());
        assert_eq!(PublicHub::HuggingFace.as_str(), "huggingface");
        assert_eq!(PublicHub::ModelScope.as_str(), "modelscope");
    }

    #[test]
    fn from_url_detects_cn_and_gitee() {
        assert_eq!(
            GatewayPair::from_url("https://ariacompute.cn"),
            GatewayPair::Cn
        );
        assert_eq!(
            GatewayPair::from_url("https://GITEE.com/ariacompute"),
            GatewayPair::Cn
        );
        assert_eq!(
            GatewayPair::from_url("https://ariacompute.com"),
            GatewayPair::Intl
        );
        assert_eq!(GatewayPair::from_url(""), GatewayPair::Intl);
    }

    #[test]
    #[serial]
    fn detect_default_follows_locale() {
        std::env::set_var("LANG", "zh_CN.UTF-8");
        assert_eq!(GatewayPair::detect_default(), GatewayPair::Cn);
        std::env::set_var("LANG", "en_US.UTF-8");
        assert_eq!(GatewayPair::detect_default(), GatewayPair::Intl);
        // no locale at all falls back to the international pair
        std::env::remove_var("LANG");
        assert_eq!(GatewayPair::detect_default(), GatewayPair::Intl);
    }
}
