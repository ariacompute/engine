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
}
