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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upgrade_urls() {
        assert_eq!(GatewayPair::INTL.upgrade_url(), "https://github.com/ariacompute");
        assert_eq!(GatewayPair::CN.upgrade_url(), "https://gitee.com/ariacompute");
    }
}
