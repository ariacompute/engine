use thiserror::Error;

pub type Result<T> = std::result::Result<T, AfmError>;

#[derive(Debug, Error)]
pub enum AfmError {
    #[error("{0}")]
    Msg(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
}

impl AfmError {
    pub fn msg(s: impl Into<String>) -> Self {
        Self::Msg(s.into())
    }
}

impl From<String> for AfmError {
    fn from(s: String) -> Self {
        Self::Msg(s)
    }
}

impl From<&str> for AfmError {
    fn from(s: &str) -> Self {
        Self::Msg(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_constructors_display_verbatim() {
        assert_eq!(AfmError::msg("boom").to_string(), "boom");
        assert_eq!(AfmError::from("plain").to_string(), "plain");
        assert_eq!(AfmError::from("owned".to_string()).to_string(), "owned");
    }

    #[test]
    fn source_errors_convert_transparently() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
        let err: AfmError = io.into();
        assert!(err.to_string().contains("missing file"));

        let json: AfmError = serde_json::from_str::<serde_json::Value>("{oops")
            .unwrap_err()
            .into();
        assert!(!json.to_string().is_empty());

        let yaml: AfmError = serde_yaml::from_str::<serde_yaml::Value>("a: [1,")
            .unwrap_err()
            .into();
        assert!(!yaml.to_string().is_empty());
    }
}
