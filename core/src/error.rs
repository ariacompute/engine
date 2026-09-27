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
