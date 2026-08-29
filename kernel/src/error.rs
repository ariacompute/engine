use thiserror::Error;

/// Unified engine error (requirements §3.6).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EngineError {
    #[error("io: {0}")]
    Io(String),
    #[error("format: {0}")]
    Format(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("quant: {0}")]
    Quant(String),
    #[error("unsupported family: {0}")]
    UnsupportedFamily(String),
    #[error("cloud: {0}")]
    Cloud(String),
    #[error("upstream: {0}")]
    Upstream(String),
    #[error("invalid param: {0}")]
    InvalidParam(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
