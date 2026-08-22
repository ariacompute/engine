//! Thin Rust SDK: prefer `aria_inference` for native use; re-exports FFI for embedding tests.

pub use aria_ffi::{
    aria_complete, aria_complete_stream, aria_embed, aria_last_error, aria_model_destroy,
    aria_model_init, aria_transcribe, AriaModelHandle,
};
pub use aria_inference::{EngineError, GenerateOpts, Generation, Session, SessionBuilder};

mod download;
pub use download::{download_model, DownloadError};

/// Options controlling model auto-download from the Dashboard private source.
#[derive(Default, Clone)]
pub struct OpenOptions {
    /// Dashboard bearer token. Required when `model_ref` is a model name.
    pub token: Option<String>,
    /// Dashboard base URL. Defaults to `https://ariacompute.com`.
    pub site: Option<String>,
}

/// High-level convenience over `Session`.
pub struct Engine {
    session: Session,
}

impl Engine {
    /// Open a local Aria quant bundle directory (path only; no download).
    pub fn open(bundle_path: impl AsRef<std::path::Path>) -> Result<Self, EngineError> {
        let session = SessionBuilder::new().model(bundle_path).build()?;
        Ok(Self { session })
    }

    /// Open a model by reference. If `model_ref` looks like a local path
    /// (contains a separator or exists on disk) it is loaded directly;
    /// otherwise it is treated as a model name and auto-downloaded from the
    /// Dashboard source into `~/.ariacompute/models/{model}` before loading.
    pub fn open_model(model_ref: &str, opts: &OpenOptions) -> Result<Self, OpenError> {
        let path = if model_ref.contains('/') || model_ref.contains('\\') || std::path::Path::new(model_ref).exists() {
            std::path::PathBuf::from(model_ref)
        } else {
            let token = opts
                .token
                .as_ref()
                .ok_or_else(|| OpenError::MissingToken(model_ref.to_string()))?;
            download_model(model_ref, token, opts.site.as_deref())
                .map_err(OpenError::Download)?
        };
        let session = SessionBuilder::new()
            .model(&path)
            .build()
            .map_err(OpenError::Engine)?;
        Ok(Self { session })
    }

    pub fn complete(&mut self, prompt: &str, opts: &GenerateOpts) -> Result<Generation, EngineError> {
        let turns = [aria_inference::ChatTurn::new("user", prompt)];
        let tokens = self.session.encode_chat(&turns);
        self.session.generate(&tokens, opts)
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        self.session.embed_text(text)
    }

    pub fn transcribe(&self, pcm: &[u8]) -> Result<String, EngineError> {
        self.session.transcribe_pcm16le(pcm)
    }
}

/// Error returned by [`Engine::open_model`].
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error("model name '{0}' requires an api token")]
    MissingToken(String),
    #[error("download failed: {0}")]
    Download(#[from] DownloadError),
    #[error("engine load failed: {0}")]
    Engine(#[from] EngineError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use aria_inference::fixture::write_tiny_q4_bundle;

    #[test]
    fn engine_complete_ok() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let mut eng = Engine::open(dir.path()).unwrap();
        let g = eng
            .complete("hi", &GenerateOpts { max_tokens: 2, temperature: 0.0 })
            .unwrap();
        assert!(!g.text.is_empty());
        assert!(!eng.embed("x").unwrap().is_empty());
        assert!(!eng.transcribe(&[0, 1, 2, 3]).unwrap().is_empty());
    }

    #[test]
    fn open_model_local_path_no_token() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let mut eng = Engine::open_model(dir.path().to_str().unwrap(), &OpenOptions::default()).unwrap();
        let g = eng
            .complete("hi", &GenerateOpts { max_tokens: 2, temperature: 0.0 })
            .unwrap();
        assert!(!g.text.is_empty());
    }

    #[test]
    fn open_model_name_requires_token() {
        let err = Engine::open_model("gemma-4-e2b-it_q4", &OpenOptions::default());
        assert!(matches!(err, Err(OpenError::MissingToken(_))));
    }
}
