//! Thin Rust SDK: prefer `aria_inference` for native use; re-exports FFI for embedding tests.

pub use aria_ffi::{
    aria_complete, aria_complete_stream, aria_embed, aria_last_error, aria_model_destroy,
    aria_model_init, aria_transcribe, AriaModelHandle,
};
pub use aria_inference::{EngineError, GenerateOpts, Generation, Session, SessionBuilder};

mod download;
pub use download::{download_model, download_model_auth, ensure_ffi_lib, DownloadError};

/// Options controlling model auto-download from the regional public hub.
#[derive(Default, Clone)]
pub struct OpenOptions {
    /// Legacy generic hub token. Dashboard `sk-` / `bfvk-` keys are ignored.
    pub token: Option<String>,
    /// Hugging Face hub token (`.com`). Same field as `aria-engine auth` `hf_token`.
    pub hf_token: Option<String>,
    /// ModelScope hub token (`.cn`). Same field as `aria-engine auth` `modelscope_api_token`.
    pub modelscope_api_token: Option<String>,
    /// Site used to pick the regional hub. Defaults to `https://ariacompute.com` (`.com` → HF, `.cn` → ModelScope).
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
    /// regional public hub into `~/.ariacompute/models/{model}` before loading.
    pub fn open_model(model_ref: &str, opts: &OpenOptions) -> Result<Self, OpenError> {
        let _ = ensure_ffi_lib(opts.site.as_deref())?;
        let path = if model_ref.contains('/') || model_ref.contains('\\') || std::path::Path::new(model_ref).exists() {
            std::path::PathBuf::from(model_ref)
        } else {
            download_model_auth(
                model_ref,
                opts.token.as_deref().unwrap_or(""),
                opts.site.as_deref(),
                opts.hf_token.as_deref(),
                opts.modelscope_api_token.as_deref(),
            )
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
        let _guard = crate::download::ENV_LOCK.lock().unwrap();
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", home.path());
        let libdir = home.path().join("lib");
        std::fs::create_dir_all(&libdir).unwrap();
        let name = if cfg!(windows) {
            "aria_ffi.dll"
        } else if cfg!(target_os = "macos") {
            "libaria_ffi.dylib"
        } else {
            "libaria_ffi.so"
        };
        std::fs::write(libdir.join(name), b"x").unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let mut eng = Engine::open_model(dir.path().to_str().unwrap(), &OpenOptions::default()).unwrap();
        let g = eng
            .complete("hi", &GenerateOpts { max_tokens: 2, temperature: 0.0 })
            .unwrap();
        assert!(!g.text.is_empty());
        std::env::remove_var("ARIA_COMPUTE_HOME");
    }
}
