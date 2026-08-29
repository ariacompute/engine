//! Thin Rust SDK: prefer `aria_inference` for native use; re-exports FFI for embedding tests.

pub use aria_ffi::{
    aria_complete, aria_complete_stream, aria_embed, aria_last_error, aria_model_destroy,
    aria_model_init, aria_transcribe, AriaModelHandle,
};
pub use aria_inference::{EngineError, GenerateOpts, Generation, Session, SessionBuilder};

mod download;
mod setup;
pub use download::{download_model, download_model_setup, ensure_ffi_lib, DownloadError};
pub use setup::{
    apply_setup, fill_setup_urls, SetupConfig, SetupError, SetupUpdates, CN_SITE, CN_UPGRADE,
};

/// Options controlling model auto-download from the regional public hub.
#[derive(Default, Clone)]
pub struct OpenOptions {
    /// Legacy generic hub token. Dashboard `sk-` / `bfvk-` keys are ignored.
    pub token: Option<String>,
    /// Hugging Face hub token (`.com`). Same field as `aria-engine setup` `hf_token`.
    pub hf_token: Option<String>,
    /// ModelScope hub token (`.cn`). Same field as `aria-engine setup` `modelscope_api_token`.
    pub modelscope_api_token: Option<String>,
    /// Site used to pick the regional hub. Defaults to `https://ariacompute.com` (`.com` → HF, `.cn` → ModelScope).
    pub site: Option<String>,
}

/// High-level convenience over `Session`.
pub struct Engine {
    session: Option<Session>,
    cfg: SetupConfig,
    generic_token: Option<String>,
}

impl Engine {
    /// Empty construct. Call [`setup`](Self::setup) then [`open`](Self::open) to download/load.
    pub fn new() -> Self {
        Self {
            session: None,
            cfg: SetupConfig::default(),
            generic_token: None,
        }
    }

    /// Set Config / Run fields on this instance only. Does not write engine.yml.
    pub fn setup(&mut self, updates: &SetupUpdates) -> Result<&mut Self, SetupError> {
        self.cfg = apply_setup(&self.cfg, updates)?;
        Ok(self)
    }

    pub fn setup_status(&self) -> &SetupConfig {
        &self.cfg
    }

    /// Reset instance defaults. Does not delete ~/.ariacompute/engine.yml.
    pub fn setup_clear(&mut self) -> &mut Self {
        self.cfg = SetupConfig::default();
        self
    }

    /// Open a local Aria quant bundle directory (path only; no download).
    pub fn from_bundle(bundle_path: impl AsRef<std::path::Path>) -> Result<Self, EngineError> {
        let session = SessionBuilder::new().model(bundle_path).build()?;
        Ok(Self {
            session: Some(session),
            cfg: SetupConfig::default(),
            generic_token: None,
        })
    }

    /// Open a local Aria quant bundle directory (path only; no download).
    pub fn open(bundle_path: impl AsRef<std::path::Path>) -> Result<Self, EngineError> {
        Self::from_bundle(bundle_path)
    }

    fn load_ref(&mut self, model_ref: &str, opts: &OpenOptions) -> Result<(), OpenError> {
        let _ = ensure_ffi_lib(opts.site.as_deref())?;
        let path = if model_ref.contains('/')
            || model_ref.contains('\\')
            || std::path::Path::new(model_ref).exists()
        {
            std::path::PathBuf::from(model_ref)
        } else {
            download_model_setup(
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
        self.session = Some(session);
        Ok(())
    }

    /// Download (if needed) and load a model using instance setup.
    pub fn open_named(&mut self, model_ref: &str) -> Result<&mut Self, OpenError> {
        let opts = OpenOptions {
            token: self.generic_token.clone(),
            hf_token: if self.cfg.hf_token.is_empty() {
                None
            } else {
                Some(self.cfg.hf_token.clone())
            },
            modelscope_api_token: if self.cfg.modelscope_api_token.is_empty() {
                None
            } else {
                Some(self.cfg.modelscope_api_token.clone())
            },
            site: if self.cfg.site_url.is_empty() {
                None
            } else {
                Some(self.cfg.site_url.clone())
            },
        };
        self.load_ref(model_ref, &opts)?;
        Ok(self)
    }

    /// Open a model by reference. If `model_ref` looks like a local path
    /// (contains a separator or exists on disk) it is loaded directly;
    /// otherwise it is treated as a model name and auto-downloaded from the
    /// regional public hub into `~/.ariacompute/models/{model}` before loading.
    pub fn open_model(model_ref: &str, opts: &OpenOptions) -> Result<Self, OpenError> {
        let mut eng = Self::new();
        let updates = SetupUpdates {
            site_url: opts.site.clone(),
            hf_token: opts.hf_token.clone(),
            modelscope_api_token: opts.modelscope_api_token.clone(),
            ..Default::default()
        };
        eng.generic_token = opts.token.clone();
        let _ = eng.setup(&updates);
        eng.load_ref(model_ref, opts)?;
        Ok(eng)
    }

    fn session_mut(&mut self) -> Result<&mut Session, EngineError> {
        self.session
            .as_mut()
            .ok_or_else(|| EngineError::InvalidParam("engine not opened".into()))
    }

    fn session_ref(&self) -> Result<&Session, EngineError> {
        self.session
            .as_ref()
            .ok_or_else(|| EngineError::InvalidParam("engine not opened".into()))
    }

    pub fn complete(&mut self, prompt: &str, opts: &GenerateOpts) -> Result<Generation, EngineError> {
        let turns = [aria_inference::ChatTurn::new("user", prompt)];
        let session = self.session_mut()?;
        let tokens = session.encode_chat(&turns);
        session.generate(&tokens, opts)
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        self.session_ref()?.embed_text(text)
    }

    pub fn transcribe(&self, pcm: &[u8]) -> Result<String, EngineError> {
        self.session_ref()?.transcribe_pcm16le(pcm)
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn setup_instance_all_fields() {
        let mut eng = Engine::new();
        eng.setup(&SetupUpdates {
            router: Some("http://127.0.0.1:8080".into()),
            site_url: Some(crate::setup::CN_SITE.into()),
            upgrade_url: Some(crate::setup::CN_UPGRADE.into()),
            compute: Some("cpu".into()),
            hf_token: Some("hf_abc".into()),
            modelscope_api_token: Some("ms_xyz".into()),
        })
        .unwrap();
        let st = eng.setup_status();
        assert_eq!(st.router, "http://127.0.0.1:8080");
        assert_eq!(st.compute, "cpu");
        assert_eq!(st.hf_token, "hf_abc");
        assert_eq!(st.modelscope_api_token, "ms_xyz");
        assert_eq!(st.site_url, crate::setup::CN_SITE);
    }

    #[test]
    fn setup_partial_merge() {
        let mut eng = Engine::new();
        eng.setup(&SetupUpdates {
            hf_token: Some("hf_one".into()),
            router: Some("http://127.0.0.1:1".into()),
            ..Default::default()
        })
        .unwrap();
        eng.setup(&SetupUpdates {
            compute: Some("cuda".into()),
            ..Default::default()
        })
        .unwrap();
        let st = eng.setup_status();
        assert_eq!(st.hf_token, "hf_one");
        assert_eq!(st.router, "http://127.0.0.1:1");
        assert_eq!(st.compute, "cuda");
    }

    #[test]
    fn setup_invalid_enum_leaves_state() {
        let mut eng = Engine::new();
        eng.setup(&SetupUpdates {
            compute: Some("cpu".into()),
            ..Default::default()
        })
        .unwrap();
        assert!(eng
            .setup(&SetupUpdates {
                compute: Some("gpu".into()),
                ..Default::default()
            })
            .is_err());
        assert_eq!(eng.setup_status().compute, "cpu");
    }

    #[test]
    fn setup_clear_resets_instance() {
        let mut eng = Engine::new();
        eng.setup(&SetupUpdates {
            hf_token: Some("hf_x".into()),
            compute: Some("cpu".into()),
            ..Default::default()
        })
        .unwrap();
        eng.setup_clear();
        let st = eng.setup_status();
        assert_eq!(st.hf_token, "");
        assert_eq!(st.compute, "auto");
    }

    #[test]
    fn setup_fills_urls_from_site_tld() {
        let mut eng = Engine::new();
        eng.setup(&SetupUpdates {
            site_url: Some("https://ariacompute.cn".into()),
            ..Default::default()
        })
        .unwrap();
        let st = eng.setup_status();
        assert_eq!(st.upgrade_url, crate::setup::CN_UPGRADE);
    }

    #[test]
    fn setup_does_not_write_engine_yml() {
        let _guard = crate::download::ENV_LOCK.lock().unwrap();
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("ARIA_COMPUTE_HOME", home.path());
        let mut eng = Engine::new();
        eng.setup(&SetupUpdates {
            router: Some("http://127.0.0.1:8080".into()),
            site_url: Some("https://ariacompute.com".into()),
            hf_token: Some("hf_x".into()),
            ..Default::default()
        })
        .unwrap();
        assert!(!home.path().join("engine.yml").is_file());
        assert!(!home.path().join("config.yml").is_file());
        std::env::remove_var("ARIA_COMPUTE_HOME");
    }
}
