//! Thin Rust SDK: prefer `aria_inference` for native use; re-exports FFI for embedding tests.

pub use aria_ffi::{
    aria_complete, aria_complete_stream, aria_embed, aria_last_error, aria_model_destroy,
    aria_model_init, aria_transcribe, AriaModelHandle,
};
pub use aria_inference::{GenerateOpts, Generation, Session, SessionBuilder};

/// High-level convenience over `Session`.
pub struct Engine {
    session: Session,
}

impl Engine {
    pub fn open(bundle_path: impl AsRef<std::path::Path>) -> Result<Self, aria_inference::EngineError> {
        let session = SessionBuilder::new().model(bundle_path).build()?;
        Ok(Self { session })
    }

    pub fn complete(&mut self, prompt: &str, opts: &GenerateOpts) -> Result<Generation, aria_inference::EngineError> {
        let tokens = self.session.encode_text(prompt);
        self.session.generate(&tokens, opts)
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, aria_inference::EngineError> {
        self.session.embed_text(text)
    }

    pub fn transcribe(&self, pcm: &[u8]) -> Result<String, aria_inference::EngineError> {
        self.session.transcribe_pcm16le(pcm)
    }
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
}
