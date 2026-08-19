//! Bundle tokenizer sidecar via HuggingFace `tokenizers`.
//!
//! Loads `tokenizer.json` for real encode + decode. Without a sidecar, Session falls
//! back to naive byte encode / `<id>` decode placeholders (tiny fixtures).

use aria_kernel::EngineError;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct BundleTokenizer {
    inner: Arc<Tokenizer>,
}

impl std::fmt::Debug for BundleTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BundleTokenizer")
            .field("vocab_size", &self.inner.get_vocab_size(false))
            .finish()
    }
}

impl BundleTokenizer {
    /// Load from a bundle directory. Returns `Ok(None)` if no `tokenizer.json`.
    pub fn try_load(dir: &Path) -> Result<Option<Self>, EngineError> {
        let path = dir.join("tokenizer.json");
        if !path.is_file() {
            return Ok(None);
        }
        let tok = Tokenizer::from_file(&path).map_err(|e| {
            EngineError::Format(format!("tokenizer.json load failed ({}): {e}", path.display()))
        })?;
        Ok(Some(Self {
            inner: Arc::new(tok),
        }))
    }

    pub fn from_tokenizer_json(raw: &str) -> Result<Self, EngineError> {
        let tok = Tokenizer::from_bytes(raw.as_bytes())
            .map_err(|e| EngineError::Format(format!("tokenizer.json parse failed: {e}")))?;
        Ok(Self {
            inner: Arc::new(tok),
        })
    }

    /// Encode text → token ids (no special tokens added; chat templates stay caller-side).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| EngineError::InvalidParam(format!("tokenizer encode failed: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids → UTF-8, skipping special tokens by default.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_opts(ids, true)
    }

    pub fn decode_opts(&self, ids: &[u32], skip_special: bool) -> String {
        match self.inner.decode(ids, skip_special) {
            Ok(s) => s,
            Err(_) => decode_placeholders(ids),
        }
    }
}

/// Fallback when no tokenizer sidecar: stable `<id>` placeholders (legacy demos).
pub fn decode_placeholders(ids: &[u32]) -> String {
    ids.iter().map(|t| format!("<{t}>")).collect()
}

/// Fallback encode when no sidecar: map UTF-8 bytes into `[0, vocab)`.
pub fn encode_naive(text: &str, vocab_size: u32) -> Vec<u32> {
    let vocab = vocab_size.max(1);
    if text.is_empty() {
        return vec![1 % vocab];
    }
    text.bytes().map(|b| (b as u32) % vocab).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_level_json() -> String {
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 2,
                    "content": "[UNK]",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "Hello": 0,
                    "world": 1,
                    "[UNK]": 2
                },
                "unk_token": "[UNK]"
            }
        })
        .to_string()
    }

    #[test]
    fn encode_decode_roundtrip_word_level() {
        let tok = BundleTokenizer::from_tokenizer_json(&word_level_json()).unwrap();
        let ids = tok.encode("Hello world").unwrap();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(tok.decode(&ids), "Hello world");
    }

    #[test]
    fn decode_skips_special() {
        let tok = BundleTokenizer::from_tokenizer_json(&word_level_json()).unwrap();
        assert_eq!(tok.decode(&[0, 2, 1]), "Hello world");
        assert!(tok.decode_opts(&[0, 2, 1], false).contains("[UNK]"));
    }

    #[test]
    fn try_load_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), word_level_json()).unwrap();
        let tok = BundleTokenizer::try_load(dir.path()).unwrap().expect("loaded");
        assert_eq!(tok.encode("Hello").unwrap(), vec![0]);
    }

    #[test]
    fn try_load_missing_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(BundleTokenizer::try_load(dir.path()).unwrap().is_none());
    }

    #[test]
    fn naive_encode_fallback() {
        assert_eq!(encode_naive("AB", 256), vec![65, 66]);
        assert_eq!(encode_naive("", 16), vec![1]);
    }
}
