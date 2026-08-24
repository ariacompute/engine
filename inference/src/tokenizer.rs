//! Bundle tokenizer sidecar via HuggingFace `tokenizers`.
//!
//! Loads `tokenizer.json` for real encode + decode. Without a sidecar, Session falls
//! back to naive byte encode / `<id>` decode placeholders (tiny fixtures).

use aria_kernel::EngineError;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

const STOP_TOKEN_STRINGS: &[&str] = &[
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "</s>",
    "<end_of_turn>",
    "<turn|>",
    "<eos>",
    "<|end|>",
];

#[derive(Clone)]
pub struct BundleTokenizer {
    inner: Arc<Tokenizer>,
    stop_ids: Vec<u32>,
}

impl std::fmt::Debug for BundleTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BundleTokenizer")
            .field("vocab_size", &self.inner.get_vocab_size(false))
            .field("stop_ids", &self.stop_ids)
            .finish()
    }
}

impl BundleTokenizer {
    fn wrap(tok: Tokenizer) -> Self {
        let stop_ids = collect_stop_ids(&tok);
        Self {
            inner: Arc::new(tok),
            stop_ids,
        }
    }

    /// Load from a bundle directory. Returns `Ok(None)` if no `tokenizer.json`.
    pub fn try_load(dir: &Path) -> Result<Option<Self>, EngineError> {
        let path = dir.join("tokenizer.json");
        if !path.is_file() {
            return Ok(None);
        }
        let tok = Tokenizer::from_file(&path).map_err(|e| {
            EngineError::Format(format!(
                "tokenizer.json load failed ({}): {e}",
                path.display()
            ))
        })?;
        Ok(Some(Self::wrap(tok)))
    }

    pub fn from_tokenizer_json(raw: &str) -> Result<Self, EngineError> {
        let tok = Tokenizer::from_bytes(raw.as_bytes())
            .map_err(|e| EngineError::Format(format!("tokenizer.json parse failed: {e}")))?;
        Ok(Self::wrap(tok))
    }

    /// Encode text → token ids (no extra specials; chat template is already in `text`).
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

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids.contains(&id)
    }

    pub fn stop_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    pub fn has_token(&self, token: &str) -> bool {
        self.inner.token_to_id(token).is_some()
    }

    /// Prefer tokenizer specials over a possibly-wrong Session family (serve used to
    /// hardcode gemma even for Qwen bundles).
    pub fn chat_family_hint(&self) -> Option<&'static str> {
        if self.has_token("<|im_start|>") {
            if self.has_token("<think>") {
                Some("qwen/qwen3-0.6b")
            } else {
                Some("chatml")
            }
        } else if self.has_token("<|turn>") {
            Some("gemma/gemma-4-e2b-it")
        } else if self.has_token("<start_of_turn>") {
            Some("gemma/gemma-3-1b-it")
        } else if self.has_token("<|eot_id|>") {
            Some("llama")
        } else {
            None
        }
    }
}

fn collect_stop_ids(tok: &Tokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut push = |id: u32| {
        if !ids.contains(&id) {
            ids.push(id);
        }
    };
    for name in STOP_TOKEN_STRINGS {
        if let Some(id) = tok.token_to_id(name) {
            push(id);
        }
        // Added tokens sometimes miss token_to_id; encode as a whole piece.
        if let Ok(enc) = tok.encode(*name, false) {
            let got = enc.get_ids();
            if got.len() == 1 {
                push(got[0]);
            }
        }
    }
    for (id, added) in tok.get_added_tokens_decoder() {
        let content = added.content;
        if STOP_TOKEN_STRINGS.contains(&content.as_str()) {
            push(id);
        }
    }
    ids
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
        let tok = BundleTokenizer::try_load(dir.path())
            .unwrap()
            .expect("loaded");
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

    #[test]
    fn stop_ids_from_added_tokens() {
        let mut v: serde_json::Value = serde_json::from_str(&word_level_json()).unwrap();
        v["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "id": 3,
                "content": "<|im_end|>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }));
        v["model"]["vocab"]["<|im_end|>"] = serde_json::json!(3);
        let tok = BundleTokenizer::from_tokenizer_json(&v.to_string()).unwrap();
        assert!(tok.is_stop(3));
        assert!(!tok.is_stop(0));
    }
}
