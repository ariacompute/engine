//! Bundle tokenizer sidecar: decode token ids → UTF-8 text.
//!
//! Loads `tokenizer.json` (HF fast format). Encode stays elsewhere (naive fallback
//! until a full BPE path is wired). Decode uses vocab + GPT-2 byte-level reverse map
//! so Qwen/GPT-style pieces become readable strings instead of `<id>` placeholders.

use aria_kernel::EngineError;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Inverse of OpenAI/HF `bytes_to_unicode` (char → original byte).
fn unicode_to_byte_map() -> HashMap<char, u8> {
    let mut bs: Vec<u8> = (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs.into_iter().map(|c| char::from_u32(c).unwrap_or('\u{FFFD}')))
        .map(|(b, c)| (c, b))
        .collect()
}

#[derive(Debug, Clone)]
pub struct BundleTokenizer {
    /// id → piece (HF vocab / added_tokens).
    id_to_token: Vec<Option<String>>,
    /// Tokens marked `special` in added_tokens (skipped on decode by default).
    special_ids: Vec<bool>,
    byte_map: HashMap<char, u8>,
    /// True when decoder / pre_tokenizer looks byte-level (Ġ / ByteLevel).
    byte_level: bool,
}

impl BundleTokenizer {
    /// Load from a bundle directory. Returns `Ok(None)` if no `tokenizer.json`.
    pub fn try_load(dir: &Path) -> Result<Option<Self>, EngineError> {
        let path = dir.join("tokenizer.json");
        if !path.is_file() {
            return Ok(None);
        }
        let raw = fs::read_to_string(&path).map_err(|e| EngineError::Io(e.to_string()))?;
        Self::from_tokenizer_json(&raw).map(Some)
    }

    pub fn from_tokenizer_json(raw: &str) -> Result<Self, EngineError> {
        let v: Value = serde_json::from_str(raw)
            .map_err(|e| EngineError::Format(format!("tokenizer.json: {e}")))?;

        let mut max_id: usize = 0;
        let mut pieces: HashMap<u32, String> = HashMap::new();
        let mut special: HashMap<u32, bool> = HashMap::new();

        if let Some(vocab) = v.pointer("/model/vocab").and_then(|x| x.as_object()) {
            for (tok, id_v) in vocab {
                let id = id_v.as_u64().ok_or_else(|| {
                    EngineError::Format(format!("tokenizer vocab id for {tok:?} not u64"))
                })? as u32;
                max_id = max_id.max(id as usize);
                pieces.insert(id, tok.clone());
            }
        }

        if let Some(added) = v.get("added_tokens").and_then(|x| x.as_array()) {
            for ent in added {
                let id = ent
                    .get("id")
                    .and_then(|x| x.as_u64())
                    .ok_or_else(|| EngineError::Format("added_tokens entry missing id".into()))?
                    as u32;
                let content = ent
                    .get("content")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| EngineError::Format("added_tokens entry missing content".into()))?
                    .to_string();
                let is_special = ent
                    .get("special")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false);
                max_id = max_id.max(id as usize);
                pieces.insert(id, content);
                special.insert(id, is_special);
            }
        }

        if pieces.is_empty() {
            return Err(EngineError::Format(
                "tokenizer.json has empty vocab / added_tokens".into(),
            ));
        }

        let mut id_to_token = vec![None; max_id + 1];
        let mut special_ids = vec![false; max_id + 1];
        for (id, tok) in pieces {
            let i = id as usize;
            id_to_token[i] = Some(tok);
            special_ids[i] = special.get(&id).copied().unwrap_or(false);
        }

        let byte_level = detect_byte_level(&v);

        Ok(Self {
            id_to_token,
            special_ids,
            byte_map: unicode_to_byte_map(),
            byte_level,
        })
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_opts(ids, true)
    }

    pub fn decode_opts(&self, ids: &[u32], skip_special: bool) -> String {
        let mut pieces = String::new();
        for &id in ids {
            let i = id as usize;
            if skip_special && self.special_ids.get(i).copied().unwrap_or(false) {
                continue;
            }
            match self.id_to_token.get(i).and_then(|t| t.as_ref()) {
                Some(tok) => pieces.push_str(tok),
                None => pieces.push_str(&format!("<{id}>")),
            }
        }
        if self.byte_level {
            byte_level_decode(&pieces, &self.byte_map)
        } else {
            // SentencePiece-style: ▁ → space
            pieces.replace('▁', " ")
        }
    }
}

fn detect_byte_level(v: &Value) -> bool {
    if v.pointer("/decoder/type")
        .and_then(|x| x.as_str())
        .is_some_and(|t| t.eq_ignore_ascii_case("ByteLevel"))
    {
        return true;
    }
    if let Some(arr) = v.get("pre_tokenizer").and_then(|p| {
        p.get("pretokenizers")
            .and_then(|x| x.as_array())
            .or_else(|| p.as_array())
    }) {
        for p in arr {
            if p.get("type")
                .and_then(|x| x.as_str())
                .is_some_and(|t| t.eq_ignore_ascii_case("ByteLevel"))
            {
                return true;
            }
        }
    }
    if v.pointer("/pre_tokenizer/type")
        .and_then(|x| x.as_str())
        .is_some_and(|t| t.eq_ignore_ascii_case("ByteLevel"))
    {
        return true;
    }
    // Heuristic: GPT/Qwen vocab uses Ġ for space.
    v.pointer("/model/vocab")
        .and_then(|x| x.as_object())
        .is_some_and(|vocab| vocab.keys().any(|k| k.contains('Ġ')))
}

fn byte_level_decode(pieces: &str, byte_map: &HashMap<char, u8>) -> String {
    let mut bytes = Vec::with_capacity(pieces.len());
    for ch in pieces.chars() {
        if let Some(&b) = byte_map.get(&ch) {
            bytes.push(b);
        } else {
            // Keep unknown unicode as UTF-8 bytes (e.g. already-decoded CJK pieces).
            let mut buf = [0u8; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
}

/// Fallback when no tokenizer sidecar: stable `<id>` placeholders (legacy demos).
pub fn decode_placeholders(ids: &[u32]) -> String {
    ids.iter().map(|t| format!("<{t}>")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_byte_level_json() -> String {
        // "Hello" as single piece + Ġworld → "Hello world" after byte-level decode.
        serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": {
                    "Hello": 1,
                    "Ġworld": 2,
                    "<|end|>": 3
                }
            },
            "added_tokens": [
                {"id": 3, "content": "<|end|>", "special": true}
            ],
            "decoder": {"type": "ByteLevel"},
            "pre_tokenizer": {"type": "ByteLevel"}
        })
        .to_string()
    }

    #[test]
    fn decodes_byte_level_with_space_marker() {
        let tok = BundleTokenizer::from_tokenizer_json(&minimal_byte_level_json()).unwrap();
        let s = tok.decode(&[1, 2, 3]);
        assert_eq!(s, "Hello world");
    }

    #[test]
    fn unknown_id_keeps_placeholder() {
        let tok = BundleTokenizer::from_tokenizer_json(&minimal_byte_level_json()).unwrap();
        let s = tok.decode_opts(&[1, 99], false);
        assert!(s.contains("<99>"), "{s}");
        assert!(s.starts_with("Hello"), "{s}");
    }

    #[test]
    fn sentencepiece_underline_to_space() {
        let raw = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": { "▁hi": 1, "▁there": 2 }
            }
        })
        .to_string();
        let tok = BundleTokenizer::from_tokenizer_json(&raw).unwrap();
        assert_eq!(tok.decode(&[1, 2]), " hi there");
    }

    #[test]
    fn try_load_missing_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(BundleTokenizer::try_load(dir.path()).unwrap().is_none());
    }
}
