use std::path::Path;

use ariacompute_core::error::{AfmError, Result};
use tokenizers::Tokenizer;

pub fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
    let candidates = [
        dir.join("tokenizer.json"),
        dir.join("tokenizer/tokenizer.json"),
    ];
    for path in &candidates {
        if path.is_file() {
            return Tokenizer::from_file(path)
                .map_err(|e| AfmError::msg(format!("tokenizer load {}: {e}", path.display())));
        }
    }
    Err(AfmError::msg(format!(
        "tokenizer.json not found under {}",
        dir.display()
    )))
}

pub fn encode_no_special(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| AfmError::msg(format!("tokenize: {e}")))?;
    Ok(enc.get_ids().to_vec())
}

pub fn special_ids(tok: &Tokenizer) -> Result<(u32, u32, u32)> {
    let vocab = tok.get_vocab(true);
    let cls = *vocab
        .get("[CLS]")
        .or_else(|| vocab.get("<s>"))
        .or_else(|| vocab.get("<cls>"))
        .ok_or_else(|| AfmError::msg("tokenizer missing CLS token"))?;
    let sep = *vocab
        .get("[SEP]")
        .or_else(|| vocab.get("</s>"))
        .or_else(|| vocab.get("<sep>"))
        .ok_or_else(|| AfmError::msg("tokenizer missing SEP token"))?;
    let mask = *vocab
        .get("[MASK]")
        .or_else(|| vocab.get("<mask>"))
        .ok_or_else(|| AfmError::msg("tokenizer missing MASK token"))?;
    Ok((cls, sep, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    const FIXTURE: &str = include_str!("../../tests/fixtures/tokenizer_wordlevel.json");

    fn write(dir: &Path, rel: &str, body: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    fn build(vocab: &[(&str, u32)], unk: Option<&str>) -> Tokenizer {
        let map: serde_json::Map<String, serde_json::Value> = vocab
            .iter()
            .map(|(k, v)| ((*k).to_string(), json!(v)))
            .collect();
        let doc = json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": map, "unk_token": unk}
        });
        Tokenizer::from_bytes(doc.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn loads_root_and_nested_tokenizer() {
        let dir = tempdir().unwrap();
        write(dir.path(), "tokenizer.json", FIXTURE);
        assert!(load_tokenizer(dir.path()).is_ok());

        let nested = tempdir().unwrap();
        write(nested.path(), "tokenizer/tokenizer.json", FIXTURE);
        assert!(load_tokenizer(nested.path()).is_ok());
    }

    #[test]
    fn missing_tokenizer_is_an_error() {
        let dir = tempdir().unwrap();
        let err = load_tokenizer(dir.path()).unwrap_err();
        assert!(err.to_string().contains("tokenizer.json not found"));
    }

    #[test]
    fn broken_tokenizer_json_is_an_error() {
        let dir = tempdir().unwrap();
        write(dir.path(), "tokenizer.json", "{not a tokenizer}");
        let err = load_tokenizer(dir.path()).unwrap_err();
        assert!(err.to_string().contains("tokenizer load"));
    }

    #[test]
    fn encodes_words_without_special_tokens() {
        let tok = load_tokenizer_from_fixture();
        let ids = encode_no_special(&tok, "choice pick").unwrap();
        assert_eq!(ids, vec![23, 27]);
    }

    #[test]
    fn unknown_word_without_unk_token_errors() {
        // `unk_token` points at a vocab entry that does not exist → encode fails.
        let tok = build(&[("known", 0)], Some("[UNK]"));
        let err = encode_no_special(&tok, "unknown").unwrap_err();
        assert!(err.to_string().contains("tokenize:"));
    }

    #[test]
    fn special_ids_prefers_square_bracket_tokens() {
        let tok = load_tokenizer_from_fixture();
        assert_eq!(special_ids(&tok).unwrap(), (1, 2, 3));
    }

    #[test]
    fn special_ids_falls_back_to_sentencepiece_names() {
        let tok = build(&[("<s>", 1), ("</s>", 2), ("<mask>", 3)], Some("<s>"));
        assert_eq!(special_ids(&tok).unwrap(), (1, 2, 3));
    }

    #[test]
    fn special_ids_reports_missing_tokens() {
        let tok = build(&[("</s>", 2), ("<mask>", 3)], Some("</s>"));
        assert!(special_ids(&tok)
            .unwrap_err()
            .to_string()
            .contains("missing CLS"));

        let tok = build(&[("[CLS]", 1), ("<mask>", 3)], Some("[CLS]"));
        assert!(special_ids(&tok)
            .unwrap_err()
            .to_string()
            .contains("missing SEP"));

        let tok = build(&[("[CLS]", 1), ("[SEP]", 2)], Some("[CLS]"));
        assert!(special_ids(&tok)
            .unwrap_err()
            .to_string()
            .contains("missing MASK"));
    }

    fn load_tokenizer_from_fixture() -> Tokenizer {
        let dir = tempdir().unwrap();
        write(dir.path(), "tokenizer.json", FIXTURE);
        load_tokenizer(dir.path()).unwrap()
    }
}
