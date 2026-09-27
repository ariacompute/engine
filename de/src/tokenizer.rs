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
