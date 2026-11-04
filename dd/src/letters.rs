//! Tokenizer plumbing + letter logit selection for the MiniCPM SemIf forward.

use std::path::Path;

use candle_core::{IndexOp, Tensor};
use tokenizers::Tokenizer;

use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::typed::softmax;

use crate::checkpoint::DecoderCheckpoint;
use crate::semif::LETTERS;
use crate::weights::candle_err;

/// Last position logits of (logits,) / (V,) / (1, V) / (1, L, V) tensors.
pub(crate) fn last_token_logits(logits: &Tensor) -> Result<Vec<f32>> {
    let dims = logits.dims();
    let t = match dims.len() {
        1 => logits.clone(),
        2 => logits.i(0).map_err(candle_err)?,
        3 => {
            let seq = dims[1];
            logits.i((0, seq - 1)).map_err(candle_err)?
        }
        n => {
            return Err(AfmError::msg(format!(
                "MiniCPM logits rank {n} unsupported"
            )))
        }
    };
    t.to_vec1::<f32>().map_err(candle_err)
}

pub(crate) fn encode_prompt(
    tok: &Tokenizer,
    prompt: &str,
    bos_id: Option<u32>,
) -> Result<Vec<u32>> {
    let enc = tok
        .encode(prompt, false)
        .map_err(|e| AfmError::msg(format!("tokenize MiniCPM prompt: {e}")))?;
    let mut ids = enc.get_ids().to_vec();
    if let Some(bos) = bos_id {
        if ids.first().copied() != Some(bos) {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
}

pub(crate) fn letter_token_ids(tok: &Tokenizer) -> Result<[u32; 16]> {
    let mut out = [0u32; 16];
    for (i, ch) in LETTERS.chars().enumerate() {
        let s = ch.to_string();
        if let Some(id) = tok.token_to_id(&s) {
            out[i] = id;
            continue;
        }
        let enc = tok
            .encode(s.as_str(), false)
            .map_err(|e| AfmError::msg(format!("tokenize letter {s}: {e}")))?;
        let ids = enc.get_ids();
        if ids.len() != 1 {
            return Err(AfmError::msg(format!(
                "letter {s} is not a single token ({} ids)",
                ids.len()
            )));
        }
        out[i] = ids[0];
    }
    Ok(out)
}

pub(crate) fn load_tokenizer(ckpt: &DecoderCheckpoint, base: &Path) -> Result<Tokenizer> {
    let mut dirs = Vec::new();
    if let Some(root) = &ckpt.root {
        dirs.push(root.clone());
        // a PEFT adapter usually keeps its own tokenizer.json
        dirs.extend(ckpt.adapter_dir.iter().filter(|ad| *ad != root).cloned());
    }
    dirs.push(base.to_path_buf());
    for dir in dirs {
        for rel in ["tokenizer.json", "tokenizer/tokenizer.json"] {
            let path = dir.join(rel);
            if path.is_file() {
                return Tokenizer::from_file(&path)
                    .map_err(|e| AfmError::msg(format!("tokenizer load {}: {e}", path.display())));
            }
        }
    }
    Err(AfmError::msg(format!(
        "tokenizer.json not found next to decoder checkpoint or {}",
        base.display()
    )))
}

/// Pick letter logits out of the last-token distribution and softmax them.
pub(crate) fn letter_probs(
    last: &[f32],
    letter_ids: &[u32; 16],
    options: &[serde_json::Value],
) -> Result<(Vec<String>, Vec<f32>)> {
    let n = options.len();
    let mut letter_logits = Vec::with_capacity(n);
    let mut option_ids = Vec::with_capacity(n);
    for (i, opt) in options.iter().enumerate() {
        let tid = letter_ids[i] as usize;
        let logit = last
            .get(tid)
            .copied()
            .ok_or_else(|| AfmError::msg(format!("letter logit OOB token {tid}")))?;
        letter_logits.push(logit);
        let id = opt
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        option_ids.push(id);
    }
    let probs = softmax(&letter_logits, 1.0);
    Ok((option_ids, probs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    const FIXTURE: &str = include_str!("../../tests/fixtures/tokenizer_wordlevel.json");

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_bytes(FIXTURE.as_bytes()).unwrap()
    }

    fn write_tokenizer(dir: &Path) -> &Path {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("tokenizer.json"), FIXTURE).unwrap();
        dir
    }

    #[test]
    fn last_token_logits_handles_ranks() {
        let dev = Device::Cpu;
        let flat = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &dev).unwrap();
        assert_eq!(last_token_logits(&flat).unwrap(), vec![1.0, 2.0]);

        let two = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &dev).unwrap();
        assert_eq!(last_token_logits(&two).unwrap(), vec![1.0, 2.0]);

        let three =
            Tensor::from_vec(vec![9.0f32, 3.0, 4.0, 5.0, 6.0, 7.0], (1, 3, 2), &dev).unwrap();
        assert_eq!(last_token_logits(&three).unwrap(), vec![6.0, 7.0]);

        let four = Tensor::zeros((1, 1, 1, 1), DType::F32, &dev).unwrap();
        let err = last_token_logits(&four).unwrap_err();
        assert!(err.to_string().contains("rank 4 unsupported"));
    }

    #[test]
    fn encode_prompt_inserts_bos_once() {
        let tok = tokenizer();
        // fixture starts with "<s>" (id 5) when encoded as text
        let ids = encode_prompt(&tok, "<s> choice", Some(5)).unwrap();
        assert_eq!(ids[0], 5);
        assert_eq!(ids.iter().filter(|id| **id == 5).count(), 1);

        let ids = encode_prompt(&tok, "choice", Some(5)).unwrap();
        assert_eq!(ids, vec![5, 23]);

        let ids = encode_prompt(&tok, "choice", None).unwrap();
        assert_eq!(ids, vec![23]);
    }

    #[test]
    fn encode_prompt_reports_tokenizer_failure() {
        let tok = Tokenizer::from_bytes(
            json!({
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "model": {"type": "WordLevel", "vocab": {"known": 0}, "unk_token": "[UNK]"}
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let err = encode_prompt(&tok, "unknown", None).unwrap_err();
        assert!(err.to_string().contains("tokenize MiniCPM prompt"));
    }

    #[test]
    fn letter_ids_come_from_vocab_or_single_token_encode() {
        let tok = tokenizer();
        let ids = letter_token_ids(&tok).unwrap();
        assert_eq!(ids[0], 7); // "A"
        assert_eq!(ids[15], 22); // "P"

        // letters absent from the vocab still collapse to the unk id
        let unk_tok = Tokenizer::from_bytes(
            json!({
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let ids = letter_token_ids(&unk_tok).unwrap();
        assert_eq!(ids, [0u32; 16]);
    }

    #[test]
    fn letter_must_be_a_single_token() {
        // "A" is absent from the vocab, and the Replace normalizer turns it into
        // "A A" so the fallback encode yields two ids.
        let tok = Tokenizer::from_bytes(
            json!({
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": {"type": "Replace", "pattern": {"String": "A"}, "content": "A A"},
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": null,
                "decoder": null,
                "model": {"type": "WordLevel", "vocab": {"X": 0, "[UNK]": 1}, "unk_token": "[UNK]"}
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let err = letter_token_ids(&tok).unwrap_err();
        assert!(err.to_string().contains("is not a single token"));
    }

    #[test]
    fn load_tokenizer_searches_root_adapter_and_base() {
        let root = tempdir().unwrap();
        write_tokenizer(root.path());
        let ckpt = DecoderCheckpoint {
            root: Some(root.path().to_path_buf()),
            config: crate::checkpoint::DdConfig::default(),
            adapter_dir: Some(root.path().to_path_buf()),
        };
        assert!(load_tokenizer(&ckpt, root.path()).is_ok());

        // tokenizer only inside the adapter dir
        let with_adapter = tempdir().unwrap();
        let adapter = with_adapter.path().join("adapter");
        write_tokenizer(&adapter);
        let ckpt = DecoderCheckpoint {
            root: Some(with_adapter.path().to_path_buf()),
            config: crate::checkpoint::DdConfig::default(),
            adapter_dir: Some(adapter),
        };
        assert!(load_tokenizer(&ckpt, with_adapter.path()).is_ok());

        // tokenizer only next to the base snapshot
        let base = tempdir().unwrap();
        write_tokenizer(&base.path().join("tokenizer"));
        let ckpt = DecoderCheckpoint {
            root: None,
            config: crate::checkpoint::DdConfig::default(),
            adapter_dir: None,
        };
        assert!(load_tokenizer(&ckpt, base.path()).is_ok());

        // nothing anywhere
        let empty = tempdir().unwrap();
        let err = load_tokenizer(&ckpt, empty.path()).unwrap_err();
        assert!(err.to_string().contains("tokenizer.json not found"));
    }

    #[test]
    fn letter_probs_selects_and_softmaxes() {
        let mut last = vec![0.0f32; 32];
        last[7] = 1.0; // A
        last[8] = 3.0; // B
        let letter_ids = letter_token_ids(&tokenizer()).unwrap();
        let (ids, probs) = letter_probs(
            &last,
            &letter_ids,
            &[json!({"id": "x"}), json!({"id": "y"})],
        )
        .unwrap();
        assert_eq!(ids, vec!["x".to_string(), "y".to_string()]);
        assert!(probs[1] > probs[0]);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn letter_probs_reports_short_distribution_and_missing_id() {
        let letter_ids = letter_token_ids(&tokenizer()).unwrap();
        let err = letter_probs(&[0.0f32; 2], &letter_ids, &[json!({"id": "x"})]).unwrap_err();
        assert!(err.to_string().contains("letter logit OOB"));

        let mut last = vec![1.0f32; 32];
        last[7] = 2.0;
        let (ids, _) = letter_probs(&last, &letter_ids, &[json!({})]).unwrap();
        assert_eq!(ids, vec![String::new()]);
    }
}
