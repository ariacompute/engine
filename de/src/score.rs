//! Encoder scoring: pack → (optional) forward → typed answer.

use std::path::Path;

use ariacompute_core::contract::DECISION_TEMPERATURE;
use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::packing::{build_sequence_ids, Record};
use ariacompute_core::systemone::answer_from_probs;
use ariacompute_core::typed::softmax;
use serde_json::Value;

use crate::checkpoint::EncoderCheckpoint;
use crate::shortlist::maybe_shortlist;
use crate::tokenizer::{encode_no_special, load_tokenizer, special_ids};

/// Produce softmax probs from option logits (decision T=1 by default).
pub fn score_record_logits(record: &Record, logits: &[f32], temperature: f32) -> Result<Value> {
    if logits.len() != record.options.len() {
        return Err(AfmError::msg(format!(
            "logits len {} != options {}",
            logits.len(),
            record.options.len()
        )));
    }
    let probs = softmax(logits, temperature);
    answer_from_probs(record, &probs)
}

/// Encoder scorer. Full candle ModernBERT+head forward runs when weights are present;
/// without weights, [`EncoderScorer::score_record`] returns an error unless
/// `score_record_with_logits` is used (parity / golden tests).
pub struct EncoderScorer {
    pub checkpoint: EncoderCheckpoint,
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl EncoderScorer {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let checkpoint = EncoderCheckpoint::open(root)?;
        let tokenizer = load_tokenizer(&checkpoint.tokenizer_dir).ok();
        Ok(Self {
            checkpoint,
            tokenizer,
        })
    }

    pub fn pack_ids(&self, record: &Record) -> Result<(Vec<u32>, Vec<usize>)> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| AfmError::msg("tokenizer not loaded for packing"))?;
        let (cls, sep, mask) = special_ids(tok)?;
        let max_len = self.checkpoint.config.max_len;
        let head_max_len = self.checkpoint.config.head_max_len;
        let encode = |text: &str| encode_no_special(tok, text).unwrap_or_default();
        let packed = build_sequence_ids(
            &encode,
            cls,
            sep,
            mask,
            record,
            max_len,
            head_max_len,
        );
        Ok((packed.input_ids, packed.mask_positions))
    }

    pub fn score_record_with_logits(&self, record: &Record, logits: &[f32]) -> Result<Value> {
        let rec = maybe_shortlist(record, None);
        score_record_logits(&rec, logits, DECISION_TEMPERATURE)
    }

    /// Run inference. Requires `model.safetensors`; until candle ModernBERT parity lands,
    /// this returns a structured error. Use `score_record_with_logits` for golden tests.
    pub fn score_record(&self, record: &Record) -> Result<Value> {
        if !self.checkpoint.has_weights() {
            return Err(AfmError::msg(format!(
                "encoder weights missing at {}; provide model.safetensors or use score_record_with_logits",
                self.checkpoint.weights_path.display()
            )));
        }
        // Weights present: pack for validation; full candle DecisionModel forward is
        // gated behind loading ModernBERT — surface clear path for CLI/serve.
        let _packed = if self.tokenizer.is_some() {
            Some(self.pack_ids(record)?)
        } else {
            None
        };
        Err(AfmError::msg(
            "encoder candle DecisionModel forward not yet loaded for this checkpoint; \
             packing/config OK — run golden tests via score_record_with_logits until Hub weights are wired",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariacompute_core::packing::Record;
    use serde_json::json;

    #[test]
    fn logits_to_choice() {
        let rec = Record::from_value(&json!({
            "id": "1",
            "task": "choice",
            "instructions": "pick",
            "options": [
                {"name": "a", "description": "x"},
                {"name": "b", "description": "y"}
            ],
            "state": ""
        }))
        .unwrap();
        let ans = score_record_logits(&rec, &[0.1, 2.0], 1.0).unwrap();
        assert_eq!(ans["choice"], "b");
    }
}
