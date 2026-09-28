//! Encoder scoring: pack → candle DecisionModel forward → typed answer.

use std::path::Path;
use std::sync::Arc;

use ariacompute_core::contract::DECISION_TEMPERATURE;
use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::packing::{build_sequence_ids, Record};
use ariacompute_core::systemone::answer_from_probs;
use ariacompute_core::typed::softmax;
use serde_json::Value;

use crate::checkpoint::EncoderCheckpoint;
use crate::model::DecisionModel;
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

/// Encoder scorer with optional loaded candle DecisionModel.
pub struct EncoderScorer {
    pub checkpoint: EncoderCheckpoint,
    tokenizer: Option<tokenizers::Tokenizer>,
    model: Option<Arc<DecisionModel>>,
}

impl EncoderScorer {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let checkpoint = EncoderCheckpoint::open(root)?;
        let tokenizer = load_tokenizer(&checkpoint.tokenizer_dir).ok();
        let model = if checkpoint.has_weights() {
            let head_layers = checkpoint.config.head_layers;
            Some(Arc::new(DecisionModel::load(
                &checkpoint.encoder_dir,
                &checkpoint.weights_path,
                head_layers,
            )?))
        } else {
            None
        };
        Ok(Self {
            checkpoint,
            tokenizer,
            model,
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

    /// Pack + DecisionModel forward + softmax → System One answer.
    pub fn score_record(&self, record: &Record) -> Result<Value> {
        let model = self.model.as_ref().ok_or_else(|| {
            AfmError::msg(format!(
                "encoder weights missing at {}; provide model.safetensors or use score_record_with_logits",
                self.checkpoint.weights_path.display()
            ))
        })?;
        if self.tokenizer.is_none() {
            return Err(AfmError::msg("tokenizer not loaded for packing"));
        }
        let rec = maybe_shortlist(record, None);
        let (input_ids, mask_positions) = self.pack_ids(&rec)?;
        if mask_positions.len() != rec.options.len() {
            return Err(AfmError::msg(format!(
                "mask positions {} != options {}",
                mask_positions.len(),
                rec.options.len()
            )));
        }
        let logits = model.forward_logits(&input_ids, &mask_positions, rec.task.type_id())?;
        score_record_logits(&rec, &logits, DECISION_TEMPERATURE)
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
