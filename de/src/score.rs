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
use crate::shortlist::maybe_shortlist;
use crate::tokenizer::{encode_no_special, load_tokenizer, special_ids};

/// Logits producer for one packed sequence (candle `DecisionModel` or a stub).
pub trait EncoderForward: Send + Sync {
    fn forward_logits(
        &self,
        input_ids: &[u32],
        mask_positions: &[usize],
        type_id: usize,
    ) -> Result<Vec<f32>>;
}

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
    forward: Option<Arc<dyn EncoderForward>>,
}

impl EncoderScorer {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let checkpoint = EncoderCheckpoint::open(root)?;
        let tokenizer = load_tokenizer(&checkpoint.tokenizer_dir).ok();
        let forward = if checkpoint.has_weights() {
            Some(crate::model::forward::load_forward(&checkpoint)?)
        } else {
            None
        };
        Ok(Self {
            checkpoint,
            tokenizer,
            forward,
        })
    }

    /// Assemble a scorer from parts — lets upper layers (FFI / serve / SDK) test
    /// their success paths without shipping ModernBERT weights.
    #[doc(hidden)]
    pub fn from_parts(
        checkpoint: EncoderCheckpoint,
        tokenizer: Option<tokenizers::Tokenizer>,
        forward: Arc<dyn EncoderForward>,
    ) -> Self {
        Self {
            checkpoint,
            tokenizer,
            forward: Some(forward),
        }
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
        let packed = build_sequence_ids(&encode, cls, sep, mask, record, max_len, head_max_len);
        Ok((packed.input_ids, packed.mask_positions))
    }

    pub fn score_record_with_logits(&self, record: &Record, logits: &[f32]) -> Result<Value> {
        let rec = maybe_shortlist(record, None);
        score_record_logits(&rec, logits, DECISION_TEMPERATURE)
    }

    /// Pack + DecisionModel forward + softmax → System One answer.
    pub fn score_record(&self, record: &Record) -> Result<Value> {
        let forward = self.forward.as_ref().ok_or_else(|| {
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
        let logits = forward.forward_logits(&input_ids, &mask_positions, rec.task.type_id())?;
        score_record_logits(&rec, &logits, DECISION_TEMPERATURE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariacompute_core::packing::{OptionItem, Record};
    use ariacompute_core::typed::Task;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    fn fixture_tokenizer() -> tokenizers::Tokenizer {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../tests/fixtures/tokenizer_wordlevel.json");
        tokenizers::Tokenizer::from_file(path).unwrap()
    }

    fn checkpoint_with(root: &Path) -> EncoderCheckpoint {
        EncoderCheckpoint {
            root: root.to_path_buf(),
            config: crate::AgentConfig::default(),
            weights_path: root.join("model.safetensors"),
            tokenizer_dir: root.to_path_buf(),
            encoder_dir: root.to_path_buf(),
        }
    }

    struct StubForward {
        logits: Vec<f32>,
        calls: AtomicUsize,
    }

    impl EncoderForward for StubForward {
        fn forward_logits(
            &self,
            input_ids: &[u32],
            mask_positions: &[usize],
            type_id: usize,
        ) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(!input_ids.is_empty());
            assert!(!mask_positions.is_empty());
            assert!(type_id <= 2);
            Ok(self.logits.clone())
        }
    }

    fn choice_record() -> Record {
        Record::from_value(&json!({
            "id": "1",
            "task": "choice",
            "instructions": "pick",
            "options": [
                {"name": "a", "description": "alpha"},
                {"name": "b", "description": "beta"}
            ],
            "state": "hello"
        }))
        .unwrap()
    }

    #[test]
    fn logits_to_choice() {
        let rec = choice_record();
        let ans = score_record_logits(&rec, &[0.1, 2.0], 1.0).unwrap();
        assert_eq!(ans["choice"], "b");
    }

    #[test]
    fn logits_len_mismatch_is_an_error() {
        let rec = choice_record();
        let err = score_record_logits(&rec, &[0.1], 1.0).unwrap_err();
        assert!(err.to_string().contains("logits len 1 != options 2"));
    }

    #[test]
    fn open_without_weights_or_tokenizer() {
        let dir = tempdir().unwrap();
        let scorer = EncoderScorer::open(dir.path()).unwrap();
        let err = scorer.score_record(&choice_record()).unwrap_err();
        assert!(err.to_string().contains("encoder weights missing"));
        let err = scorer.pack_ids(&choice_record()).unwrap_err();
        assert!(err.to_string().contains("tokenizer not loaded"));
    }

    #[test]
    fn open_with_weights_surfaces_forward_load_errors() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"not safetensors").unwrap();
        let err = EncoderScorer::open(dir.path())
            .err()
            .expect("expected a forward load error");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn open_rejects_missing_dir() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope");
        let err = EncoderScorer::open(&missing)
            .err()
            .expect("expected missing checkpoint error");
        assert!(err.to_string().contains("not a directory"));
    }

    #[test]
    fn score_record_packs_and_uses_forward_logits() {
        let dir = tempdir().unwrap();
        let scorer = EncoderScorer::from_parts(
            checkpoint_with(dir.path()),
            Some(fixture_tokenizer()),
            Arc::new(StubForward {
                logits: vec![0.2, 3.0],
                calls: AtomicUsize::new(0),
            }),
        );
        let ans = scorer.score_record(&choice_record()).unwrap();
        assert_eq!(ans["choice"], "b");
        let (ids, markers) = scorer.pack_ids(&choice_record()).unwrap();
        assert_eq!(ids[0], 1); // [CLS]
        assert_eq!(markers.len(), 2);
    }

    #[test]
    fn score_record_needs_tokenizer_even_with_weights() {
        let dir = tempdir().unwrap();
        let scorer = EncoderScorer::from_parts(
            checkpoint_with(dir.path()),
            None,
            Arc::new(StubForward {
                logits: vec![1.0, 0.0],
                calls: AtomicUsize::new(0),
            }),
        );
        let err = scorer.score_record(&choice_record()).unwrap_err();
        assert!(err.to_string().contains("tokenizer not loaded"));
    }

    #[test]
    fn score_record_rejects_mask_option_mismatch() {
        let dir = tempdir().unwrap();
        // max_len 0 truncates every id away, so no MASK position survives.
        let mut ckpt = checkpoint_with(dir.path());
        ckpt.config.max_len = 0;
        let scorer = EncoderScorer::from_parts(
            ckpt,
            Some(fixture_tokenizer()),
            Arc::new(StubForward {
                logits: vec![1.0, 0.0],
                calls: AtomicUsize::new(0),
            }),
        );
        let err = scorer.score_record(&choice_record()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("mask positions 0 != options 2") || msg.contains("no MASK"),
            "{msg}"
        );
    }

    #[test]
    fn score_record_propagates_forward_error() {
        struct Failing;
        impl EncoderForward for Failing {
            fn forward_logits(
                &self,
                _ids: &[u32],
                _mask: &[usize],
                _type_id: usize,
            ) -> Result<Vec<f32>> {
                Err(AfmError::msg("boom"))
            }
        }
        let dir = tempdir().unwrap();
        let scorer = EncoderScorer::from_parts(
            checkpoint_with(dir.path()),
            Some(fixture_tokenizer()),
            Arc::new(Failing),
        );
        let err = scorer.score_record(&choice_record()).unwrap_err();
        assert_eq!(err.to_string(), "boom");
    }

    #[test]
    fn score_record_with_logits_shortlists_large_choice() {
        let dir = tempdir().unwrap();
        let scorer = EncoderScorer::open(dir.path()).unwrap();
        let options: Vec<OptionItem> = (0..50)
            .map(|i| OptionItem {
                name: format!("o{i}"),
                description: String::new(),
            })
            .collect();
        let rec = Record {
            id: "big".into(),
            task: Task::Choice,
            instructions: "pick".into(),
            options,
            state: json!("s"),
            label: None,
        };
        let logits = vec![0.0_f32; 50];
        let err = scorer.score_record_with_logits(&rec, &logits).unwrap_err();
        assert!(err.to_string().contains("logits len 50 != options 20"));
        let ans = scorer
            .score_record_with_logits(&rec, &[0.0_f32; 20])
            .unwrap();
        assert!(ans["choice"].as_str().unwrap().starts_with('o'));
    }
}
