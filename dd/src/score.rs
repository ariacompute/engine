//! Decoder scoring map + scorer shell (MiniCPM forward when weights available).

use std::path::Path;
use std::sync::{Arc, Mutex};

use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::packing::Record;
use ariacompute_core::systemone::answer_systemone_dd;
use ariacompute_core::typed::Task;
use serde_json::{json, Map, Value};

use crate::checkpoint::DecoderCheckpoint;
use crate::forward::MiniCpmForward;
use crate::semif::record_to_semif_row;

/// First-token letter scorer (candle MiniCPM forward or a stub).
pub trait LetterScorer: Send {
    fn letter_logits(&mut self, row: &Value) -> Result<(Vec<String>, Vec<f32>)>;
}

/// Map SemIf option_ids/probabilities onto AFM-D option names + argmax label.
pub fn probs_from_semif_out(record: &Record, out: &Value) -> Result<(Map<String, Value>, String)> {
    let option_ids = out
        .get("option_ids")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AfmError::msg("semif out missing option_ids"))?;
    let probabilities = out
        .get("probabilities")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AfmError::msg("semif out missing probabilities"))?;
    let mut probs = Map::new();
    for (k, v) in option_ids.iter().zip(probabilities.iter()) {
        let key = k.as_str().unwrap_or("").to_string();
        let p = v.as_f64().unwrap_or(0.0);
        probs.insert(key, json!(p));
    }
    let mut label = probs
        .iter()
        .max_by(|a, b| {
            a.1.as_f64()
                .unwrap_or(0.0)
                .partial_cmp(&b.1.as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(k, _)| k.clone())
        .unwrap_or_default();

    match record.task {
        Task::Noul => {
            let p_true = probs
                .get("true")
                .or_else(|| probs.get("yes"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let mut mapped = Map::new();
            for opt in &record.options {
                let name = &opt.name;
                if name == "true" || name == "yes" {
                    mapped.insert(name.clone(), json!(p_true));
                } else if name == "false" || name == "no" {
                    mapped.insert(name.clone(), json!(1.0 - p_true));
                }
            }
            // packing canonicalises noul options to false/true, so `mapped` is
            // never empty for a valid record.
            probs = mapped;
            label = probs
                .iter()
                .max_by(|a, b| {
                    a.1.as_f64()
                        .unwrap_or(0.0)
                        .partial_cmp(&b.1.as_f64().unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone())
                .unwrap_or(label);
        }
        Task::Score => {
            let mut mapped = Map::new();
            for (i, opt) in record.options.iter().enumerate() {
                let p = probs
                    .get(&i.to_string())
                    .or_else(|| probs.get(&opt.name))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                mapped.insert(opt.name.clone(), json!(p));
            }
            let total: f64 = mapped.values().filter_map(|v| v.as_f64()).sum();
            if total > 0.0 {
                for v in mapped.values_mut() {
                    if let Some(p) = v.as_f64() {
                        *v = json!(p / total);
                    }
                }
            }
            probs = mapped;
            label = probs
                .iter()
                .max_by(|a, b| {
                    a.1.as_f64()
                        .unwrap_or(0.0)
                        .partial_cmp(&b.1.as_f64().unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone())
                .unwrap_or(label);
        }
        Task::Choice => {
            let mut mapped = Map::new();
            for opt in &record.options {
                let p = probs.get(&opt.name).and_then(|v| v.as_f64()).unwrap_or(0.0);
                mapped.insert(opt.name.clone(), json!(p));
            }
            let total: f64 = mapped.values().filter_map(|v| v.as_f64()).sum();
            if total > 0.0 {
                for v in mapped.values_mut() {
                    if let Some(p) = v.as_f64() {
                        *v = json!(p / total);
                    }
                }
            }
            probs = mapped;
            label = probs
                .iter()
                .max_by(|a, b| {
                    a.1.as_f64()
                        .unwrap_or(0.0)
                        .partial_cmp(&b.1.as_f64().unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k.clone())
                .unwrap_or(label);
        }
    }
    Ok((probs, label))
}

/// Build the candle MiniCPM forward (default `DecoderScorer` factory).
fn load_default(ckpt: &DecoderCheckpoint) -> Result<Box<dyn LetterScorer + Send>> {
    Ok(Box::new(MiniCpmForward::load(ckpt)?))
}

type LetterFactory =
    dyn Fn(&DecoderCheckpoint) -> Result<Box<dyn LetterScorer + Send>> + Send + Sync;

pub struct DecoderScorer {
    pub checkpoint: DecoderCheckpoint,
    inner: Mutex<Option<Box<dyn LetterScorer + Send>>>,
    factory: Arc<LetterFactory>,
}

impl DecoderScorer {
    pub fn open(checkpoint: Option<impl AsRef<Path>>) -> Result<Self> {
        Ok(Self {
            checkpoint: DecoderCheckpoint::open(checkpoint)?,
            inner: Mutex::new(None),
            factory: Arc::new(load_default),
        })
    }

    /// Assemble a scorer with a ready-made letter scorer (upper-layer tests).
    #[doc(hidden)]
    pub fn from_parts(checkpoint: DecoderCheckpoint, scorer: Box<dyn LetterScorer + Send>) -> Self {
        Self {
            checkpoint,
            inner: Mutex::new(Some(scorer)),
            factory: Arc::new(load_default),
        }
    }

    /// Assemble a scorer that builds its letter scorer lazily (default path).
    #[doc(hidden)]
    pub fn with_factory(checkpoint: DecoderCheckpoint, factory: Arc<LetterFactory>) -> Self {
        Self {
            checkpoint,
            inner: Mutex::new(None),
            factory,
        }
    }

    /// Score from a SemIf-style out dict (golden / parity path).
    pub fn score_from_semif_out(&self, record: &Record, out: &Value) -> Result<Value> {
        let _row = record_to_semif_row(record)?;
        let (probs, label) = probs_from_semif_out(record, out)?;
        Ok(json!({
            "probs": probs,
            "label": label,
            "confidence": probs.values().filter_map(|v| v.as_f64()).fold(0.0_f64, f64::max),
            "semif": out,
            "systemone": answer_systemone_dd(record, &probs, &label),
        }))
    }

    pub fn score_record(&self, record: &Record) -> Result<Value> {
        let row = record_to_semif_row(record)?;
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some((self.factory)(&self.checkpoint)?);
        }
        let fwd = guard.as_mut().expect("MiniCPM forward just loaded");
        let (option_ids, probabilities) = fwd.letter_logits(&row)?;
        let out = json!({
            "option_ids": option_ids,
            "probabilities": probabilities,
        });
        self.score_from_semif_out(record, &out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariacompute_core::packing::Record;

    fn rec(task: &str, options: Value) -> Record {
        Record::from_value(&json!({
            "id": "1",
            "task": task,
            "options": options,
            "state": "",
            "instructions": "x"
        }))
        .unwrap()
    }

    fn choice_rec() -> Record {
        rec(
            "choice",
            json!([
                {"name": "a", "description": ""},
                {"name": "b", "description": ""}
            ]),
        )
    }

    #[test]
    fn map_choice_probs() {
        let r = choice_rec();
        let out = json!({
            "option_ids": ["a", "b"],
            "probabilities": [0.2, 0.8]
        });
        let (probs, label) = probs_from_semif_out(&r, &out).unwrap();
        assert_eq!(label, "b");
        assert!((probs["b"].as_f64().unwrap() - 0.8).abs() < 1e-9);
        let scored = DecoderScorer::open(None::<&str>)
            .unwrap()
            .score_from_semif_out(&r, &out)
            .unwrap();
        assert_eq!(scored["systemone"]["choice"], "b");
    }

    #[test]
    fn map_choice_without_known_ids_and_zero_total() {
        let r = choice_rec();
        let out = json!({"option_ids": ["zzz"], "probabilities": ["not-a-number"]});
        let (probs, label) = probs_from_semif_out(&r, &out).unwrap();
        // every option sits at 0.0 → max_by keeps the last tie
        assert_eq!(label, "b");
        assert_eq!(probs.len(), 2);

        // non-string ids collapse onto "" and are dropped by the choice remap
        let out = json!({"option_ids": [1], "probabilities": [0.5]});
        let (probs, label) = probs_from_semif_out(&r, &out).unwrap();
        assert_eq!(label, "b");
        assert_eq!(probs["a"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn missing_fields_are_reported() {
        let r = choice_rec();
        let err = probs_from_semif_out(&r, &json!({"probabilities": [0.5]})).unwrap_err();
        assert!(err.to_string().contains("missing option_ids"));
        let err = probs_from_semif_out(&r, &json!({"option_ids": ["a"]})).unwrap_err();
        assert!(err.to_string().contains("missing probabilities"));
    }

    #[test]
    fn noul_maps_true_false_and_yes_no() {
        let r = rec(
            "noul",
            json!([
                {"name": "false", "description": ""},
                {"name": "true", "description": ""}
            ]),
        );
        let out = json!({"option_ids": ["false", "true"], "probabilities": [0.25, 0.75]});
        let (probs, label) = probs_from_semif_out(&r, &out).unwrap();
        assert_eq!(label, "true");
        assert!((probs["false"].as_f64().unwrap() - 0.25).abs() < 1e-9);

        // aliases: yes/no record options are canonicalised to true/false
        let alias = rec(
            "noul",
            json!([
                {"name": "no", "description": ""},
                {"name": "yes", "description": ""}
            ]),
        );
        assert_eq!(alias.options[0].name, "false");
        let out = json!({"option_ids": ["yes", "no"], "probabilities": [0.9, 0.1]});
        let (probs, label) = probs_from_semif_out(&alias, &out).unwrap();
        assert_eq!(label, "true");
        assert!((probs["false"].as_f64().unwrap() - 0.1).abs() < 1e-9);

        // unknown option names are canonicalised away from the noul mapping, and
        // with no true/false mass the false side wins
        let plain = rec(
            "noul",
            json!([
                {"name": "maybe", "description": ""},
                {"name": "sure", "description": ""}
            ]),
        );
        assert_eq!(plain.options.len(), 4);
        let out = json!({"option_ids": ["maybe", "sure"], "probabilities": [0.3, 0.7]});
        let (probs, label) = probs_from_semif_out(&plain, &out).unwrap();
        assert_eq!(label, "false");
        assert_eq!(probs.len(), 2);
        assert!((probs["true"].as_f64().unwrap()).abs() < 1e-9);
    }

    #[test]
    fn score_maps_numeric_and_named_ids_and_normalizes() {
        let r = rec(
            "score",
            json!([
                {"name": "level-0", "description": "low"},
                {"name": "level-1", "description": "high"}
            ]),
        );
        let out = json!({"option_ids": ["0", "1"], "probabilities": [1.0, 3.0]});
        let (probs, label) = probs_from_semif_out(&r, &out).unwrap();
        assert_eq!(label, "level-1");
        assert!((probs["level-0"].as_f64().unwrap() - 0.25).abs() < 1e-9);

        // name-keyed ids are accepted as a fallback
        let out = json!({"option_ids": ["level-1"], "probabilities": [2.0]});
        let (probs, label) = probs_from_semif_out(&r, &out).unwrap();
        assert_eq!(label, "level-1");
        assert_eq!(probs["level-0"].as_f64().unwrap(), 0.0);

        // zero total leaves the raw values untouched
        let out = json!({"option_ids": ["0"], "probabilities": [0.0]});
        let (probs, _) = probs_from_semif_out(&r, &out).unwrap();
        assert_eq!(probs["level-0"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn score_from_semif_out_builds_systemone_answer() {
        let r = rec(
            "score",
            json!([
                {"name": "level-0", "description": ""},
                {"name": "level-1", "description": ""}
            ]),
        );
        let scorer = DecoderScorer::open(None::<&str>).unwrap();
        let out = json!({"option_ids": ["0", "1"], "probabilities": [0.1, 0.9]});
        let scored = scorer.score_from_semif_out(&r, &out).unwrap();
        assert_eq!(scored["systemone"]["choice"], "level-1");
        assert!((scored["confidence"].as_f64().unwrap() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn noul_systemone_answer_and_single_option_error() {
        let r = rec(
            "noul",
            json!([
                {"name": "false", "description": ""},
                {"name": "true", "description": ""}
            ]),
        );
        let scorer = DecoderScorer::open(None::<&str>).unwrap();
        let out = json!({"option_ids": ["true"], "probabilities": [0.6]});
        let scored = scorer.score_from_semif_out(&r, &out).unwrap();
        assert!((scored["systemone"]["noul"].as_f64().unwrap() - 0.6).abs() < 1e-9);

        let single = rec("choice", json!([{"name": "only", "description": ""}]));
        let err = scorer.score_from_semif_out(&single, &out).unwrap_err();
        assert!(err.to_string().contains("need 2..16 options"));
    }

    struct StubScorer {
        ids: Vec<String>,
        probs: Vec<f32>,
        calls: usize,
    }

    impl LetterScorer for StubScorer {
        fn letter_logits(&mut self, row: &Value) -> Result<(Vec<String>, Vec<f32>)> {
            assert!(row.get("options").is_some());
            self.calls += 1;
            Ok((self.ids.clone(), self.probs.clone()))
        }
    }

    #[test]
    fn score_record_uses_ready_made_letter_scorer() {
        let scorer = DecoderScorer::from_parts(
            DecoderCheckpoint::open(None::<&str>).unwrap(),
            Box::new(StubScorer {
                ids: vec!["a".into(), "b".into()],
                probs: vec![0.1, 0.9],
                calls: 0,
            }),
        );
        let scored = scorer.score_record(&choice_rec()).unwrap();
        assert_eq!(scored["label"], "b");
    }

    #[test]
    fn score_record_builds_letters_lazily_from_factory() {
        let scorer = DecoderScorer::with_factory(
            DecoderCheckpoint::open(None::<&str>).unwrap(),
            Arc::new(|_ckpt: &DecoderCheckpoint| {
                Ok(Box::new(StubScorer {
                    ids: vec!["a".into(), "b".into()],
                    probs: vec![0.7, 0.3],
                    calls: 0,
                }) as Box<dyn LetterScorer + Send>)
            }),
        );
        let scored = scorer.score_record(&choice_rec()).unwrap();
        assert_eq!(scored["label"], "a");
        // second call reuses the cached scorer (factory not re-entered)
        let scored = scorer.score_record(&choice_rec()).unwrap();
        assert_eq!(scored["label"], "a");
    }

    #[test]
    fn factory_and_scorer_errors_propagate() {
        let failing_factory = DecoderScorer::with_factory(
            DecoderCheckpoint::open(None::<&str>).unwrap(),
            Arc::new(|_ckpt: &DecoderCheckpoint| Err(AfmError::msg("no weights"))),
        );
        let err = failing_factory.score_record(&choice_rec()).unwrap_err();
        assert_eq!(err.to_string(), "no weights");

        struct Failing;
        impl LetterScorer for Failing {
            fn letter_logits(&mut self, _row: &Value) -> Result<(Vec<String>, Vec<f32>)> {
                Err(AfmError::msg("forward boom"))
            }
        }
        let failing = DecoderScorer::from_parts(
            DecoderCheckpoint::open(None::<&str>).unwrap(),
            Box::new(Failing),
        );
        let err = failing.score_record(&choice_rec()).unwrap_err();
        assert_eq!(err.to_string(), "forward boom");
    }

    #[test]
    fn default_factory_reports_missing_weights() {
        let dir = tempfile::tempdir().unwrap();
        let scorer = DecoderScorer::open(Some(dir.path())).unwrap();
        let err = scorer.score_record(&choice_rec()).unwrap_err();
        assert!(err
            .to_string()
            .contains("decoder MiniCPM weights not found"));
    }

    #[test]
    fn poisoned_lock_still_recovers() {
        struct Panicking;
        impl LetterScorer for Panicking {
            fn letter_logits(&mut self, _row: &Value) -> Result<(Vec<String>, Vec<f32>)> {
                panic!("scorer exploded");
            }
        }
        let scorer = DecoderScorer::from_parts(
            DecoderCheckpoint::open(None::<&str>).unwrap(),
            Box::new(Panicking),
        );
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scorer.score_record(&choice_rec())
        }));
        std::panic::set_hook(hook);
        assert!(first.is_err());
        assert!(scorer.inner.is_poisoned());
        let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scorer.score_record(&choice_rec())
        }));
        assert!(second.is_err());
    }
}
