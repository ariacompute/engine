//! Decoder scoring map + scorer shell (MiniCPM forward when weights available).

use std::path::Path;

use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::packing::Record;
use ariacompute_core::systemone::answer_systemone_dd;
use ariacompute_core::typed::Task;
use serde_json::{json, Map, Value};

use crate::checkpoint::DecoderCheckpoint;
use crate::semif::record_to_semif_row;

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
            if !mapped.is_empty() {
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
                let p = probs
                    .get(&opt.name)
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
    }
    Ok((probs, label))
}

pub struct DecoderScorer {
    pub checkpoint: DecoderCheckpoint,
}

impl DecoderScorer {
    pub fn open(checkpoint: Option<impl AsRef<Path>>) -> Result<Self> {
        Ok(Self {
            checkpoint: DecoderCheckpoint::open(checkpoint)?,
        })
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
        let _row = record_to_semif_row(record)?;
        Err(AfmError::msg(format!(
            "decoder MiniCPM forward not loaded (base={}, rev={}); use score_from_semif_out for golden tests",
            self.checkpoint.config.base_model, self.checkpoint.config.base_revision
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariacompute_core::packing::Record;

    #[test]
    fn map_choice_probs() {
        let rec = Record::from_value(&json!({
            "id": "1",
            "task": "choice",
            "options": [
                {"name": "a", "description": ""},
                {"name": "b", "description": ""}
            ],
            "state": "",
            "instructions": "x"
        }))
        .unwrap();
        let out = json!({
            "option_ids": ["a", "b"],
            "probabilities": [0.2, 0.8]
        });
        let (probs, label) = probs_from_semif_out(&rec, &out).unwrap();
        assert_eq!(label, "b");
        assert!((probs["b"].as_f64().unwrap() - 0.8).abs() < 1e-9);
        let scored = DecoderScorer::open(None::<&str>)
            .unwrap()
            .score_from_semif_out(&rec, &out)
            .unwrap();
        assert_eq!(scored["systemone"]["choice"], "b");
    }
}
