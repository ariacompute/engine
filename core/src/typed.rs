use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AfmError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Choice,
    Score,
    Noul,
}

impl Task {
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Choice => "choice",
            Task::Score => "score",
            Task::Noul => "noul",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "choice" => Ok(Task::Choice),
            "score" => Ok(Task::Score),
            "noul" => Ok(Task::Noul),
            other => Err(AfmError::msg(format!("unknown task {other:?}"))),
        }
    }

    pub fn type_id(self) -> usize {
        match self {
            Task::Choice => 0,
            Task::Score => 1,
            Task::Noul => 2,
        }
    }
}

/// Softmax over logits; masked positions get near-zero mass.
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = temperature.max(1e-6);
    let scaled: Vec<f32> = logits.iter().map(|x| x / t).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    exps.into_iter().map(|e| e / sum).collect()
}

pub fn confidence_from_prob_list(probs: &[f32]) -> f32 {
    probs.iter().cloned().fold(0.0_f32, f32::max)
}

/// Build AFM-D / System One answer object from option probabilities.
pub fn typed_answer(task: Task, names: &[String], probabilities: &[f32]) -> Result<Value> {
    if names.len() != probabilities.len() {
        return Err(AfmError::msg("probability length must match option names"));
    }
    let mut probs_map = serde_json::Map::new();
    for (n, p) in names.iter().zip(probabilities.iter()) {
        probs_map.insert(n.clone(), Value::from(*p));
    }
    let index = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let conf = confidence_from_prob_list(probabilities);
    match task {
        Task::Choice => Ok(serde_json::json!({
            "choice": names[index],
            "probabilities": probs_map,
            "confidence": conf,
        })),
        Task::Score => {
            let expected: f32 = probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| i as f32 * p)
                .sum();
            Ok(serde_json::json!({
                "score": expected,
                "probabilities": probs_map,
                "confidence": conf,
            }))
        }
        Task::Noul => {
            let mut p_true = 0.0_f32;
            let mut mapped = serde_json::Map::new();
            for (n, p) in names.iter().zip(probabilities.iter()) {
                let key = n.to_ascii_lowercase();
                if key == "true" || key == "yes" {
                    p_true = *p;
                }
                mapped.insert(n.clone(), Value::from(*p));
            }
            if !mapped.contains_key("true") && mapped.contains_key("yes") {
                p_true = mapped
                    .get("yes")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                mapped.insert("true".into(), Value::from(p_true));
                let p_false = mapped
                    .get("no")
                    .or_else(|| mapped.get("false"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or((1.0 - p_true) as f64) as f32;
                mapped.insert("false".into(), Value::from(p_false));
            }
            if !mapped.contains_key("true") {
                return Err(AfmError::msg("noul options must include true (or yes)"));
            }
            let conf_n = p_true.max(1.0 - p_true);
            Ok(serde_json::json!({
                "noul": p_true,
                "probabilities": mapped,
                "confidence": conf_n,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_peaks() {
        let p = softmax(&[1.0, 3.0, 1.0], 1.0);
        assert!(p[1] > p[0] && p[1] > p[2]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn choice_answer() {
        let names = vec!["a".into(), "b".into()];
        let ans = typed_answer(Task::Choice, &names, &[0.2, 0.8]).unwrap();
        assert_eq!(ans["choice"], "b");
    }
}
