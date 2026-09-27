//! System One HTTP request/response shapes (AFM-D / Kev / NeoHorse compatible).

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::error::{AfmError, Result};
use crate::packing::{OptionItem, Record};
use crate::typed::{typed_answer, Task};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneRequest {
    #[serde(default)]
    pub state: Value,
    #[serde(default)]
    pub questions: Map<String, Value>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub answers: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

pub type SystemOneAnswer = Value;

/// Invert System One question body → AFM-D record (matches `dd/serve.py`).
pub fn record_from_systemone_question(qid: &str, state: &Value, question: &Value) -> Result<Record> {
    let qtype = question
        .get("type")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let instructions = question
        .get("instructions")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("Select the best-fitting option given the state.")
        .to_string();
    let criteria = question.get("criteria");
    let (task, options) = match qtype.as_str() {
        "noul" => {
            let crit = criteria.and_then(|c| c.as_object());
            let false_text = crit
                .and_then(|c| c.get("false"))
                .and_then(|v| v.as_str())
                .unwrap_or("no, the statement does not hold");
            let true_text = crit
                .and_then(|c| c.get("true"))
                .and_then(|v| v.as_str())
                .unwrap_or("yes, the statement holds");
            (
                Task::Noul,
                vec![
                    OptionItem {
                        name: "false".into(),
                        description: false_text.into(),
                    },
                    OptionItem {
                        name: "true".into(),
                        description: true_text.into(),
                    },
                ],
            )
        }
        "choice" => {
            let obj = criteria
                .and_then(|c| c.as_object())
                .ok_or_else(|| AfmError::msg(format!("systemone choice {qid} needs nonempty criteria dict")))?;
            if obj.is_empty() {
                return Err(AfmError::msg(format!(
                    "systemone choice {qid} needs nonempty criteria dict"
                )));
            }
            let options: Vec<OptionItem> = obj
                .iter()
                .map(|(k, v)| OptionItem {
                    name: k.clone(),
                    description: if v.is_null() {
                        String::new()
                    } else {
                        v.as_str().unwrap_or(&v.to_string()).to_string()
                    },
                })
                .collect();
            (Task::Choice, options)
        }
        "score" => {
            let arr = criteria
                .and_then(|c| c.as_array())
                .ok_or_else(|| AfmError::msg(format!("systemone score {qid} needs 2..10 levels")))?;
            if !(2..=10).contains(&arr.len()) {
                return Err(AfmError::msg(format!(
                    "systemone score {qid} needs 2..10 levels"
                )));
            }
            let options: Vec<OptionItem> = arr
                .iter()
                .enumerate()
                .map(|(i, item)| OptionItem {
                    name: format!("level-{i}"),
                    description: item.as_str().unwrap_or(&item.to_string()).to_string(),
                })
                .collect();
            (Task::Score, options)
        }
        other => {
            return Err(AfmError::msg(format!(
                "unsupported systemone type {other:?} for {qid}"
            )))
        }
    };
    Ok(Record {
        id: qid.to_string(),
        task,
        instructions,
        options,
        state: state.clone(),
        label: None,
    })
}

pub fn answer_from_probs(record: &Record, probs: &[f32]) -> Result<Value> {
    let names: Vec<String> = record.options.iter().map(|o| o.name.clone()).collect();
    typed_answer(record.task, &names, probs)
}

/// Decoder-style System One answer (includes choice on score).
pub fn answer_systemone_dd(record: &Record, probs: &Map<String, Value>, label: &str) -> Value {
    let conf = probs
        .values()
        .filter_map(|v| v.as_f64())
        .fold(0.0_f64, f64::max);
    match record.task {
        Task::Noul => {
            let mut p_true = 0.0;
            for (name, value) in probs {
                if name == "true" || name == "yes" {
                    p_true = value.as_f64().unwrap_or(0.0);
                    break;
                }
            }
            json!({
                "noul": p_true,
                "probabilities": probs,
                "confidence": conf,
            })
        }
        Task::Score => {
            let names: Vec<&String> = probs.keys().collect();
            let expected: f64 = names
                .iter()
                .enumerate()
                .map(|(i, name)| i as f64 * probs.get(*name).and_then(|v| v.as_f64()).unwrap_or(0.0))
                .sum();
            json!({
                "score": expected,
                "choice": label,
                "probabilities": probs,
                "confidence": conf,
            })
        }
        Task::Choice => json!({
            "choice": label,
            "probabilities": probs,
            "confidence": conf,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_choice_question() {
        let q = json!({
            "type": "choice",
            "instructions": "pick",
            "criteria": {"a": "alpha", "b": "beta"}
        });
        let rec = record_from_systemone_question("q1", &json!("state"), &q).unwrap();
        assert_eq!(rec.task, Task::Choice);
        assert_eq!(rec.options.len(), 2);
    }
}
