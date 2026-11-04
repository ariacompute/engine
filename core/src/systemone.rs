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
pub fn record_from_systemone_question(
    qid: &str,
    state: &Value,
    question: &Value,
) -> Result<Record> {
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
            let obj = criteria.and_then(|c| c.as_object()).ok_or_else(|| {
                AfmError::msg(format!(
                    "systemone choice {qid} needs nonempty criteria dict"
                ))
            })?;
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
            let arr = criteria.and_then(|c| c.as_array()).ok_or_else(|| {
                AfmError::msg(format!("systemone score {qid} needs 2..10 levels"))
            })?;
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
                .map(|(i, name)| {
                    i as f64 * probs.get(*name).and_then(|v| v.as_f64()).unwrap_or(0.0)
                })
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

    #[test]
    fn choice_criteria_render_null_string_and_scalar_values() {
        let q = json!({"type": "CHOICE", "criteria": {"a": null, "b": "beta", "c": 3}});
        let rec = record_from_systemone_question("q", &Value::Null, &q).unwrap();
        assert_eq!(
            rec.instructions,
            "Select the best-fitting option given the state."
        );
        assert_eq!(rec.options[0].description, "");
        assert_eq!(rec.options[1].description, "beta");
        assert_eq!(rec.options[2].description, "3");
        assert_eq!(rec.state, Value::Null);

        // empty instructions fall back to the default prompt
        let q = json!({"type": "choice", "instructions": "   ", "criteria": {"a": "x"}});
        let rec = record_from_systemone_question("q", &Value::Null, &q).unwrap();
        assert_eq!(
            rec.instructions,
            "Select the best-fitting option given the state."
        );
    }

    #[test]
    fn choice_requires_nonempty_criteria_dict() {
        let q = json!({"type": "choice"});
        let err = record_from_systemone_question("q1", &Value::Null, &q).unwrap_err();
        assert!(err
            .to_string()
            .contains("choice q1 needs nonempty criteria dict"));

        let q = json!({"type": "choice", "criteria": {}});
        let err = record_from_systemone_question("q1", &Value::Null, &q).unwrap_err();
        assert!(err
            .to_string()
            .contains("choice q1 needs nonempty criteria dict"));

        let q = json!({"type": "choice", "criteria": [1, 2]});
        let err = record_from_systemone_question("q1", &Value::Null, &q).unwrap_err();
        assert!(err.to_string().contains("needs nonempty criteria dict"));
    }

    #[test]
    fn score_criteria_accepts_two_to_ten_levels() {
        let q = json!({"type": "score", "criteria": ["low", 2]});
        let rec = record_from_systemone_question("q", &json!("s"), &q).unwrap();
        assert_eq!(rec.task, Task::Score);
        assert_eq!(rec.options[0].name, "level-0");
        assert_eq!(rec.options[1].description, "2");

        for criteria in [
            json!(["only"]),
            json!((0..11).map(|i| i.to_string()).collect::<Vec<_>>()),
        ] {
            let q = json!({"type": "score", "criteria": criteria});
            let err = record_from_systemone_question("q", &Value::Null, &q).unwrap_err();
            assert!(err.to_string().contains("score q needs 2..10 levels"));
        }
        let q = json!({"type": "score"});
        let err = record_from_systemone_question("q", &Value::Null, &q).unwrap_err();
        assert!(err.to_string().contains("needs 2..10 levels"));
    }

    #[test]
    fn noul_uses_criteria_texts_with_defaults() {
        let q = json!({"type": "noul", "criteria": {"true": "it holds", "false": "it fails"}});
        let rec = record_from_systemone_question("q", &Value::Null, &q).unwrap();
        assert_eq!(rec.options[0].name, "false");
        assert_eq!(rec.options[0].description, "it fails");
        assert_eq!(rec.options[1].description, "it holds");

        let q = json!({"type": "noul", "criteria": {"true": 1}});
        let rec = record_from_systemone_question("q", &Value::Null, &q).unwrap();
        assert_eq!(
            rec.options[0].description,
            "no, the statement does not hold"
        );
        assert_eq!(rec.options[1].description, "yes, the statement holds");

        let q = json!({"type": "noul"});
        let rec = record_from_systemone_question("q", &Value::Null, &q).unwrap();
        assert_eq!(rec.options.len(), 2);
    }

    #[test]
    fn unknown_or_missing_type_is_rejected() {
        let err = record_from_systemone_question("q9", &Value::Null, &json!({"type": "rank"}))
            .unwrap_err();
        assert!(err.to_string().contains("unsupported systemone type"));
        let err = record_from_systemone_question("q9", &Value::Null, &json!({})).unwrap_err();
        assert!(err.to_string().contains("unsupported systemone type"));
    }

    #[test]
    fn answer_from_probs_builds_the_record_answer() {
        let rec = Record {
            id: "r".into(),
            task: Task::Choice,
            instructions: "pick".into(),
            options: vec![
                OptionItem {
                    name: "a".into(),
                    description: String::new(),
                },
                OptionItem {
                    name: "b".into(),
                    description: String::new(),
                },
            ],
            state: Value::Null,
            label: None,
        };
        let ans = answer_from_probs(&rec, &[0.1, 0.9]).unwrap();
        assert_eq!(ans["choice"], "b");
    }

    #[test]
    fn answer_systemone_dd_covers_every_task() {
        let mut probs = Map::new();
        probs.insert("true".into(), json!(0.8));
        probs.insert("false".into(), json!(0.2));

        let noul = Record {
            id: "n".into(),
            task: Task::Noul,
            instructions: String::new(),
            options: vec![],
            state: Value::Null,
            label: None,
        };
        let ans = answer_systemone_dd(&noul, &probs, "true");
        assert!((ans["noul"].as_f64().unwrap() - 0.8).abs() < 1e-9);

        // yes alias
        let mut yes = Map::new();
        yes.insert("yes".into(), json!(0.6));
        let ans = answer_systemone_dd(&noul, &yes, "yes");
        assert!((ans["noul"].as_f64().unwrap() - 0.6).abs() < 1e-9);

        let score = Record {
            id: "s".into(),
            task: Task::Score,
            instructions: String::new(),
            options: vec![],
            state: Value::Null,
            label: None,
        };
        let mut levels = Map::new();
        levels.insert("level-0".into(), json!(0.5));
        levels.insert("level-1".into(), json!(0.5));
        let ans = answer_systemone_dd(&score, &levels, "level-1");
        assert_eq!(ans["choice"], "level-1");
        assert!((ans["score"].as_f64().unwrap() - 0.5).abs() < 1e-9);

        let choice = Record {
            id: "c".into(),
            task: Task::Choice,
            instructions: String::new(),
            options: vec![],
            state: Value::Null,
            label: None,
        };
        // non-numeric probabilities are ignored by the confidence fold
        let mut mixed = Map::new();
        mixed.insert("a".into(), json!("n/a"));
        mixed.insert("b".into(), json!(0.4));
        let ans = answer_systemone_dd(&choice, &mixed, "b");
        assert_eq!(ans["choice"], "b");
        assert!((ans["confidence"].as_f64().unwrap() - 0.4).abs() < 1e-9);
    }
}
