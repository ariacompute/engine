//! AFM-D record packing helpers (Laya-aligned text forms; token ids need a tokenizer).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::contract::{HEAD_MAX_LEN, MAX_LEN, OPTION_DESC_MAX};
use crate::error::{AfmError, Result};
use crate::typed::Task;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionItem {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub task: Task,
    #[serde(default)]
    pub instructions: String,
    #[serde(default)]
    pub options: Vec<OptionItem>,
    #[serde(default)]
    pub state: Value,
    #[serde(default)]
    pub label: Option<Value>,
}

impl Record {
    pub fn from_value(v: &Value) -> Result<Self> {
        let id = v
            .get("id")
            .or_else(|| v.get("qid"))
            .and_then(|x| x.as_str())
            .unwrap_or("q")
            .to_string();
        let task = Task::parse(v.get("task").and_then(|x| x.as_str()).unwrap_or(""))?;
        let instructions = v
            .get("instructions")
            .and_then(|x| x.as_str())
            .unwrap_or("Select the best-fitting option given the state.")
            .to_string();
        let options = parse_options(v, task)?;
        Ok(Self {
            id,
            task,
            instructions,
            options,
            state: v.get("state").cloned().unwrap_or(Value::String(String::new())),
            label: v.get("label").cloned(),
        })
    }
}

fn noul_polarity(name: &str) -> Option<&'static str> {
    match name.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" => Some("true"),
        "false" | "no" => Some("false"),
        _ => None,
    }
}

fn parse_options(v: &Value, task: Task) -> Result<Vec<OptionItem>> {
    if let Some(arr) = v.get("options").and_then(|x| x.as_array()) {
        let mut opts: Vec<OptionItem> = arr
            .iter()
            .map(|o| OptionItem {
                name: o
                    .get("name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                description: o
                    .get("description")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            })
            .collect();
        if task == Task::Noul {
            opts = canonicalize_noul(opts);
        }
        return Ok(dedupe_option_texts(opts));
    }
    // criteria fallback
    let criteria = v.get("criteria");
    match task {
        Task::Noul => {
            let crit = criteria.and_then(|c| c.as_object());
            let false_text = crit
                .and_then(|c| c.get("false"))
                .map(render_criterion)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "no, the statement does not hold".into());
            let true_text = crit
                .and_then(|c| c.get("true"))
                .map(render_criterion)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "yes, the statement holds".into());
            Ok(dedupe_option_texts(vec![
                OptionItem {
                    name: "false".into(),
                    description: false_text,
                },
                OptionItem {
                    name: "true".into(),
                    description: true_text,
                },
            ]))
        }
        Task::Choice => {
            if let Some(obj) = criteria.and_then(|c| c.as_object()) {
                let opts: Vec<OptionItem> = obj
                    .iter()
                    .map(|(k, v)| OptionItem {
                        name: k.clone(),
                        description: if v.is_null() || v.as_str() == Some("") {
                            String::new()
                        } else {
                            render_criterion(v)
                        },
                    })
                    .collect();
                return Ok(dedupe_option_texts(opts));
            }
            Err(AfmError::msg("choice record needs options or criteria dict"))
        }
        Task::Score => {
            if let Some(arr) = criteria.and_then(|c| c.as_array()) {
                let opts: Vec<OptionItem> = arr
                    .iter()
                    .enumerate()
                    .map(|(i, item)| OptionItem {
                        name: format!("level-{i}"),
                        description: render_criterion(item),
                    })
                    .collect();
                return Ok(dedupe_option_texts(opts));
            }
            Err(AfmError::msg("score record needs options or criteria list"))
        }
    }
}

fn canonicalize_noul(options: Vec<OptionItem>) -> Vec<OptionItem> {
    let mut remapped = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for opt in options {
        let name = noul_polarity(&opt.name)
            .unwrap_or(opt.name.as_str())
            .to_string();
        if seen.insert(name.clone()) {
            remapped.push(OptionItem {
                name,
                description: opt.description,
            });
        }
    }
    if !seen.contains("false") {
        remapped.insert(
            0,
            OptionItem {
                name: "false".into(),
                description: "no, the statement does not hold".into(),
            },
        );
    }
    if !seen.contains("true") {
        remapped.push(OptionItem {
            name: "true".into(),
            description: "yes, the statement holds".into(),
        });
    }
    remapped.sort_by_key(|o| match o.name.as_str() {
        "false" => 0,
        "true" => 1,
        _ => 99,
    });
    remapped
}

fn dedupe_option_texts(options: Vec<OptionItem>) -> Vec<OptionItem> {
    let mut used = std::collections::HashSet::new();
    let mut out = Vec::new();
    for opt in options {
        let mut text = if opt.description.trim().is_empty() {
            opt.name.clone()
        } else {
            opt.description.trim().to_string()
        };
        if used.contains(&text) {
            text = format!("{}: {text}", opt.name);
        }
        if used.contains(&text) {
            let mut n = 2;
            while used.contains(&format!("{text} ({n})")) {
                n += 1;
            }
            text = format!("{text} ({n})");
        }
        used.insert(text.clone());
        out.push(OptionItem {
            name: opt.name,
            description: text,
        });
    }
    out
}

pub fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Render option display texts in label-index order (Laya `render_options`).
pub fn render_option_texts(task: Task, options: &[OptionItem]) -> Vec<String> {
    match task {
        Task::Choice => options
            .iter()
            .map(|o| {
                if o.description.is_empty() {
                    o.name.clone()
                } else {
                    format!("{}: {}", o.name, o.description)
                }
            })
            .collect(),
        Task::Score => options
            .iter()
            .enumerate()
            .map(|(i, o)| format!("level {i}: {}", o.description))
            .collect(),
        Task::Noul => {
            let by: std::collections::HashMap<&str, &OptionItem> =
                options.iter().map(|o| (o.name.as_str(), o)).collect();
            let false_desc = by
                .get("false")
                .map(|o| o.description.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("no, the statement does not hold");
            let true_desc = by
                .get("true")
                .map(|o| o.description.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("yes, the statement holds");
            vec![
                format!("false: {false_desc}"),
                format!("true: {true_desc}"),
            ]
        }
    }
}

/// Head instruction string: `{task} question: {instructions}`.
pub fn head_instruction(task: Task, instructions: &str) -> String {
    format!("{} question: {}", task.as_str(), instructions)
}

/// Text-form packed prompt (for debugging / golden fixtures without tokenizer).
pub fn pack_record_text(record: &Record) -> String {
    let opts = render_option_texts(record.task, &record.options);
    let head = head_instruction(record.task, &record.instructions);
    let mut parts = vec![format!("[CLS] {head} [SEP]")];
    for o in &opts {
        parts.push(format!("[MASK] {o}"));
    }
    parts.push(format!(
        "[SEP] {} [SEP]",
        serialize_state(&record.state)
    ));
    parts.join(" ")
}

pub struct PackedIds {
    pub input_ids: Vec<u32>,
    pub mask_positions: Vec<usize>,
}

/// Build Laya-style sequence ids given special-token ids and a encode-fn for text pieces.
pub fn build_sequence_ids(
    encode: &dyn Fn(&str) -> Vec<u32>,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
    record: &Record,
    max_len: usize,
    head_max_len: usize,
) -> PackedIds {
    let opts = render_option_texts(record.task, &record.options);
    let ins = record.instructions.replace("[MASK]", " ");
    let head_text = format!("{} question: {ins}", record.task.as_str());
    let mut head_ids = encode(&head_text);
    let mut opt_ids: Vec<Vec<u32>> = opts
        .iter()
        .map(|o| {
            let mut v = vec![mask_id];
            let piece = encode(&format!(" {}", o.replace("[MASK]", " ")));
            v.extend(piece.into_iter().take(OPTION_DESC_MAX));
            v
        })
        .collect();
    let mut opt_budget = head_max_len.saturating_sub(opt_ids.iter().map(|o| o.len()).sum());
    if opt_budget < 16 {
        let per = ((head_max_len.saturating_sub(16)) / opt_ids.len().max(1)).max(4);
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head_max_len.saturating_sub(opt_ids.iter().map(|o| o.len()).sum());
    }
    head_ids.truncate(opt_budget.max(8));

    let mut ids = vec![cls_id];
    ids.extend(head_ids);
    ids.push(sep_id);
    let mut markers = Vec::new();
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(sep_id);
    let room = max_len.saturating_sub(ids.len() + 1);
    let st = encode(&serialize_state(&record.state).replace("[MASK]", " "));
    let st_tail = if st.len() > room {
        st[st.len() - room..].to_vec()
    } else {
        st
    };
    ids.extend(st_tail);
    ids.push(sep_id);
    ids.truncate(max_len);
    let markers: Vec<usize> = markers.into_iter().filter(|m| *m < max_len).collect();
    PackedIds {
        input_ids: ids,
        mask_positions: markers,
    }
}

pub fn default_budgets() -> (usize, usize) {
    (MAX_LEN, HEAD_MAX_LEN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pack_noul_text() {
        let rec = Record::from_value(&json!({
            "id": "1",
            "task": "noul",
            "instructions": "Is this true?",
            "options": [
                {"name": "yes", "description": "holds"},
                {"name": "no", "description": "does not"}
            ],
            "state": "hello"
        }))
        .unwrap();
        assert_eq!(rec.options[0].name, "false");
        assert_eq!(rec.options[1].name, "true");
        let text = pack_record_text(&rec);
        assert!(text.contains("[MASK] false:"));
        assert!(text.contains("[MASK] true:"));
        assert!(text.contains("[CLS] noul question:"));
    }

    #[test]
    fn build_sequence_markers() {
        let encode = |s: &str| -> Vec<u32> {
            s.chars().map(|c| c as u32).collect()
        };
        let rec = Record::from_value(&json!({
            "id": "c",
            "task": "choice",
            "instructions": "pick",
            "options": [
                {"name": "a", "description": "alpha"},
                {"name": "b", "description": "beta"}
            ],
            "state": "s"
        }))
        .unwrap();
        let packed = build_sequence_ids(&encode, 1, 2, 3, &rec, 256, 128);
        assert_eq!(packed.mask_positions.len(), 2);
        assert_eq!(packed.input_ids[0], 1);
        assert_eq!(packed.input_ids[packed.mask_positions[0]], 3);
    }
}
