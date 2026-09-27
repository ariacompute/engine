//! SemIf decision-row conversion (match `afm_d.dd.data.record_to_semif_row`).

use ariacompute_core::contract::MAX_OPTIONS_DD;
use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::packing::Record;
use ariacompute_core::typed::Task;
use serde_json::{json, Value};

pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";

pub fn record_to_semif_row(record: &Record) -> Result<Value> {
    let options = &record.options;
    if !(2..=MAX_OPTIONS_DD).contains(&options.len()) {
        return Err(AfmError::msg(format!(
            "decoder row {}: need 2..{MAX_OPTIONS_DD} options, got {}",
            record.id,
            options.len()
        )));
    }
    let options_out: Vec<Value> = match record.task {
        Task::Noul => {
            let by: std::collections::HashMap<&str, _> =
                options.iter().map(|o| (o.name.as_str(), o)).collect();
            let mut built = Vec::new();
            for key in ["true", "false"] {
                let src = by.get(key).copied().or_else(|| {
                    by.get(if key == "true" { "yes" } else { "no" })
                        .copied()
                });
                let desc = src
                    .map(|o| o.description.clone())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| format!("The proposition is {key}."));
                built.push(json!({"id": key, "description": format!("{key}: {desc}")}));
            }
            built
        }
        Task::Choice => options
            .iter()
            .map(|opt| {
                json!({
                    "id": opt.name,
                    "description": format!("{}: {}", opt.name, if opt.description.is_empty() { &opt.name } else { &opt.description }),
                })
            })
            .collect(),
        Task::Score => options
            .iter()
            .enumerate()
            .map(|(i, opt)| {
                json!({
                    "id": i.to_string(),
                    "description": format!("{i}: {}", if opt.description.is_empty() { &opt.name } else { &opt.description }),
                })
            })
            .collect(),
    };
    let instructions = if record.instructions.trim().is_empty() {
        "Select the best-fitting option given the state.".to_string()
    } else {
        record.instructions.trim().to_string()
    };
    Ok(json!({
        "id": record.id,
        "state": record.state,
        "question": instructions,
        "options": options_out,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn choice_row() {
        let rec = Record::from_value(&json!({
            "id": "1",
            "task": "choice",
            "instructions": "pick",
            "options": [
                {"name": "a", "description": "alpha"},
                {"name": "b", "description": "beta"}
            ],
            "state": "s"
        }))
        .unwrap();
        let row = record_to_semif_row(&rec).unwrap();
        assert_eq!(row["options"].as_array().unwrap().len(), 2);
        assert_eq!(row["options"][0]["id"], "a");
    }
}
