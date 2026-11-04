//! SemIf `direct` user JSON + MiniCPM5 chat template (thinking off, leading BOS).

use serde_json::{json, Value};

use ariacompute_core::error::{AfmError, Result};

use crate::pyjson;
use crate::semif::LETTERS;

pub const SYSTEM: &str =
    "Apply the supplied criterion to the supplied evidence. Choose exactly one \
listed option. Respond with only its uppercase letter, with no explanation or reasoning.";

pub const BOS: &str = "<s>";
pub const POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub fn pre() -> String {
    format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n")
}

/// SemIf user payload from a `record_to_semif_row` object.
pub fn user_from_row(row: &Value) -> Result<String> {
    let options = row
        .get("options")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AfmError::msg("semif row missing options"))?;
    let mut option_vals = Vec::with_capacity(options.len());
    for (i, opt) in options.iter().enumerate() {
        let letter = LETTERS
            .chars()
            .nth(i)
            .ok_or_else(|| AfmError::msg("semif row exceeds A–P"))?;
        let desc = opt
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AfmError::msg("semif option missing description"))?;
        option_vals.push(pyjson::dumps_object(&[
            ("letter", json!(letter.to_string())),
            ("description", json!(desc)),
        ]));
    }
    let state = row.get("state").cloned().unwrap_or(Value::Null);
    let criterion = row.get("question").cloned().unwrap_or(Value::Null);
    Ok(format!(
        "{{\"evidence\": {}, \"criterion\": {}, \"options\": [{}]}}",
        pyjson::dumps(&state),
        pyjson::dumps(&criterion),
        option_vals.join(", ")
    ))
}

/// Full MiniCPM5 prompt string (`apply_chat_template` + BOS, thinking off).
pub fn prompt_text(row: &Value) -> Result<String> {
    let user = user_from_row(row)?;
    Ok(format!("{BOS}{}{user}{POST}", pre()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semif::record_to_semif_row;
    use ariacompute_core::packing::Record;

    #[test]
    fn refund_choice_user_json() {
        let rec = Record::from_value(&json!({
            "id": "q1",
            "task": "choice",
            "instructions": "Pick the best action",
            "options": [
                {"name": "refund", "description": "issue a full refund"},
                {"name": "deny", "description": "deny the request"}
            ],
            "state": "user wants a refund"
        }))
        .unwrap();
        let row = record_to_semif_row(&rec).unwrap();
        let user = user_from_row(&row).unwrap();
        assert_eq!(
            user,
            "{\"evidence\": \"user wants a refund\", \"criterion\": \"Pick the best action\", \
             \"options\": [{\"letter\": \"A\", \"description\": \"refund: issue a full refund\"}, \
             {\"letter\": \"B\", \"description\": \"deny: deny the request\"}]}"
        );
        let prompt = prompt_text(&row).unwrap();
        assert!(prompt.starts_with("<s><|im_start|>system\n"));
        assert!(prompt.contains(SYSTEM));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn missing_options_is_an_error() {
        let err = user_from_row(&json!({"state": "s", "question": "q"})).unwrap_err();
        assert!(err.to_string().contains("semif row missing options"));
        let err = prompt_text(&json!({})).unwrap_err();
        assert!(err.to_string().contains("semif row missing options"));
    }

    #[test]
    fn option_without_description_is_an_error() {
        let row = json!({"options": [{"letter": "A"}]});
        let err = user_from_row(&row).unwrap_err();
        assert!(err.to_string().contains("semif option missing description"));
    }

    #[test]
    fn more_than_sixteen_options_exceeds_letters() {
        let options: Vec<Value> = (0..17)
            .map(|i| json!({"id": i.to_string(), "description": format!("d{i}")}))
            .collect();
        let err = user_from_row(&json!({"options": options})).unwrap_err();
        assert!(err.to_string().contains("exceeds A–P"));
    }

    #[test]
    fn missing_state_and_criterion_render_as_null() {
        let row = json!({"options": [{"id": "a", "description": "alpha"}, {"id": "b", "description": "beta"}]});
        let user = user_from_row(&row).unwrap();
        assert!(user.starts_with("{\"evidence\": null, \"criterion\": null"));
        assert!(user.contains("\"letter\": \"A\""));
        assert!(user.contains("\"letter\": \"B\""));
    }

    #[test]
    fn pre_matches_the_system_prompt() {
        let pre = pre();
        assert!(pre.starts_with("<|im_start|>system\n"));
        assert!(pre.ends_with("<|im_start|>user\n"));
        assert!(!pre.contains(BOS));
    }
}
