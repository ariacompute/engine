//! Shortlist high-K Choice options (port of `afm_d.de.shortlist` thresholding).

use ariacompute_core::contract::{DEFAULT_SHORTLIST_K, SHORTLIST_OPTION_THRESHOLD};
use ariacompute_core::packing::{OptionItem, Record};
use ariacompute_core::typed::Task;

/// When Choice K is large, keep first `k` options (embedding shortlist is deferred to weight load).
pub fn maybe_shortlist(record: &Record, k: Option<usize>) -> Record {
    if record.task != Task::Choice {
        return record.clone();
    }
    if record.options.len() <= SHORTLIST_OPTION_THRESHOLD {
        return record.clone();
    }
    let keep = k.unwrap_or(DEFAULT_SHORTLIST_K).min(record.options.len());
    let options: Vec<OptionItem> = record.options.iter().take(keep).cloned().collect();
    let mut out = record.clone();
    out.options = options;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariacompute_core::packing::OptionItem;

    #[test]
    fn shortlists_large_choice() {
        let options: Vec<OptionItem> = (0..50)
            .map(|i| OptionItem {
                name: format!("o{i}"),
                description: format!("d{i}"),
            })
            .collect();
        let rec = Record {
            id: "x".into(),
            task: Task::Choice,
            instructions: "i".into(),
            options,
            state: serde_json::json!("s"),
            label: None,
        };
        let slim = maybe_shortlist(&rec, Some(20));
        assert_eq!(slim.options.len(), 20);
    }
}
