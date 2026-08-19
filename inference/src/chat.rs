//! Family chat templates (OpenAI `messages` → prompt string).
//!
//! Qwen3 instruct models require ChatML + a closed `<think>` block when thinking
//! is off; raw user text (e.g. `"Hello"`) yields garbage completions.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

impl ChatTurn {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Render messages with a generation prompt for the next assistant turn.
pub fn apply_chat_template(family_path: &str, messages: &[ChatTurn]) -> String {
    let path = family_path.to_ascii_lowercase();
    if path.contains("gemma") {
        gemma_it(messages)
    } else if path.contains("llama") {
        llama3(messages)
    } else if path.contains("qwen3") {
        // Instruct Qwen3 defaults to thinking; close the block so chat answers directly.
        qwen_chatml(messages, /*empty_think=*/ true)
    } else {
        // Qwen2, LFM, Nanbeige, Bonsai, Inkling: ChatML without think tags.
        qwen_chatml(messages, /*empty_think=*/ false)
    }
}

fn map_role(role: &str) -> &str {
    match role {
        "assistant" | "model" => "assistant",
        "system" => "system",
        _ => "user",
    }
}

/// Qwen ChatML. `empty_think` injects a closed `<think>` (Qwen3 non-thinking).
fn qwen_chatml(messages: &[ChatTurn], empty_think: bool) -> String {
    let mut out = String::new();
    for m in messages {
        let role = map_role(&m.role);
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    if empty_think {
        out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    } else {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// Drop Qwen3 thinking and ChatML specials from decoded assistant text.
pub fn strip_assistant_visible(raw: &str) -> String {
    let mut s = raw.replace("<|im_end|>", "").replace("<|im_start|>", "");
    if let Some(idx) = s.rfind("</think>") {
        s = s[idx + "</think>".len()..].to_string();
    } else if let Some(idx) = s.find("<think>") {
        s = s[idx + "<think>".len()..].to_string();
    }
    s.trim().to_string()
}

fn gemma_it(messages: &[ChatTurn]) -> String {
    let mut out = String::from("<bos>");
    for m in messages {
        let role = match map_role(&m.role) {
            "assistant" => "model",
            "system" => "user",
            other => other,
        };
        out.push_str("<start_of_turn>");
        out.push_str(role);
        out.push('\n');
        if map_role(&m.role) == "system" {
            out.push_str("System: ");
        }
        out.push_str(&m.content);
        out.push_str("<end_of_turn>\n");
    }
    out.push_str("<start_of_turn>model\n");
    out
}

fn llama3(messages: &[ChatTurn]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for m in messages {
        let role = map_role(&m.role);
        out.push_str("<|start_header_id|>");
        out.push_str(role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(&m.content);
        out.push_str("<|eot_id|>");
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_wraps_user_and_closes_think() {
        let s = apply_chat_template(
            "qwen/qwen3-0.6b",
            &[ChatTurn::new("user", "Hello")],
        );
        assert!(s.contains("<|im_start|>user\nHello<|im_end|>"), "{s}");
        assert!(s.contains("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{s}");
        assert!(!s.ends_with("Hello"), "{s}");
    }

    #[test]
    fn gemma_uses_turn_markers() {
        let s = apply_chat_template(
            "gemma/gemma-4-e2b-it",
            &[ChatTurn::new("user", "Hi")],
        );
        assert!(s.contains("<start_of_turn>user\nHi<end_of_turn>"), "{s}");
        assert!(s.ends_with("<start_of_turn>model\n"), "{s}");
    }

    #[test]
    fn system_and_user_order_preserved() {
        let s = apply_chat_template(
            "qwen/qwen3-0.6b",
            &[
                ChatTurn::new("system", "Be brief."),
                ChatTurn::new("user", "Hi"),
            ],
        );
        let sys = s.find("<|im_start|>system").unwrap();
        let usr = s.find("<|im_start|>user").unwrap();
        assert!(sys < usr);
    }

    #[test]
    fn strip_think_keeps_answer() {
        let raw = "<think>\nreason\n</think>\n\nHello there<|im_end|>";
        assert_eq!(strip_assistant_visible(raw), "Hello there");
    }
}
