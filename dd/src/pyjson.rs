//! CPython `json.dumps` default separators (`", "` / `": "`), `ensure_ascii=False`.
//!
//! Matches Ollaya `pyjson` / SemIf user JSON so letter logits see the same prompt bytes.

use std::fmt::Write;

use serde_json::{Number, Value};

pub fn dumps(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

/// Like `dumps` for an object, but keeps the given key order (Python 3.7+ `dict`).
pub fn dumps_object(entries: &[(&str, Value)]) -> String {
    let mut out = String::from("{");
    for (i, (k, v)) in entries.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_str(&mut out, k);
        out.push_str(": ");
        write_value(&mut out, v);
    }
    out.push('}');
    out
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_str(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, k);
                out.push_str(": ");
                write_value(out, v);
            }
            out.push('}');
        }
    }
}

fn write_number(out: &mut String, n: &Number) {
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else {
        let x = n.as_f64().unwrap_or(f64::NAN);
        if x.fract() == 0.0 && x.is_finite() && x.abs() < 1e15 {
            let _ = write!(out, "{x:.1}");
        } else {
            let _ = write!(out, "{x}");
        }
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_default_separators() {
        assert_eq!(
            dumps(&json!({"a": [1, true], "b": "x"})),
            "{\"a\": [1, true], \"b\": \"x\"}"
        );
    }
}
