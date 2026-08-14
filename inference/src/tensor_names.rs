//! Tensor name aliases: tiny/`--tiny` uses GGUF-style `blk.*`; real `model` quantize
//! keeps HuggingFace state_dict names.

/// Candidates for a logical weight (first hit wins).
pub fn emb_names() -> [&'static str; 2] {
    ["token_embd.weight", "model.embed_tokens.weight"]
}

pub fn output_norm_names() -> [&'static str; 2] {
    ["output_norm.weight", "model.norm.weight"]
}

/// LM head; last entry is weight-tied embedding fallback.
pub fn output_names() -> [&'static str; 3] {
    [
        "output.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
}

pub fn attn_norm_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.attn_norm.weight"),
        format!("model.layers.{layer}.input_layernorm.weight"),
    ]
}

pub fn ffn_norm_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.ffn_norm.weight"),
        format!("model.layers.{layer}.post_attention_layernorm.weight"),
    ]
}

pub fn attn_q_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.attn_q.weight"),
        format!("model.layers.{layer}.self_attn.q_proj.weight"),
    ]
}

pub fn attn_k_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.attn_k.weight"),
        format!("model.layers.{layer}.self_attn.k_proj.weight"),
    ]
}

pub fn attn_v_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.attn_v.weight"),
        format!("model.layers.{layer}.self_attn.v_proj.weight"),
    ]
}

pub fn attn_o_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.attn_output.weight"),
        format!("model.layers.{layer}.self_attn.o_proj.weight"),
    ]
}

pub fn ffn_gate_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.ffn_gate.weight"),
        format!("model.layers.{layer}.mlp.gate_proj.weight"),
    ]
}

pub fn ffn_up_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.ffn_up.weight"),
        format!("model.layers.{layer}.mlp.up_proj.weight"),
    ]
}

pub fn ffn_down_names(layer: usize) -> [String; 2] {
    [
        format!("blk.{layer}.ffn_down.weight"),
        format!("model.layers.{layer}.mlp.down_proj.weight"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_cover_hf_and_blk() {
        let a = attn_norm_names(0);
        assert!(a.iter().any(|s| s.contains("attn_norm")));
        assert!(a.iter().any(|s| s.contains("input_layernorm")));
        assert!(emb_names().contains(&"model.embed_tokens.weight"));
        assert!(output_names().contains(&"lm_head.weight"));
    }
}
