//! Tensor name aliases across Aria families.
//!
//! Tiny/`--tiny` uses GGUF-style `blk.*`. Real `model` quantize keeps HuggingFace
//! state_dict names, which vary by family:
//! - Qwen / Gemma-3 text / LFM / Nanbeige / …: `model.layers.*`
//! - Gemma-4 / Gemma-3n / LFM-VL / …: `model.language_model.layers.*`
//! - OpenVLA-style: `language_model.model.layers.*`
//! - Gemma-2+ FFN pre-norm: `pre_feedforward_layernorm` (in addition to
//!   LLaMA/Qwen `post_attention_layernorm`)

/// HF-style decoder layer prefixes (without trailing layer index).
const HF_LAYER_PREFIXES: &[&str] = &[
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.model.layers",
];

fn with_layer_suffix(layer: usize, suffix: &str) -> Vec<String> {
    HF_LAYER_PREFIXES
        .iter()
        .map(|p| format!("{p}.{layer}.{suffix}"))
        .collect()
}

/// Candidates for a logical weight (first hit wins).
pub fn emb_names() -> Vec<&'static str> {
    vec![
        "token_embd.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
    ]
}

pub fn output_norm_names() -> Vec<&'static str> {
    vec![
        "output_norm.weight",
        "model.norm.weight",
        "model.language_model.norm.weight",
        "language_model.model.norm.weight",
        "model.model.norm.weight",
    ]
}

/// LM head; last entries are weight-tied embedding fallbacks.
pub fn output_names() -> Vec<&'static str> {
    vec![
        "output.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.model.embed_tokens.weight",
    ]
}

pub fn attn_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_norm.weight")];
    names.extend(with_layer_suffix(layer, "input_layernorm.weight"));
    names
}

pub fn ffn_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_norm.weight")];
    // LLaMA / Qwen: post-attention = pre-FFN.
    names.extend(with_layer_suffix(layer, "post_attention_layernorm.weight"));
    // Gemma-2 / Gemma-3 / Gemma-4: dedicated pre-FFN norm.
    names.extend(with_layer_suffix(layer, "pre_feedforward_layernorm.weight"));
    names
}

pub fn attn_q_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_q.weight")];
    names.extend(with_layer_suffix(layer, "self_attn.q_proj.weight"));
    names
}

pub fn attn_k_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_k.weight")];
    names.extend(with_layer_suffix(layer, "self_attn.k_proj.weight"));
    names
}

pub fn attn_v_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_v.weight")];
    names.extend(with_layer_suffix(layer, "self_attn.v_proj.weight"));
    names
}

pub fn attn_o_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_output.weight")];
    names.extend(with_layer_suffix(layer, "self_attn.o_proj.weight"));
    names
}

pub fn ffn_gate_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_gate.weight")];
    names.extend(with_layer_suffix(layer, "mlp.gate_proj.weight"));
    names
}

pub fn ffn_up_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_up.weight")];
    names.extend(with_layer_suffix(layer, "mlp.up_proj.weight"));
    names
}

pub fn ffn_down_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_down.weight")];
    names.extend(with_layer_suffix(layer, "mlp.down_proj.weight"));
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_cover_hf_blk_and_language_model() {
        let a = attn_norm_names(0);
        assert!(a.iter().any(|s| s.contains("attn_norm")));
        assert!(a.iter().any(|s| s == "model.layers.0.input_layernorm.weight"));
        assert!(a
            .iter()
            .any(|s| s == "model.language_model.layers.0.input_layernorm.weight"));
        assert!(a
            .iter()
            .any(|s| s == "language_model.model.layers.0.input_layernorm.weight"));

        let f = ffn_norm_names(0);
        assert!(f
            .iter()
            .any(|s| s == "model.layers.0.post_attention_layernorm.weight"));
        assert!(f
            .iter()
            .any(|s| s == "model.layers.0.pre_feedforward_layernorm.weight"));
        assert!(f.iter().any(|s| s.contains("language_model")
            && s.contains("pre_feedforward_layernorm")));

        assert!(emb_names().contains(&"model.embed_tokens.weight"));
        assert!(emb_names().contains(&"model.language_model.embed_tokens.weight"));
        assert!(output_names().contains(&"lm_head.weight"));
        assert!(output_norm_names().contains(&"model.language_model.norm.weight"));
    }
}
