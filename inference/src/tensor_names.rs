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
    // LFM2/2.5: operator_norm before attn or short-conv.
    names.extend(with_layer_suffix(layer, "operator_norm.weight"));
    names
}

pub fn ffn_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_norm.weight")];
    // Gemma-2 / Gemma-3 / Gemma-4: dedicated pre-FFN norm first.
    names.extend(pre_feedforward_norm_names(layer));
    // LLaMA / Qwen: post-attention = pre-FFN.
    names.extend(with_layer_suffix(layer, "post_attention_layernorm.weight"));
    // LFM2 dense / conv layers.
    names.extend(with_layer_suffix(layer, "ffn_norm.weight"));
    names
}

/// Gemma-2+ dedicated pre-FFN norm (presence implies the 4-norm residual graph).
pub fn pre_feedforward_norm_names(layer: usize) -> Vec<String> {
    with_layer_suffix(layer, "pre_feedforward_layernorm.weight")
}

/// Gemma-2+ residual post-attn norm (not the LLaMA FFN norm alias).
pub fn attn_post_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_post_norm.weight")];
    names.extend(with_layer_suffix(layer, "post_attention_layernorm.weight"));
    names
}

/// Gemma-2+ residual post-FFN norm.
pub fn ffn_post_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_post_norm.weight")];
    names.extend(with_layer_suffix(layer, "post_feedforward_layernorm.weight"));
    names
}

pub fn embed_per_layer_names() -> Vec<&'static str> {
    vec![
        "model.embed_tokens_per_layer.weight",
        "model.language_model.embed_tokens_per_layer.weight",
        "language_model.model.embed_tokens_per_layer.weight",
    ]
}

pub fn per_layer_model_projection_names() -> Vec<&'static str> {
    vec![
        "model.per_layer_model_projection.weight",
        "model.language_model.per_layer_model_projection.weight",
        "language_model.model.per_layer_model_projection.weight",
    ]
}

pub fn per_layer_projection_norm_names() -> Vec<&'static str> {
    vec![
        "model.per_layer_projection_norm.weight",
        "model.language_model.per_layer_projection_norm.weight",
        "language_model.model.per_layer_projection_norm.weight",
    ]
}

pub fn layer_ple_gate_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ple_gate.weight")];
    names.extend(with_layer_suffix(layer, "per_layer_input_gate.weight"));
    names
}

pub fn layer_ple_proj_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ple_proj.weight")];
    names.extend(with_layer_suffix(layer, "per_layer_projection.weight"));
    names
}

pub fn layer_ple_post_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ple_post_norm.weight")];
    names.extend(with_layer_suffix(layer, "post_per_layer_input_norm.weight"));
    names
}

pub fn attn_q_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_q_norm.weight")];
    names.extend(with_layer_suffix(layer, "self_attn.q_norm.weight"));
    names.extend(with_layer_suffix(layer, "self_attn.query_layernorm.weight"));
    names
}

pub fn attn_k_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_k_norm.weight")];
    names.extend(with_layer_suffix(layer, "self_attn.k_norm.weight"));
    names.extend(with_layer_suffix(layer, "self_attn.key_layernorm.weight"));
    names
}

pub fn attn_v_norm_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.attn_v_norm.weight")];
    names.extend(with_layer_suffix(layer, "self_attn.v_norm.weight"));
    names.extend(with_layer_suffix(layer, "self_attn.value_layernorm.weight"));
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
    names.extend(with_layer_suffix(layer, "mlp.w1.weight"));
    names.extend(with_layer_suffix(layer, "feed_forward.w1.weight"));
    names
}

pub fn ffn_up_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_up.weight")];
    names.extend(with_layer_suffix(layer, "mlp.up_proj.weight"));
    names.extend(with_layer_suffix(layer, "mlp.w3.weight"));
    names.extend(with_layer_suffix(layer, "feed_forward.w3.weight"));
    names
}

pub fn ffn_down_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_down.weight")];
    names.extend(with_layer_suffix(layer, "mlp.down_proj.weight"));
    names.extend(with_layer_suffix(layer, "mlp.w2.weight"));
    names.extend(with_layer_suffix(layer, "feed_forward.w2.weight"));
    names
}

/// LFM2 short-conv `in_proj` (projects to 3×hidden: B, C, x).
pub fn conv_in_proj_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.conv_in.weight")];
    names.extend(with_layer_suffix(layer, "conv.in_proj.weight"));
    names
}

pub fn conv_out_proj_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.conv_out.weight")];
    names.extend(with_layer_suffix(layer, "conv.out_proj.weight"));
    names
}

/// Depthwise Conv1d weight; often stored as `[hidden, kernel]` after squeeze.
pub fn conv_kernel_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.conv_kernel.weight")];
    names.extend(with_layer_suffix(layer, "conv.conv.weight"));
    names
}

/// MoE router / gate (logits over experts).
pub fn moe_router_names(layer: usize) -> Vec<String> {
    let mut names = vec![
        format!("blk.{layer}.ffn_gate_inp.weight"),
        format!("blk.{layer}.moe_gate.weight"),
    ];
    names.extend(with_layer_suffix(layer, "block_sparse_moe.gate.weight"));
    names.extend(with_layer_suffix(layer, "mlp.gate.weight"));
    names.extend(with_layer_suffix(layer, "feed_forward.gate.weight"));
    names
}

pub fn moe_expert_gate_names(layer: usize, expert: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_gate.{expert}.weight")];
    names.extend(with_layer_suffix(
        layer,
        &format!("block_sparse_moe.experts.{expert}.w1.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("block_sparse_moe.experts.{expert}.gate_proj.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("mlp.experts.{expert}.gate_proj.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("mlp.experts.{expert}.w1.weight"),
    ));
    names
}

pub fn moe_expert_up_names(layer: usize, expert: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_up.{expert}.weight")];
    names.extend(with_layer_suffix(
        layer,
        &format!("block_sparse_moe.experts.{expert}.w3.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("block_sparse_moe.experts.{expert}.up_proj.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("mlp.experts.{expert}.up_proj.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("mlp.experts.{expert}.w3.weight"),
    ));
    names
}

pub fn moe_expert_down_names(layer: usize, expert: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.ffn_down.{expert}.weight")];
    names.extend(with_layer_suffix(
        layer,
        &format!("block_sparse_moe.experts.{expert}.w2.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("block_sparse_moe.experts.{expert}.down_proj.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("mlp.experts.{expert}.down_proj.weight"),
    ));
    names.extend(with_layer_suffix(
        layer,
        &format!("mlp.experts.{expert}.w2.weight"),
    ));
    names
}

/// Qwen3.5 / Bonsai Gated DeltaNet projections.
pub fn linear_in_proj_qkvz_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.linear_attn.in_proj_qkvz.weight")];
    names.extend(with_layer_suffix(layer, "linear_attn.in_proj_qkvz.weight"));
    names.extend(with_layer_suffix(
        layer,
        "self_attn.in_proj_qkvz.weight",
    ));
    names
}

pub fn linear_in_proj_ba_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.linear_attn.in_proj_ba.weight")];
    names.extend(with_layer_suffix(layer, "linear_attn.in_proj_ba.weight"));
    names.extend(with_layer_suffix(layer, "self_attn.in_proj_ba.weight"));
    names
}

pub fn linear_conv1d_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.linear_attn.conv1d.weight")];
    names.extend(with_layer_suffix(layer, "linear_attn.conv1d.weight"));
    names.extend(with_layer_suffix(layer, "self_attn.conv1d.weight"));
    names
}

pub fn linear_out_proj_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.linear_attn.out_proj.weight")];
    names.extend(with_layer_suffix(layer, "linear_attn.out_proj.weight"));
    names.extend(with_layer_suffix(layer, "self_attn.out_proj.weight"));
    names
}

pub fn linear_a_log_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.linear_attn.A_log")];
    names.extend(with_layer_suffix(layer, "linear_attn.A_log"));
    names.extend(with_layer_suffix(layer, "self_attn.A_log"));
    names
}

pub fn linear_dt_bias_names(layer: usize) -> Vec<String> {
    let mut names = vec![format!("blk.{layer}.linear_attn.dt_bias")];
    names.extend(with_layer_suffix(layer, "linear_attn.dt_bias"));
    names.extend(with_layer_suffix(layer, "self_attn.dt_bias"));
    names
}

/// Vision projector / merger (first hit wins at load).
pub fn vision_proj_names() -> Vec<&'static str> {
    vec![
        "mm_projector.weight",
        "vision_projector.weight",
        "multi_modal_projector.linear.weight",
        "multi_modal_projector.weight",
        "model.visual.merger.mlp.0.weight",
        "visual.merger.mlp.0.weight",
        "model.mm_projector.weight",
        "vision_tower.projector.weight",
    ]
}

/// VLA action / policy head.
pub fn action_head_names() -> Vec<&'static str> {
    vec![
        "action_head.weight",
        "action_out.weight",
        "model.action_head.weight",
        "model.action_out_proj.weight",
        "policy.action_out.weight",
    ]
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
            .any(|s| s == "model.layers.0.operator_norm.weight"));
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
        assert!(embed_per_layer_names()
            .contains(&"model.language_model.embed_tokens_per_layer.weight"));
        assert!(attn_post_norm_names(0)
            .iter()
            .any(|s| s == "model.layers.0.post_attention_layernorm.weight"));
        assert!(ffn_post_norm_names(0)
            .iter()
            .any(|s| s.contains("post_feedforward_layernorm")));
        assert!(layer_ple_gate_names(0)
            .iter()
            .any(|s| s.contains("per_layer_input_gate.weight")));
    }
}
