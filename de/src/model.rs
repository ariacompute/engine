//! Candle DecisionModel: ModernBERT encoder + Laya typed MASK head.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{layer_norm, linear, linear_no_bias, ops, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert::{Config as ModernBertConfig, ModernBert};
use serde::Deserialize;

use ariacompute_core::error::{AfmError, Result};

/// Load encoder/config.json into candle ModernBERT config (maps HF rope_parameters).
pub fn load_modernbert_config(encoder_dir: &Path) -> Result<ModernBertConfig> {
    let path = encoder_dir.join("config.json");
    let text = fs::read_to_string(&path).map_err(|e| {
        AfmError::msg(format!("read {}: {e}", path.display()))
    })?;
    let hf: HfModernBertConfig = serde_json::from_str(&text)?;
    Ok(hf.into_candle())
}

#[derive(Debug, Deserialize)]
struct HfModernBertConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_eps")]
    norm_eps: f64,
    pad_token_id: u32,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    #[serde(default)]
    global_rope_theta: Option<f64>,
    #[serde(default)]
    local_rope_theta: Option<f64>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default)]
    full_attention: Option<RopeTheta>,
    #[serde(default)]
    sliding_attention: Option<RopeTheta>,
}

#[derive(Debug, Deserialize)]
struct RopeTheta {
    rope_theta: f64,
}

fn default_eps() -> f64 {
    1e-5
}

impl HfModernBertConfig {
    fn into_candle(self) -> ModernBertConfig {
        let global = self
            .global_rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|r| r.full_attention.as_ref())
                    .map(|t| t.rope_theta)
            })
            .unwrap_or(160_000.0);
        let local = self
            .local_rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|r| r.sliding_attention.as_ref())
                    .map(|t| t.rope_theta)
            })
            .unwrap_or(10_000.0);
        let eps = if self.layer_norm_eps > 0.0 {
            self.layer_norm_eps
        } else {
            self.norm_eps
        };
        ModernBertConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            layer_norm_eps: eps,
            pad_token_id: self.pad_token_id,
            global_attn_every_n_layers: self.global_attn_every_n_layers,
            global_rope_theta: global,
            local_attention: self.local_attention,
            local_rope_theta: local,
            classifier_config: None,
        }
    }
}

/// Remap Laya `encoder.*` → candle `model.*`; keep head/scorer/type_emb keys.
fn load_weight_map(weights: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let raw = candle_core::safetensors::load(weights, device)
        .map_err(|e| AfmError::msg(format!("safetensors {}: {e}", weights.display())))?;
    let mut out = HashMap::with_capacity(raw.len());
    for (k, t) in raw {
        let key = if let Some(rest) = k.strip_prefix("encoder.") {
            format!("model.{rest}")
        } else {
            k
        };
        out.insert(key, t);
    }
    Ok(out)
}

/// PyTorch-compatible TransformerEncoderLayer (norm_first, ReLU FFN, batch_first).
struct EncoderLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl EncoderLayer {
    fn load(vb: VarBuilder<'_>, hidden: usize, n_heads: usize) -> CandleResult<Self> {
        let head_dim = hidden / n_heads;
        let norm1 = layer_norm(hidden, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(hidden, 1e-5, vb.pp("norm2"))?;
        let in_w = vb.get((3 * hidden, hidden), "self_attn.in_proj_weight")?;
        let in_b = vb.get(3 * hidden, "self_attn.in_proj_bias")?;
        let in_proj = Linear::new(in_w, Some(in_b));
        let out_proj = linear(hidden, hidden, vb.pp("self_attn.out_proj"))?;
        let linear1 = linear(hidden, 4 * hidden, vb.pp("linear1"))?;
        let linear2 = linear(4 * hidden, hidden, vb.pp("linear2"))?;
        Ok(Self {
            norm1,
            norm2,
            in_proj,
            out_proj,
            linear1,
            linear2,
            n_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, key_padding_mask: &Tensor) -> CandleResult<Tensor> {
        // key_padding_mask: (B, L) bool-as-u8/f32 where 1 = pad (ignore)
        let residual = xs;
        let xs = xs.apply(&self.norm1)?;
        let xs = self.mha(&xs, key_padding_mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let ys = xs
            .apply(&self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        residual + ys
    }

    fn mha(&self, xs: &Tensor, key_padding_mask: &Tensor) -> CandleResult<Tensor> {
        let (b, l, h) = xs.dims3()?;
        let qkv = xs.apply(&self.in_proj)?; // (B, L, 3H)
        let qkv = qkv.reshape((b, l, 3, self.n_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, B, heads, L, dim)
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        // mask: (B, 1, 1, L) u8 — nonzero = pad → add -inf
        let mask = key_padding_mask.unsqueeze(1)?.unsqueeze(2)?; // U8
        let neg_inf = Tensor::full(f32::NEG_INFINITY, mask.shape(), xs.device())?
            .to_dtype(attn.dtype())?;
        let zeros = Tensor::zeros(mask.shape(), attn.dtype(), xs.device())?;
        let add = mask.where_cond(&neg_inf, &zeros)?;
        let attn = attn.broadcast_add(&add)?;
        let attn = ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?; // (B, heads, L, dim)
        let out = out
            .transpose(1, 2)?
            .reshape((b, l, h))?;
        out.apply(&self.out_proj)
    }
}

struct ScorerMlp {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl ScorerMlp {
    fn load(vb: VarBuilder<'_>, hidden: usize) -> CandleResult<Self> {
        let norm = layer_norm(hidden, 1e-5, vb.pp("0"))?;
        let fc1 = linear(hidden, hidden, vb.pp("1"))?;
        let fc2 = linear(hidden, 1, vb.pp("3"))?;
        Ok(Self { norm, fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = xs.apply(&self.norm)?.apply(&self.fc1)?.gelu_erf()?;
        xs.apply(&self.fc2)
    }
}

/// Full Laya DecisionModel (encoder + typed head + scorer).
pub struct DecisionModel {
    encoder: ModernBert,
    head: Vec<EncoderLayer>,
    type_emb: Tensor,
    scorer: ScorerMlp,
    device: Device,
}

impl DecisionModel {
    pub fn load(encoder_dir: &Path, weights: &Path, head_layers: usize) -> Result<Self> {
        let device = Device::Cpu;
        let cfg = load_modernbert_config(encoder_dir)?;
        let tensors = load_weight_map(weights, &device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let encoder = ModernBert::load(vb.clone(), &cfg)
            .map_err(|e| AfmError::msg(format!("ModernBert load: {e}")))?;
        let hidden = cfg.hidden_size;
        let n_heads = (hidden / 64).max(1);
        let mut head = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            let layer = EncoderLayer::load(vb.pp(format!("head.layers.{i}")), hidden, n_heads)
                .map_err(|e| AfmError::msg(format!("head.layers.{i}: {e}")))?;
            head.push(layer);
        }
        let type_emb = vb
            .get((3, hidden), "type_emb.weight")
            .map_err(|e| AfmError::msg(format!("type_emb: {e}")))?;
        let scorer = ScorerMlp::load(vb.pp("scorer"), hidden)
            .map_err(|e| AfmError::msg(format!("scorer: {e}")))?;
        Ok(Self {
            encoder,
            head,
            type_emb,
            scorer,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Forward one packed sequence; returns option logits (K,).
    pub fn forward_logits(
        &self,
        input_ids: &[u32],
        mask_positions: &[usize],
        type_id: usize,
    ) -> Result<Vec<f32>> {
        if mask_positions.is_empty() {
            return Err(AfmError::msg("no MASK positions to score"));
        }
        if type_id > 2 {
            return Err(AfmError::msg(format!("bad type_id {type_id}")));
        }
        self.forward_logits_inner(input_ids, mask_positions, type_id)
            .map_err(|e| AfmError::msg(format!("DecisionModel forward: {e}")))
    }

    fn forward_logits_inner(
        &self,
        input_ids: &[u32],
        mask_positions: &[usize],
        type_id: usize,
    ) -> CandleResult<Vec<f32>> {
        let l = input_ids.len();
        let ids = Tensor::from_vec(input_ids.to_vec(), (1, l), &self.device)?;
        let attn = Tensor::ones((1, l), DType::U8, &self.device)?;
        let mut h = self.encoder.forward(&ids, &attn)?; // (1, L, H)
        let emb = self.type_emb.i(type_id)?; // (H,)
        h = h.broadcast_add(&emb)?;
        // pad mask all-false for unpadded batch
        let pad = Tensor::zeros((1, l), DType::U8, &self.device)?;
        for layer in &self.head {
            h = layer.forward(&h, &pad)?;
        }
        let mut mask_h = Vec::with_capacity(mask_positions.len());
        for &pos in mask_positions {
            if pos >= l {
                candle_core::bail!("mask pos {pos} >= seq len {l}");
            }
            mask_h.push(h.i((0, pos, ..))?.unsqueeze(0)?); // (1, H)
        }
        let mask_h = Tensor::cat(&mask_h, 0)?.unsqueeze(0)?; // (1, K, H)
        let logits = self.scorer.forward(&mask_h)?.squeeze(D::Minus1)?.squeeze(0)?; // (K,)
        logits.to_vec1::<f32>()
    }
}

pub fn cpu() -> Device {
    Device::Cpu
}

/// Typed decision head over MASK hidden states (unit-test helper).
pub struct DecisionHead {
    pub type_emb: Tensor,
    pub scorer: Linear,
}

impl DecisionHead {
    pub fn new(vb: VarBuilder<'_>, hidden: usize, n_types: usize) -> CandleResult<Self> {
        let type_emb = vb.get((n_types, hidden), "type_emb")?;
        let scorer = linear_no_bias(hidden, 1, vb.pp("scorer"))?;
        Ok(Self { type_emb, scorer })
    }

    pub fn forward(&self, mask_h: &Tensor, type_id: usize) -> CandleResult<Tensor> {
        let emb = self.type_emb.i(type_id)?;
        let h = mask_h.broadcast_add(&emb)?;
        let logits = self.scorer.forward(&h)?.squeeze(D::Minus1)?;
        Ok(logits)
    }
}

/// Shared handle for serve / decide.
pub type SharedDecisionModel = Arc<DecisionModel>;
