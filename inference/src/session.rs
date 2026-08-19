use crate::bundle::{load_bundle, Bundle, LoadedWeight};
use crate::chat::{apply_chat_template, strip_assistant_visible, ChatTurn};
use crate::family::{effective_rope_theta, graph_hook, require_runnable, ArchClass, Family};
use crate::tokenizer::{decode_placeholders, encode_naive, BundleTokenizer};
use crate::multimodal::asr_transcribe_pcm16le;
use crate::tensor_names::{
    action_head_names, attn_k_names, attn_k_norm_names, attn_norm_names, attn_o_names, attn_q_names,
    attn_q_norm_names, attn_v_names, conv_in_proj_names, conv_kernel_names, conv_out_proj_names,
    emb_names, ffn_down_names, ffn_gate_names, ffn_norm_names, ffn_up_names, linear_a_log_names,
    linear_conv1d_names, linear_dt_bias_names, linear_in_proj_ba_names, linear_in_proj_qkvz_names,
    linear_out_proj_names, moe_expert_down_names, moe_expert_gate_names, moe_expert_up_names,
    moe_router_names, output_names, output_norm_names, vision_proj_names,
};
use aria_kernel::{
    attention, gated_delta_step, geglu, hdm_linear, linear, moe_topk_route, rms_norm, rms_norm_gemma,
    rope_half, short_conv_step, silu_vec, softplus, swiglu, EngineError, GatedDeltaStep,
};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GenerateOpts {
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        Self {
            max_tokens: 16,
            temperature: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub text: String,
}

#[derive(Clone)]
struct MatWeight {
    data: Vec<f32>,
    hdm_seed: Option<i64>,
}

impl MatWeight {
    fn from_loaded(w: LoadedWeight) -> Self {
        Self {
            data: w.data,
            hdm_seed: w.hdm_seed,
        }
    }

    fn gemm(&self, x: &[f32], out_f: usize, in_f: usize) -> Result<Vec<f32>, EngineError> {
        if let Some(seed) = self.hdm_seed {
            hdm_linear(x, &self.data, out_f, in_f, Some(seed))
        } else {
            linear(x, &self.data, out_f, in_f)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AttnKind {
    Sliding,
    Full,
}

/// Persistent caches for autoregressive decode (KV / short-conv / DeltaNet).
struct DecodeState {
    k_caches: Vec<Vec<f32>>,
    v_caches: Vec<Vec<f32>>,
    last_kv_src: HashMap<AttnKind, usize>,
    conv_states: Vec<Option<Vec<f32>>>,
    delta_states: Vec<Option<Vec<f32>>>,
    /// Next absolute position (RoPE / seq index).
    pos: usize,
}

#[derive(Clone)]
struct AttnWeights {
    wq: MatWeight,
    /// None on Gemma-4 KV-consumer layers (reuse producer cache).
    wk: Option<MatWeight>,
    wv: Option<MatWeight>,
    wo: MatWeight,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    kind: AttnKind,
}

#[derive(Clone)]
struct ConvWeights {
    in_proj: MatWeight,
    out_proj: MatWeight,
    /// Depthwise kernel `[hidden * kernel]`.
    kernel: Vec<f32>,
    kernel_size: usize,
}

/// Qwen3.5 / Bonsai Gated DeltaNet (linear attention).
#[derive(Clone)]
struct DeltaWeights {
    qkvz: MatWeight,
    ba: MatWeight,
    conv: Vec<f32>,
    conv_k: usize,
    out_proj: MatWeight,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    n_k_heads: usize,
    n_v_heads: usize,
    head_k: usize,
    head_v: usize,
}

#[derive(Clone)]
enum LayerOp {
    Attn(AttnWeights),
    Conv(ConvWeights),
    Linear(DeltaWeights),
}

#[derive(Clone)]
struct ExpertWeights {
    gate: MatWeight,
    up: MatWeight,
    down: MatWeight,
}

#[derive(Clone)]
enum FfnWeights {
    Dense {
        gate: MatWeight,
        up: MatWeight,
        down: MatWeight,
    },
    MoE {
        router: MatWeight,
        experts: Vec<ExpertWeights>,
        top_k: usize,
        use_sigmoid: bool,
    },
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    op: LayerOp,
    ffn: FfnWeights,
}

struct ModelWeights {
    emb: MatWeight,
    layers: Vec<LayerWeights>,
    output_norm: Vec<f32>,
    output: MatWeight,
    vision: Option<MatWeight>,
    action: Option<MatWeight>,
}

pub struct Session {
    family: Family,
    bundle: Bundle,
    weights: ModelWeights,
    conf: crate::bundle::ModelConfig,
    use_gemma_norm: bool,
    use_geglu: bool,
    tokenizer: Option<BundleTokenizer>,
    /// Cleared at the start of each `generate`; reused across decode steps.
    decode: Option<DecodeState>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("family", &self.family)
            .field("model", &self.conf.hidden_size)
            .finish()
    }
}

pub struct SessionBuilder {
    path: Option<std::path::PathBuf>,
    family_path: String,
}

impl SessionBuilder {
    pub fn new() -> Self {
        Self {
            path: None,
            family_path: "gemma/gemma-4-e2b-it".into(),
        }
    }

    pub fn model(mut self, path: impl AsRef<Path>) -> Self {
        self.path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn family(mut self, path: impl Into<String>) -> Self {
        self.family_path = path.into();
        self
    }

    pub fn build(self) -> Result<Session, EngineError> {
        let family = require_runnable(&self.family_path)?;
        let _hook = graph_hook(family.arch);
        let path = self
            .path
            .ok_or_else(|| EngineError::InvalidParam("model path required".into()))?;
        let bundle = load_bundle(&path)?;
        reject_unsupported_geometry(&bundle.model, family)?;
        let tokenizer = BundleTokenizer::try_load(&path)?;
        let weights = materialize(&bundle, family)?;
        let mut conf = bundle.model.clone();
        conf.rope_theta = effective_rope_theta(family.path(), conf.rope_theta);
        let act = conf
            .hidden_act
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase();
        let use_geglu = act.contains("gelu");
        // Never apply Gemma (1+w) RMSNorm to Qwen/LFM/… even if act string is odd.
        let use_gemma_norm = family.path().contains("gemma");
        Ok(Session {
            family,
            bundle,
            weights,
            conf,
            use_gemma_norm,
            use_geglu,
            tokenizer,
            decode: None,
        })
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn reject_unsupported_geometry(
    conf: &crate::bundle::ModelConfig,
    family: Family,
) -> Result<(), EngineError> {
    let path = family.path();
    // Hybrid linear-attn families must declare layer_types so we don't SDPA-fake.
    if (path.contains("qwen3.5") || path.contains("bonsai")) && conf.layer_types.is_none() {
        return Err(EngineError::Unsupported(format!(
            "{path}: requires model.layer_types for Gated DeltaNet / full_attention mix"
        )));
    }
    // MoE families need explicit expert count (router/experts materialize from that).
    if family.is_moe() && conf.num_experts.unwrap_or(0) == 0 {
        return Err(EngineError::Unsupported(format!(
            "{path}: MoE family requires model.num_experts > 0 in bundle config"
        )));
    }
    Ok(())
}

fn layer_type_str(conf: &crate::bundle::ModelConfig, layer: usize) -> String {
    conf.layer_types
        .as_ref()
        .and_then(|t| t.get(layer))
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "full_attention".into())
}

fn layer_is_conv(conf: &crate::bundle::ModelConfig, layer: usize) -> bool {
    layer_type_str(conf, layer).contains("conv")
}

fn layer_is_linear(conf: &crate::bundle::ModelConfig, layer: usize) -> bool {
    let t = layer_type_str(conf, layer);
    t.contains("linear_attention") || t.contains("delta")
}

fn attn_kind(conf: &crate::bundle::ModelConfig, layer: usize) -> AttnKind {
    if layer_type_str(conf, layer).contains("sliding") {
        AttnKind::Sliding
    } else {
        AttnKind::Full
    }
}

fn is_kv_consumer(conf: &crate::bundle::ModelConfig, layer: usize) -> bool {
    let n = conf.num_kv_shared_layers.unwrap_or(0);
    n > 0 && layer >= conf.num_layers.saturating_sub(n)
}

fn materialize(b: &Bundle, family: Family) -> Result<ModelWeights, EngineError> {
    fn any_mat(b: &Bundle, names: &[String]) -> Result<MatWeight, EngineError> {
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        Ok(MatWeight::from_loaded(b.weight_loaded_any(&refs)?))
    }
    fn any_vec(b: &Bundle, names: &[String]) -> Result<Vec<f32>, EngineError> {
        Ok(any_mat(b, names)?.data)
    }
    fn optional_vec(b: &Bundle, names: &[String]) -> Option<Vec<f32>> {
        any_vec(b, names).ok()
    }

    let m = &b.model;
    let hidden = m.hidden_size;
    let n_experts = m.num_experts.unwrap_or(0);
    let top_k = m.num_experts_per_tok.unwrap_or(1).max(1);
    // LFM MoE uses sigmoid routing; Mixtral/Inkling-style uses softmax.
    let use_sigmoid_router = n_experts > 0 && m.layer_types.is_some();

    let mut layers = Vec::with_capacity(m.num_layers);
    let mut prev_wk: Option<MatWeight> = None;
    let mut prev_wv: Option<MatWeight> = None;
    for layer in 0..m.num_layers {
        let attn_norm = any_vec(b, &attn_norm_names(layer))?;
        let ffn_norm = any_vec(b, &ffn_norm_names(layer))?;

        let op = if layer_is_conv(m, layer) {
            let in_proj = any_mat(b, &conv_in_proj_names(layer))?;
            let out_proj = any_mat(b, &conv_out_proj_names(layer))?;
            let kw = any_mat(b, &conv_kernel_names(layer))?;
            let kernel_size = m.conv_l_cache.unwrap_or(3).max(1);
            if kw.data.len() % hidden != 0 {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} conv kernel len {} not divisible by hidden {hidden}",
                    kw.data.len()
                )));
            }
            let inferred_k = kw.data.len() / hidden;
            let kernel_size = if inferred_k > 0 { inferred_k } else { kernel_size };
            // Accept [H,K] or squeezed [H,1,K] (same flat length).
            if kw.data.len() != hidden * kernel_size {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} conv kernel len {} != hidden*kernel {hidden}*{kernel_size}",
                    kw.data.len()
                )));
            }
            let kernel = kw.data;
            if in_proj.data.len() != 3 * hidden * hidden {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} conv in_proj len {} != 3*hidden*hidden",
                    in_proj.data.len()
                )));
            }
            if out_proj.data.len() != hidden * hidden {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} conv out_proj len {} != hidden*hidden",
                    out_proj.data.len()
                )));
            }
            LayerOp::Conv(ConvWeights {
                in_proj,
                out_proj,
                kernel,
                kernel_size,
            })
        } else if layer_is_linear(m, layer) {
            let qkvz = any_mat(b, &linear_in_proj_qkvz_names(layer))?;
            let ba = any_mat(b, &linear_in_proj_ba_names(layer))?;
            let conv_w = any_mat(b, &linear_conv1d_names(layer))?;
            let out_proj = any_mat(b, &linear_out_proj_names(layer))?;
            let a_log = any_vec(b, &linear_a_log_names(layer))?;
            let dt_bias = any_vec(b, &linear_dt_bias_names(layer))?;
            let n_v_heads = a_log.len();
            if n_v_heads == 0 || dt_bias.len() != n_v_heads {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} A_log/dt_bias head mismatch"
                )));
            }
            if ba.data.len() % hidden != 0 {
                return Err(EngineError::ShapeMismatch(
                    "linear in_proj_ba not divisible by hidden".into(),
                ));
            }
            if ba.data.len() / hidden != 2 * n_v_heads {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} in_proj_ba out {} != 2*n_v_heads {}",
                    ba.data.len() / hidden,
                    2 * n_v_heads
                )));
            }
            if qkvz.data.len() % hidden != 0 {
                return Err(EngineError::ShapeMismatch(
                    "linear in_proj_qkvz not divisible by hidden".into(),
                ));
            }
            let qkvz_out = qkvz.data.len() / hidden;
            // Equal k/v dims: qkvz = 2*key + 2*value = 4*key.
            if !qkvz_out.is_multiple_of(4) {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} qkvz out {qkvz_out} not divisible by 4"
                )));
            }
            let key_dim = qkvz_out / 4;
            let value_dim = key_dim;
            let n_k_heads = n_v_heads;
            if n_k_heads == 0
                || !key_dim.is_multiple_of(n_k_heads)
                || !value_dim.is_multiple_of(n_v_heads)
            {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} cannot infer DeltaNet head dims"
                )));
            }
            let head_k = key_dim / n_k_heads;
            let head_v = value_dim / n_v_heads;
            let conv_dim = key_dim * 2 + value_dim;
            if conv_w.data.len() % conv_dim != 0 {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} conv1d len {} not divisible by conv_dim {conv_dim}",
                    conv_w.data.len()
                )));
            }
            let conv_k = conv_w.data.len() / conv_dim;
            if out_proj.data.len() != hidden * value_dim {
                return Err(EngineError::ShapeMismatch(format!(
                    "layer {layer} linear out_proj len {} != hidden*value_dim",
                    out_proj.data.len()
                )));
            }
            LayerOp::Linear(DeltaWeights {
                qkvz,
                ba,
                conv: conv_w.data,
                conv_k,
                out_proj,
                a_log,
                dt_bias,
                n_k_heads,
                n_v_heads,
                head_k,
                head_v,
            })
        } else {
            let consumer = is_kv_consumer(m, layer);
            let (wk, wv) = if consumer {
                (None, None)
            } else {
                let wk = match any_mat(b, &attn_k_names(layer)) {
                    Ok(w) => {
                        prev_wk = Some(w.clone());
                        Some(w)
                    }
                    Err(e) => Some(prev_wk.clone().ok_or_else(|| {
                        EngineError::Format(format!(
                            "missing k_proj for layer {layer} and no prior KV to share ({e})"
                        ))
                    })?),
                };
                let wv = match any_mat(b, &attn_v_names(layer)) {
                    Ok(w) => {
                        prev_wv = Some(w.clone());
                        Some(w)
                    }
                    Err(e) => Some(prev_wv.clone().ok_or_else(|| {
                        EngineError::Format(format!(
                            "missing v_proj for layer {layer} and no prior KV to share ({e})"
                        ))
                    })?),
                };
                (wk, wv)
            };
            LayerOp::Attn(AttnWeights {
                wq: any_mat(b, &attn_q_names(layer))?,
                wk,
                wv,
                wo: any_mat(b, &attn_o_names(layer))?,
                q_norm: optional_vec(b, &attn_q_norm_names(layer)),
                k_norm: optional_vec(b, &attn_k_norm_names(layer)),
                kind: attn_kind(m, layer),
            })
        };

        let ffn = if n_experts > 0 {
            match any_mat(b, &moe_router_names(layer)) {
                Ok(router) => {
                    if router.data.len() != n_experts * hidden {
                        return Err(EngineError::ShapeMismatch(format!(
                            "layer {layer} MoE router len {} != num_experts*hidden {n_experts}*{hidden}",
                            router.data.len()
                        )));
                    }
                    let mut experts = Vec::with_capacity(n_experts);
                    for e in 0..n_experts {
                        experts.push(ExpertWeights {
                            gate: any_mat(b, &moe_expert_gate_names(layer, e))?,
                            up: any_mat(b, &moe_expert_up_names(layer, e))?,
                            down: any_mat(b, &moe_expert_down_names(layer, e))?,
                        });
                    }
                    FfnWeights::MoE {
                        router,
                        experts,
                        top_k,
                        use_sigmoid: use_sigmoid_router,
                    }
                }
                Err(_) => {
                    // Dense FFN on this layer (e.g. LFM2-A1B first layers).
                    FfnWeights::Dense {
                        gate: any_mat(b, &ffn_gate_names(layer))?,
                        up: any_mat(b, &ffn_up_names(layer))?,
                        down: any_mat(b, &ffn_down_names(layer))?,
                    }
                }
            }
        } else {
            FfnWeights::Dense {
                gate: any_mat(b, &ffn_gate_names(layer))?,
                up: any_mat(b, &ffn_up_names(layer))?,
                down: any_mat(b, &ffn_down_names(layer))?,
            }
        };

        layers.push(LayerWeights {
            attn_norm,
            ffn_norm,
            op,
            ffn,
        });
    }
    let emb_n = emb_names();
    let out_norm_n = output_norm_names();
    let out_n = output_names();
    let vis_n: Vec<String> = vision_proj_names().iter().map(|s| (*s).to_string()).collect();
    let act_n: Vec<String> = action_head_names().iter().map(|s| (*s).to_string()).collect();
    let emb = MatWeight::from_loaded(b.weight_loaded_any(&emb_n)?);
    let output = if m.tie_word_embeddings.unwrap_or(false)
        || (family.path().contains("qwen3") && !family.path().contains("qwen3.5"))
    {
        // Qwen3-0.6B/1.7B tie lm_head to embed; prefer embed even if a separate
        // lm_head tensor exists (often a worse-quantized copy).
        emb.clone()
    } else {
        MatWeight::from_loaded(b.weight_loaded_any(&out_n)?)
    };
    Ok(ModelWeights {
        emb,
        layers,
        output_norm: b.weight_loaded_any(&out_norm_n)?.data,
        output,
        vision: any_mat(b, &vis_n).ok(),
        action: any_mat(b, &act_n).ok(),
    })
}

impl Session {
    pub fn family(&self) -> Family {
        self.family
    }

    pub fn model_id(&self) -> &str {
        self.family.path()
    }

    pub fn config(&self) -> &crate::bundle::ModelConfig {
        &self.conf
    }

    pub fn bundle(&self) -> &Bundle {
        &self.bundle
    }

    /// Greedy (temperature<=0) generation from prompt token ids.
    /// Prefills the prompt once, then runs one incremental decode step per new token.
    pub fn generate(&mut self, prompt: &[u32], opts: &GenerateOpts) -> Result<Generation, EngineError> {
        if opts.max_tokens == 0 {
            return Err(EngineError::InvalidParam("max_tokens must be > 0".into()));
        }
        let mut tokens: Vec<u32> = prompt.to_vec();
        if tokens.is_empty() {
            tokens.push(1);
        }
        self.decode = Some(self.fresh_decode_state());
        let result = (|| {
            let mut logits = Vec::new();
            for &tok in &tokens {
                logits = self.forward_step(tok)?;
            }
            let mut generated = Vec::new();
            for _ in 0..opts.max_tokens {
                let next = if opts.temperature <= 0.0 {
                    argmax(&logits)
                } else {
                    // Stage A: still greedy for determinism; temperature reserved.
                    argmax(&logits)
                };
                generated.push(next);
                tokens.push(next);
                if self.is_stop_id(next) {
                    generated.pop();
                    break;
                }
                logits = self.forward_step(next)?;
            }
            let text = self.decode_tokens(&generated);
            Ok(Generation {
                tokens: generated,
                text,
            })
        })();
        self.decode = None;
        result
    }

    /// Map token ids → UTF-8 via bundle `tokenizer.json` (byte-level when applicable).
    /// Falls back to `<id>` placeholders when no sidecar is present.
    pub fn decode_tokens(&self, ids: &[u32]) -> String {
        match &self.tokenizer {
            Some(tok) => {
                let raw = tok.decode_opts(ids, false);
                strip_assistant_visible(&raw)
            }
            None => decode_placeholders(ids),
        }
    }

    /// Encode with bundle `tokenizer.json` when present; else naive byte fallback.
    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        match &self.tokenizer {
            Some(tok) => match tok.encode(text) {
                Ok(ids) if !ids.is_empty() => ids,
                Ok(_) => encode_naive(text, self.conf.vocab_size as u32),
                Err(_) => encode_naive(text, self.conf.vocab_size as u32),
            },
            None => encode_naive(text, self.conf.vocab_size as u32),
        }
    }

    /// Encode OpenAI-style messages with the family / tokenizer chat template.
    pub fn encode_chat(&self, messages: &[ChatTurn]) -> Vec<u32> {
        let family = self
            .tokenizer
            .as_ref()
            .and_then(|t| t.chat_family_hint())
            .unwrap_or(self.family.path());
        let prompt = apply_chat_template(family, messages);
        self.encode_text(&prompt)
    }

    fn is_stop_id(&self, id: u32) -> bool {
        match &self.tokenizer {
            Some(t) => t.is_stop(id),
            None => id == 0,
        }
    }

    pub fn arch(&self) -> ArchClass {
        self.family.arch
    }

    pub fn graph_hook_name(&self) -> &'static str {
        graph_hook(self.family.arch)
    }

    /// Mean-pool token embeddings (stage C `/v1/embeddings`).
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        let toks = self.encode_text(text);
        let hidden = self.conf.hidden_size;
        let vocab = self.conf.vocab_size;
        let mut acc = vec![0.0f32; hidden];
        if toks.is_empty() {
            return Ok(acc);
        }
        for &tok in &toks {
            let tid = (tok as usize) % vocab;
            let row = &self.weights.emb.data[tid * hidden..(tid + 1) * hidden];
            for i in 0..hidden {
                acc[i] += row[i];
            }
        }
        let inv = 1.0 / toks.len() as f32;
        for v in &mut acc {
            *v *= inv;
        }
        Ok(acc)
    }

    /// Stage C VL: project RGB via bundle vision weights (no mean-pool stub).
    pub fn vision_prefix(
        &self,
        rgb: &[u8],
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>, EngineError> {
        if !matches!(self.family.arch, ArchClass::VL | ArchClass::VLA) {
            return Err(EngineError::Unsupported(format!(
                "vision_prefix not available for arch {:?}",
                self.family.arch
            )));
        }
        let Some(proj) = &self.weights.vision else {
            return Err(EngineError::Unsupported(format!(
                "{}: no vision projector tensor in bundle",
                self.family.path()
            )));
        };
        let hidden = self.conf.hidden_size;
        if hidden == 0 || proj.data.len() % hidden != 0 {
            return Err(EngineError::ShapeMismatch(
                "vision projector not divisible by hidden_size".into(),
            ));
        }
        let in_f = proj.data.len() / hidden;
        let need = height
            .checked_mul(width)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| EngineError::InvalidParam("vision size overflow".into()))?;
        if rgb.len() < need {
            return Err(EngineError::ShapeMismatch(format!(
                "rgb len {} < {}x{}x3",
                rgb.len(),
                height,
                width
            )));
        }
        let mut feat = vec![0.0f32; in_f];
        let pixels = height * width;
        if in_f == 3 {
            let mut acc = [0.0f32; 3];
            for p in 0..pixels {
                acc[0] += rgb[p * 3] as f32 / 255.0;
                acc[1] += rgb[p * 3 + 1] as f32 / 255.0;
                acc[2] += rgb[p * 3 + 2] as f32 / 255.0;
            }
            let s = 1.0 / pixels.max(1) as f32;
            feat[0] = acc[0] * s;
            feat[1] = acc[1] * s;
            feat[2] = acc[2] * s;
        } else {
            for i in 0..in_f {
                feat[i] = rgb[i % need] as f32 / 255.0;
            }
        }
        proj.gemm(&feat, hidden, in_f)
    }

    /// Stage C VLA: project last-token embedding with bundle action weights.
    pub fn predict_action(
        &self,
        prompt: &str,
        action_dim: usize,
    ) -> Result<Vec<f32>, EngineError> {
        if self.family.arch != ArchClass::VLA {
            return Err(EngineError::Unsupported(format!(
                "predict_action requires VLA, got {:?}",
                self.family.arch
            )));
        }
        if action_dim == 0 {
            return Err(EngineError::InvalidParam("action_dim must be > 0".into()));
        }
        let Some(head) = &self.weights.action else {
            return Err(EngineError::Unsupported(format!(
                "{}: no action head tensor in bundle",
                self.family.path()
            )));
        };
        let h = self.embed_text(prompt)?;
        let hidden = self.conf.hidden_size;
        if head.data.len() % hidden != 0 {
            return Err(EngineError::ShapeMismatch(
                "action head not divisible by hidden_size".into(),
            ));
        }
        let out_f = head.data.len() / hidden;
        if out_f != action_dim {
            return Err(EngineError::ShapeMismatch(format!(
                "action head out {out_f} != requested {action_dim}"
            )));
        }
        head.gemm(&h, out_f, hidden)
    }

    /// Stage C ASR stub bound to session vocab.
    pub fn transcribe_pcm16le(&self, pcm: &[u8]) -> Result<String, EngineError> {
        asr_transcribe_pcm16le(pcm, self.conf.vocab_size as u32)
    }

    fn norm(&self, x: &[f32], weight: &[f32]) -> Result<Vec<f32>, EngineError> {
        if self.use_gemma_norm {
            rms_norm_gemma(x, weight, 1e-6)
        } else {
            rms_norm(x, weight, 1e-6)
        }
    }

    fn apply_ffn(&self, layer: &LayerWeights, xn2: &[f32], hidden: usize) -> Result<Vec<f32>, EngineError> {
        match &layer.ffn {
            FfnWeights::Dense { gate, up, down } => {
                if gate.data.len() % hidden != 0 {
                    return Err(EngineError::ShapeMismatch(
                        "dense gate len not divisible by hidden".into(),
                    ));
                }
                let inter = gate.data.len() / hidden;
                if inter == 0 || up.data.len() != inter * hidden || down.data.len() != hidden * inter
                {
                    return Err(EngineError::ShapeMismatch(
                        "dense FFN weight shape mismatch".into(),
                    ));
                }
                let g = gate.gemm(xn2, inter, hidden)?;
                let u = up.gemm(xn2, inter, hidden)?;
                let h = if self.use_geglu {
                    geglu(&g, &u)?
                } else {
                    swiglu(&g, &u)?
                };
                down.gemm(&h, hidden, inter)
            }
            FfnWeights::MoE {
                router,
                experts,
                top_k,
                use_sigmoid,
            } => {
                let n_exp = experts.len();
                let logits = router.gemm(xn2, n_exp, hidden)?;
                let (ids, weights) = moe_topk_route(&logits, *top_k, *use_sigmoid)?;
                let mut acc = vec![0.0f32; hidden];
                for (ei, &w) in ids.iter().zip(weights.iter()) {
                    let ex = &experts[*ei];
                    if ex.gate.data.len() % hidden != 0 {
                        return Err(EngineError::ShapeMismatch(
                            "expert gate len not divisible by hidden".into(),
                        ));
                    }
                    let inter = ex.gate.data.len() / hidden;
                    let g = ex.gate.gemm(xn2, inter, hidden)?;
                    let u = ex.up.gemm(xn2, inter, hidden)?;
                    let h = swiglu(&g, &u)?;
                    let down = ex.down.gemm(&h, hidden, inter)?;
                    for i in 0..hidden {
                        acc[i] += w * down[i];
                    }
                }
                Ok(acc)
            }
        }
    }

    fn fresh_decode_state(&self) -> DecodeState {
        let hidden = self.conf.hidden_size;
        DecodeState {
            k_caches: (0..self.conf.num_layers).map(|_| Vec::new()).collect(),
            v_caches: (0..self.conf.num_layers).map(|_| Vec::new()).collect(),
            last_kv_src: HashMap::new(),
            conv_states: self
                .weights
                .layers
                .iter()
                .map(|layer| match &layer.op {
                    LayerOp::Conv(c) => {
                        let hist = c.kernel_size.saturating_sub(1);
                        Some(vec![0.0f32; hidden * hist])
                    }
                    LayerOp::Linear(d) => {
                        let conv_dim = d.n_k_heads * d.head_k * 2 + d.n_v_heads * d.head_v;
                        let hist = d.conv_k.saturating_sub(1);
                        Some(vec![0.0f32; conv_dim * hist])
                    }
                    LayerOp::Attn(_) => None,
                })
                .collect(),
            delta_states: self
                .weights
                .layers
                .iter()
                .map(|layer| match &layer.op {
                    LayerOp::Linear(d) => {
                        Some(vec![0.0f32; d.n_v_heads * d.head_k * d.head_v])
                    }
                    _ => None,
                })
                .collect(),
            pos: 0,
        }
    }

    /// Full-sequence forward (allocates fresh caches). Used by tests / parity checks.
    #[cfg(test)]
    fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        let mut state = self.fresh_decode_state();
        let mut logits = Vec::new();
        for &tok in tokens {
            logits = self.forward_step_with(&mut state, tok)?;
        }
        Ok(logits)
    }

    fn forward_step(&mut self, tok: u32) -> Result<Vec<f32>, EngineError> {
        let mut owned = self.decode.take().ok_or_else(|| {
            EngineError::InvalidParam("decode state missing; call generate/prefill first".into())
        })?;
        let logits = self.forward_step_with(&mut owned, tok);
        self.decode = Some(owned);
        logits
    }

    fn forward_step_with(
        &self,
        state: &mut DecodeState,
        tok: u32,
    ) -> Result<Vec<f32>, EngineError> {
        let hidden = self.conf.hidden_size;
        let n_heads = self.conf.num_attention_heads;
        let n_kv = self.conf.num_kv_heads;
        let vocab = self.conf.vocab_size;
        if hidden == 0 {
            return Err(EngineError::ShapeMismatch("hidden_size is 0".into()));
        }
        if self.weights.emb.data.len() < vocab.saturating_mul(hidden)
            || !self.weights.emb.data.len().is_multiple_of(hidden)
        {
            return Err(EngineError::ShapeMismatch(format!(
                "embedding length {} not compatible with vocab={vocab} hidden={hidden}",
                self.weights.emb.data.len()
            )));
        }
        let pos = state.pos;
        let tid = (tok as usize) % vocab;
        let mut x = vec![0.0f32; hidden];
        x.copy_from_slice(&self.weights.emb.data[tid * hidden..(tid + 1) * hidden]);

        for (li, layer) in self.weights.layers.iter().enumerate() {
            let xn = self.norm(&x, &layer.attn_norm)?;
            match &layer.op {
                LayerOp::Attn(attn) => {
                    if attn.wq.data.len() % hidden != 0 {
                        return Err(EngineError::ShapeMismatch(
                            "attn q proj weight not divisible by hidden_size".into(),
                        ));
                    }
                    let q_dim = attn.wq.data.len() / hidden;
                    if n_heads == 0 || !q_dim.is_multiple_of(n_heads) {
                        return Err(EngineError::ShapeMismatch(
                            "q_dim not divisible by num_attention_heads".into(),
                        ));
                    }
                    let head_dim = self
                        .conf
                        .head_dim
                        .filter(|d| *d > 0 && q_dim == n_heads * *d)
                        .unwrap_or(q_dim / n_heads);
                    if attn.wo.data.len() != hidden * q_dim {
                        return Err(EngineError::ShapeMismatch(
                            "attn output proj weight shape mismatch".into(),
                        ));
                    }
                    let mut q = attn.wq.gemm(&xn, q_dim, hidden)?;
                    if let Some(qn) = &attn.q_norm {
                        q = rms_norm(&q, qn, 1e-6)?;
                    }
                    rope_half(&mut q, head_dim, pos, self.conf.rope_theta)?;

                    let (k_src, v_src) = if let (Some(wk), Some(wv)) = (&attn.wk, &attn.wv) {
                        if wk.data.len() % hidden != 0 || wv.data.len() % hidden != 0 {
                            return Err(EngineError::ShapeMismatch(
                                "attn kv proj weight not divisible by hidden_size".into(),
                            ));
                        }
                        let k_dim = wk.data.len() / hidden;
                        let v_dim = wv.data.len() / hidden;
                        if k_dim != n_kv * head_dim || v_dim != n_kv * head_dim {
                            return Err(EngineError::ShapeMismatch(format!(
                                "kv dims {k_dim}/{v_dim} != n_kv*head_dim {}",
                                n_kv * head_dim
                            )));
                        }
                        let mut k = wk.gemm(&xn, k_dim, hidden)?;
                        let v = wv.gemm(&xn, v_dim, hidden)?;
                        if let Some(kn) = &attn.k_norm {
                            k = rms_norm(&k, kn, 1e-6)?;
                        }
                        rope_half(&mut k, head_dim, pos, self.conf.rope_theta)?;
                        state.k_caches[li].extend_from_slice(&k);
                        state.v_caches[li].extend_from_slice(&v);
                        state.last_kv_src.insert(attn.kind, li);
                        (li, li)
                    } else {
                        let src = state.last_kv_src.get(&attn.kind).copied().ok_or_else(|| {
                            EngineError::Format(format!(
                                "KV-consumer layer {li} has no producer of kind {:?}",
                                attn.kind
                            ))
                        })?;
                        (src, src)
                    };
                    let attn_out = attention(
                        &q,
                        &state.k_caches[k_src],
                        &state.v_caches[v_src],
                        n_heads,
                        n_kv,
                        head_dim,
                    )?;
                    let ao = attn.wo.gemm(&attn_out, hidden, q_dim)?;
                    for i in 0..hidden {
                        x[i] += ao[i];
                    }
                }
                LayerOp::Conv(conv) => {
                    let bcx = conv.in_proj.gemm(&xn, 3 * hidden, hidden)?;
                    let mut bx = vec![0.0f32; hidden];
                    let mut c_gate = vec![0.0f32; hidden];
                    for i in 0..hidden {
                        let b = bcx[i];
                        let c = bcx[hidden + i];
                        let xx = bcx[2 * hidden + i];
                        bx[i] = b * xx;
                        c_gate[i] = c;
                    }
                    let cstate = state.conv_states[li].as_mut().ok_or_else(|| {
                        EngineError::ShapeMismatch("missing conv state".into())
                    })?;
                    let conv_y =
                        short_conv_step(&bx, &conv.kernel, cstate, hidden, conv.kernel_size)?;
                    let mut y = vec![0.0f32; hidden];
                    for i in 0..hidden {
                        y[i] = c_gate[i] * conv_y[i];
                    }
                    let ao = conv.out_proj.gemm(&y, hidden, hidden)?;
                    for i in 0..hidden {
                        x[i] += ao[i];
                    }
                }
                LayerOp::Linear(dn) => {
                    let key_dim = dn.n_k_heads * dn.head_k;
                    let value_dim = dn.n_v_heads * dn.head_v;
                    let qkvz_out = 2 * key_dim + 2 * value_dim;
                    let mixed = dn.qkvz.gemm(&xn, qkvz_out, hidden)?;
                    let mut q = mixed[0..key_dim].to_vec();
                    let mut k = mixed[key_dim..2 * key_dim].to_vec();
                    let mut v = mixed[2 * key_dim..2 * key_dim + value_dim].to_vec();
                    let z = mixed[2 * key_dim + value_dim..].to_vec();
                    let mut qkv = Vec::with_capacity(key_dim * 2 + value_dim);
                    qkv.extend_from_slice(&q);
                    qkv.extend_from_slice(&k);
                    qkv.extend_from_slice(&v);
                    let conv_dim = qkv.len();
                    let cstate = state.conv_states[li].as_mut().ok_or_else(|| {
                        EngineError::ShapeMismatch("missing delta conv state".into())
                    })?;
                    let mut mixed_c =
                        short_conv_step(&qkv, &dn.conv, cstate, conv_dim, dn.conv_k)?;
                    silu_vec(&mut mixed_c);
                    q.copy_from_slice(&mixed_c[0..key_dim]);
                    k.copy_from_slice(&mixed_c[key_dim..2 * key_dim]);
                    v.copy_from_slice(&mixed_c[2 * key_dim..]);
                    let ba = dn.ba.gemm(&xn, 2 * dn.n_v_heads, hidden)?;
                    let mut beta = vec![0.0f32; dn.n_v_heads];
                    let mut g = vec![0.0f32; dn.n_v_heads];
                    for h in 0..dn.n_v_heads {
                        beta[h] = 1.0 / (1.0 + (-ba[h]).exp());
                        let alpha =
                            -dn.a_log[h].exp() * softplus(ba[dn.n_v_heads + h] + dn.dt_bias[h]);
                        g[h] = alpha.exp();
                    }
                    if dn.n_v_heads != dn.n_k_heads {
                        return Err(EngineError::Unsupported(
                            "DeltaNet GQA (n_v != n_k) not implemented".into(),
                        ));
                    }
                    let s = state.delta_states[li].as_mut().ok_or_else(|| {
                        EngineError::ShapeMismatch("missing delta recurrent state".into())
                    })?;
                    let mut core = gated_delta_step(GatedDeltaStep {
                        q: &q,
                        k: &k,
                        v: &v,
                        g: &g,
                        beta: &beta,
                        state: s,
                        n_heads: dn.n_v_heads,
                        dk: dn.head_k,
                        dv: dn.head_v,
                    })?;
                    // RMSNormGated approx: rms(core) * silu(z)
                    let ones = vec![1.0f32; dn.head_v];
                    core = rms_norm(&core, &ones, 1e-6)?;
                    let mut z_act = z;
                    silu_vec(&mut z_act);
                    for i in 0..core.len() {
                        core[i] *= z_act[i];
                    }
                    let ao = dn.out_proj.gemm(&core, hidden, value_dim)?;
                    for i in 0..hidden {
                        x[i] += ao[i];
                    }
                }
            }
            let xn2 = self.norm(&x, &layer.ffn_norm)?;
            let down = self.apply_ffn(layer, &xn2, hidden)?;
            for i in 0..hidden {
                x[i] += down[i];
            }
        }
        state.pos += 1;
        let xn = self.norm(&x, &self.weights.output_norm)?;
        if !self.weights.output.data.len().is_multiple_of(hidden) {
            return Err(EngineError::ShapeMismatch(format!(
                "lm_head len {} not divisible by hidden {hidden}",
                self.weights.output.data.len()
            )));
        }
        let out_rows = self.weights.output.data.len() / hidden;
        self.weights.output.gemm(&xn, out_rows, hidden)
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

/// Optional confidence heuristic for hybrid: mean max-softmax over last logits proxy.
pub fn confidence_from_logits(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut maxp = 0.0f32;
    for &x in logits {
        let e = (x - m).exp();
        sum += e;
        if e > maxp {
            maxp = e;
        }
    }
    if sum > 0.0 {
        maxp / sum
    } else {
        0.0
    }
}

#[allow(dead_code)]
pub fn cache_shapes_ok(cache: &HashMap<usize, Vec<f32>>, kv_dim: usize) -> bool {
    cache.values().all(|v| v.len().is_multiple_of(kv_dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{arch_class_representatives, graph_hook, lookup_family, require_stage_b};
    use crate::fixture::write_tiny_q4_bundle;
    use serde_json::{json, Value};

    #[test]
    fn generate_tokens() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("gemma/gemma-4-e2b-it")
            .build()
            .unwrap();
        let prompt = s.encode_text("hi");
        let gen = s
            .generate(
                &prompt,
                &GenerateOpts {
                    max_tokens: 4,
                    temperature: 0.0,
                },
            )
            .unwrap();
        assert!(!gen.tokens.is_empty());
        assert!(!gen.text.is_empty());
    }

    #[test]
    fn materialize_accepts_hf_tensor_names() {
        // Minimal HF-named raw bundle matching Qwen-style paths.
        let dir = tempfile::tempdir().unwrap();
        let hidden = 8usize;
        let layers = 1usize;
        let inter = 16usize;
        let vocab = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 4usize; // q_dim = 8, k_dim = 4
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv * head_dim;

        let mut tensors = serde_json::Map::new();
        let mut bin = Vec::new();
        let mut add_raw = |name: &str, shape: Vec<usize>, data: &[f32]| {
            let offset = bin.len();
            for &v in data {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let nbytes = data.len() * 4;
            let mut meta = serde_json::Map::new();
            meta.insert("kind".into(), json!("raw"));
            meta.insert("dtype".into(), json!("f32"));
            meta.insert("shape".into(), json!(shape));
            meta.insert("offsets".into(), json!({ "data": [offset, nbytes] }));
            tensors.insert(name.to_string(), Value::Object(meta));
        };
        let emb: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.01).collect();
        add_raw("model.embed_tokens.weight", vec![vocab, hidden], &emb);
        let n1 = vec![1.0f32; hidden];
        add_raw("model.layers.0.input_layernorm.weight", vec![hidden], &n1);
        add_raw(
            "model.layers.0.post_attention_layernorm.weight",
            vec![hidden],
            &n1,
        );
        let wq = vec![0.01f32; q_dim * hidden];
        let wk = vec![0.01f32; k_dim * hidden];
        let wv = vec![0.01f32; k_dim * hidden];
        let wo = vec![0.01f32; hidden * q_dim];
        add_raw(
            "model.layers.0.self_attn.q_proj.weight",
            vec![q_dim, hidden],
            &wq,
        );
        add_raw(
            "model.layers.0.self_attn.k_proj.weight",
            vec![k_dim, hidden],
            &wk,
        );
        add_raw(
            "model.layers.0.self_attn.v_proj.weight",
            vec![k_dim, hidden],
            &wv,
        );
        add_raw(
            "model.layers.0.self_attn.o_proj.weight",
            vec![hidden, q_dim],
            &wo,
        );
        let g = vec![0.01f32; inter * hidden];
        let d = vec![0.01f32; hidden * inter];
        add_raw(
            "model.layers.0.mlp.gate_proj.weight",
            vec![inter, hidden],
            &g,
        );
        add_raw("model.layers.0.mlp.up_proj.weight", vec![inter, hidden], &g);
        add_raw(
            "model.layers.0.mlp.down_proj.weight",
            vec![hidden, inter],
            &d,
        );
        add_raw("model.norm.weight", vec![hidden], &n1);
        add_raw("lm_head.weight", vec![vocab, hidden], &emb);

        let cfg = json!({
            "format": "aria-quant-bundle",
            "format_version": 2,
            "quantization": "test",
            "group_size_default": 32,
            "hadamard_seed": 0,
            "model": {
                "hidden_size": hidden,
                "num_layers": layers,
                "num_attention_heads": n_heads,
                "num_kv_heads": n_kv,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "context_length": 32,
                "rope_theta": 10000.0
            },
            "tensors": tensors
        });
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();

        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("qwen/qwen3-0.6b")
            .build()
            .unwrap();
        let gen = s
            .generate(&[1, 2], &GenerateOpts {
                max_tokens: 2,
                temperature: 0.0,
            })
            .unwrap();
        assert_eq!(gen.tokens.len(), 2);
    }

    #[test]
    fn materialize_accepts_language_model_prefix_and_pre_ffn_norm() {
        // Gemma-4 / Gemma-3n VL-style HF paths.
        let dir = tempfile::tempdir().unwrap();
        let hidden = 8usize;
        let layers = 1usize;
        let inter = 16usize;
        let vocab = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 4usize;
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv * head_dim;

        let mut tensors = serde_json::Map::new();
        let mut bin = Vec::new();
        let mut add_raw = |name: &str, shape: Vec<usize>, data: &[f32]| {
            let offset = bin.len();
            for &v in data {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let nbytes = data.len() * 4;
            let mut meta = serde_json::Map::new();
            meta.insert("kind".into(), json!("raw"));
            meta.insert("dtype".into(), json!("f32"));
            meta.insert("shape".into(), json!(shape));
            meta.insert("offsets".into(), json!({ "data": [offset, nbytes] }));
            tensors.insert(name.to_string(), Value::Object(meta));
        };
        let emb: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.01).collect();
        let p = "model.language_model";
        add_raw(&format!("{p}.embed_tokens.weight"), vec![vocab, hidden], &emb);
        let n1 = vec![1.0f32; hidden];
        add_raw(
            &format!("{p}.layers.0.input_layernorm.weight"),
            vec![hidden],
            &n1,
        );
        add_raw(
            &format!("{p}.layers.0.pre_feedforward_layernorm.weight"),
            vec![hidden],
            &n1,
        );
        let wq = vec![0.01f32; q_dim * hidden];
        let wk = vec![0.01f32; k_dim * hidden];
        let wv = vec![0.01f32; k_dim * hidden];
        let wo = vec![0.01f32; hidden * q_dim];
        add_raw(
            &format!("{p}.layers.0.self_attn.q_proj.weight"),
            vec![q_dim, hidden],
            &wq,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.k_proj.weight"),
            vec![k_dim, hidden],
            &wk,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.v_proj.weight"),
            vec![k_dim, hidden],
            &wv,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.o_proj.weight"),
            vec![hidden, q_dim],
            &wo,
        );
        let g = vec![0.01f32; inter * hidden];
        let d = vec![0.01f32; hidden * inter];
        add_raw(
            &format!("{p}.layers.0.mlp.gate_proj.weight"),
            vec![inter, hidden],
            &g,
        );
        add_raw(
            &format!("{p}.layers.0.mlp.up_proj.weight"),
            vec![inter, hidden],
            &g,
        );
        add_raw(
            &format!("{p}.layers.0.mlp.down_proj.weight"),
            vec![hidden, inter],
            &d,
        );
        add_raw(&format!("{p}.norm.weight"), vec![hidden], &n1);
        add_raw("lm_head.weight", vec![vocab, hidden], &emb);

        let cfg = json!({
            "format": "aria-quant-bundle",
            "format_version": 2,
            "quantization": "test",
            "group_size_default": 32,
            "hadamard_seed": 0,
            "model": {
                "hidden_size": hidden,
                "num_layers": layers,
                "num_attention_heads": n_heads,
                "num_kv_heads": n_kv,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "context_length": 32,
                "rope_theta": 10000.0
            },
            "tensors": tensors
        });
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();

        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("gemma/gemma-4-e2b-it")
            .build()
            .unwrap();
        let gen = s
            .generate(&[1, 2], &GenerateOpts {
                max_tokens: 2,
                temperature: 0.0,
            })
            .unwrap();
        assert_eq!(gen.tokens.len(), 2);
    }

    #[test]
    fn gemma4_style_double_wide_mlp_and_shared_kv() {
        // Config intermediate_size stays at the narrow width; layer 1 is 2× (KV-shared)
        // and omits k/v projections — reuses producer KV cache (num_kv_shared_layers=1).
        let dir = tempfile::tempdir().unwrap();
        let hidden = 8usize;
        let layers = 2usize;
        let inter = 16usize;
        let inter_wide = 32usize;
        let vocab = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 4usize;
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv * head_dim;
        let p = "model.language_model";

        let mut tensors = serde_json::Map::new();
        let mut bin = Vec::new();
        let mut add_raw = |name: &str, shape: Vec<usize>, data: &[f32]| {
            let offset = bin.len();
            for &v in data {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let nbytes = data.len() * 4;
            let mut meta = serde_json::Map::new();
            meta.insert("kind".into(), json!("raw"));
            meta.insert("dtype".into(), json!("f32"));
            meta.insert("shape".into(), json!(shape));
            meta.insert("offsets".into(), json!({ "data": [offset, nbytes] }));
            tensors.insert(name.to_string(), Value::Object(meta));
        };
        let emb: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.01).collect();
        add_raw(&format!("{p}.embed_tokens.weight"), vec![vocab, hidden], &emb);
        let n1 = vec![1.0f32; hidden];
        let wq = vec![0.01f32; q_dim * hidden];
        let wk = vec![0.01f32; k_dim * hidden];
        let wv = vec![0.01f32; k_dim * hidden];
        let wo = vec![0.01f32; hidden * q_dim];
        for li in 0..layers {
            let layer_inter = if li == 0 { inter } else { inter_wide };
            add_raw(
                &format!("{p}.layers.{li}.input_layernorm.weight"),
                vec![hidden],
                &n1,
            );
            add_raw(
                &format!("{p}.layers.{li}.pre_feedforward_layernorm.weight"),
                vec![hidden],
                &n1,
            );
            add_raw(
                &format!("{p}.layers.{li}.self_attn.q_proj.weight"),
                vec![q_dim, hidden],
                &wq,
            );
            if li == 0 {
                add_raw(
                    &format!("{p}.layers.{li}.self_attn.k_proj.weight"),
                    vec![k_dim, hidden],
                    &wk,
                );
                add_raw(
                    &format!("{p}.layers.{li}.self_attn.v_proj.weight"),
                    vec![k_dim, hidden],
                    &wv,
                );
            }
            add_raw(
                &format!("{p}.layers.{li}.self_attn.o_proj.weight"),
                vec![hidden, q_dim],
                &wo,
            );
            let g = vec![0.01f32; layer_inter * hidden];
            let d = vec![0.01f32; hidden * layer_inter];
            add_raw(
                &format!("{p}.layers.{li}.mlp.gate_proj.weight"),
                vec![layer_inter, hidden],
                &g,
            );
            add_raw(
                &format!("{p}.layers.{li}.mlp.up_proj.weight"),
                vec![layer_inter, hidden],
                &g,
            );
            add_raw(
                &format!("{p}.layers.{li}.mlp.down_proj.weight"),
                vec![hidden, layer_inter],
                &d,
            );
        }
        add_raw(&format!("{p}.norm.weight"), vec![hidden], &n1);
        add_raw("lm_head.weight", vec![vocab, hidden], &emb);

        let cfg = json!({
            "format": "aria-quant-bundle",
            "format_version": 2,
            "quantization": "test",
            "group_size_default": 32,
            "hadamard_seed": 0,
            "model": {
                "hidden_size": hidden,
                "num_layers": layers,
                "num_attention_heads": n_heads,
                "num_kv_heads": n_kv,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "context_length": 32,
                "rope_theta": 10000.0,
                "num_kv_shared_layers": 1,
                "layer_types": ["full_attention", "full_attention"]
            },
            "tensors": tensors
        });
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();

        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("gemma/gemma-4-e2b-it")
            .build()
            .unwrap();
        let gen = s
            .generate(
                &[1, 2],
                &GenerateOpts {
                    max_tokens: 2,
                    temperature: 0.0,
                },
            )
            .unwrap();
        assert_eq!(gen.tokens.len(), 2);
    }

    #[test]
    fn stage_b_arch_classes_generate() {
        for (path, arch) in arch_class_representatives() {
            if matches!(arch, ArchClass::VL | ArchClass::VLA | ArchClass::TextMoE) {
                continue; // stage C / MoE gated separately
            }
            // Hybrid linear_attention families refuse dense Session until DeltaNet lands.
            if path.contains("qwen3.5") || path.contains("bonsai") {
                let dir = tempfile::tempdir().unwrap();
                write_tiny_q4_bundle(dir.path()).unwrap();
                let err = SessionBuilder::new()
                    .model(dir.path())
                    .family(*path)
                    .build()
                    .unwrap_err();
                assert!(
                    matches!(err, EngineError::Unsupported(_)),
                    "{path}: {err:?}"
                );
                continue;
            }
            assert!(require_stage_b(path).is_ok(), "{path}");
            let dir = tempfile::tempdir().unwrap();
            write_tiny_q4_bundle(dir.path()).unwrap();
            let mut s = SessionBuilder::new()
                .model(dir.path())
                .family(*path)
                .build()
                .unwrap();
            assert_eq!(s.arch(), *arch);
            assert!(!s.graph_hook_name().is_empty());
            let gen = s
                .generate(
                    &s.encode_text("ok"),
                    &GenerateOpts {
                        max_tokens: 2,
                        temperature: 0.0,
                    },
                )
                .unwrap();
            assert!(!gen.tokens.is_empty(), "{path}");
        }
    }

    #[test]
    fn stage_c_vl_vla_hooks() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let s = SessionBuilder::new()
            .model(dir.path())
            .family("lfm/lfm2-vl-450m")
            .build()
            .unwrap();
        let rgb = vec![10u8; 3 * 4 * 4];
        let err = s.vision_prefix(&rgb, 4, 4).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)));

        let vla = SessionBuilder::new()
            .model(dir.path())
            .family("openvla/openvla-7b")
            .build()
            .unwrap();
        let err = vla.predict_action("move", 7).unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)));
        let emb = vla.embed_text("hello").unwrap();
        assert_eq!(emb.len(), vla.config().hidden_size);
    }

    #[test]
    fn unknown_family() {
        let err = SessionBuilder::new()
            .model("/tmp")
            .family("no/such-model")
            .build()
            .unwrap_err();
        assert!(matches!(err, EngineError::UnsupportedFamily(_)));
    }

    #[test]
    fn greedy_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("gemma/gemma-4-e2b-it")
            .build()
            .unwrap();
        let prompt = s.encode_text("hi");
        let opts = GenerateOpts {
            max_tokens: 3,
            temperature: 0.0,
        };
        let a = s.generate(&prompt, &opts).unwrap();
        let b = s.generate(&prompt, &opts).unwrap();
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.tokens.len(), 3);
    }

    #[test]
    fn encode_chat_is_longer_than_raw_user_text() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let s = SessionBuilder::new()
            .model(dir.path())
            .family("qwen/qwen3-0.6b")
            .build()
            .unwrap();
        let raw = s.encode_text("Hello");
        let chat = s.encode_chat(&[ChatTurn::new("user", "Hello")]);
        assert!(
            chat.len() > raw.len(),
            "chat template should wrap the user turn (raw={}, chat={})",
            raw.len(),
            chat.len()
        );
        assert!(
            (s.config().rope_theta - 1_000_000.0).abs() < 1.0,
            "Qwen3 must not keep Llama-default rope_theta=10000, got {}",
            s.config().rope_theta
        );
    }

    #[test]
    fn incremental_decode_matches_full_recompute() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("gemma/gemma-4-e2b-it")
            .build()
            .unwrap();
        let prompt = s.encode_text("hi");
        let max_tokens = 5usize;

        // Legacy path: re-run full forward over the growing prefix each step.
        let mut prefix = prompt.clone();
        if prefix.is_empty() {
            prefix.push(1);
        }
        let mut full_tokens = Vec::new();
        for _ in 0..max_tokens {
            let logits = s.forward(&prefix).unwrap();
            let next = argmax(&logits);
            full_tokens.push(next);
            prefix.push(next);
            if s.is_stop_id(next) {
                full_tokens.pop();
                break;
            }
        }

        let incr = s
            .generate(
                &prompt,
                &GenerateOpts {
                    max_tokens,
                    temperature: 0.0,
                },
            )
            .unwrap();
        assert_eq!(
            incr.tokens, full_tokens,
            "incremental decode must match full-recompute greedy tokens"
        );
    }

    #[test]
    fn max_tokens_zero_rejected() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("gemma/gemma-4-e2b-it")
            .build()
            .unwrap();
        let err = s
            .generate(
                &s.encode_text("x"),
                &GenerateOpts {
                    max_tokens: 0,
                    temperature: 0.0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::InvalidParam(_)));
    }

    #[test]
    fn moe_family_refuses_dense_stub() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let err = SessionBuilder::new()
            .model(dir.path())
            .family("lfm/lfm2-8b-a1b")
            .build()
            .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)));
        assert_eq!(
            lookup_family("lfm/lfm2-8b-a1b").unwrap().arch,
            ArchClass::TextMoE
        );
        assert_eq!(graph_hook(ArchClass::TextMoE), "text_moe_decoder");
    }

    #[test]
    fn geometry_gates_conv_and_experts() {
        // layer_types=conv without conv.* weights → Format (not silent dense attn).
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let cfg_path = dir.path().join("config.json");
        let mut cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        cfg["model"]["layer_types"] = json!(["conv", "full_attention"]);
        std::fs::write(&cfg_path, cfg.to_string()).unwrap();
        let err = SessionBuilder::new()
            .model(dir.path())
            .family("lfm/lfm2-350m")
            .build()
            .unwrap_err();
        assert!(
            matches!(err, EngineError::Format(_)),
            "expected missing conv tensors, got {err:?}"
        );

        // linear_attention still hard-gated.
        let dir2 = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir2.path()).unwrap();
        let cfg_path = dir2.path().join("config.json");
        let mut cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        cfg["model"]["layer_types"] = json!(["linear_attention", "full_attention"]);
        std::fs::write(&cfg_path, cfg.to_string()).unwrap();
        let err = SessionBuilder::new()
            .model(dir2.path())
            .family("gemma/gemma-3-270m-it")
            .build()
            .unwrap_err();
        assert!(
            matches!(err, EngineError::Format(_)),
            "expected missing DeltaNet tensors, got {err:?}"
        );
    }

    #[test]
    fn lfm_short_conv_and_attn_generate() {
        let dir = tempfile::tempdir().unwrap();
        let hidden = 8usize;
        let layers = 2usize;
        let inter = 16usize;
        let vocab = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 4usize;
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv * head_dim;
        let kernel = 3usize;

        let mut tensors = serde_json::Map::new();
        let mut bin = Vec::new();
        let mut add_raw = |name: &str, shape: Vec<usize>, data: &[f32]| {
            let offset = bin.len();
            for &v in data {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let nbytes = data.len() * 4;
            let mut meta = serde_json::Map::new();
            meta.insert("kind".into(), json!("raw"));
            meta.insert("dtype".into(), json!("f32"));
            meta.insert("shape".into(), json!(shape));
            meta.insert("offsets".into(), json!({ "data": [offset, nbytes] }));
            tensors.insert(name.to_string(), Value::Object(meta));
        };
        let emb: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.01).collect();
        add_raw("model.embed_tokens.weight", vec![vocab, hidden], &emb);
        let n1 = vec![1.0f32; hidden];
        // Layer 0: short-conv
        add_raw("model.layers.0.operator_norm.weight", vec![hidden], &n1);
        add_raw("model.layers.0.ffn_norm.weight", vec![hidden], &n1);
        let in_proj = vec![0.02f32; 3 * hidden * hidden];
        let out_proj = vec![0.02f32; hidden * hidden];
        let conv_w = vec![0.1f32; hidden * kernel];
        add_raw(
            "model.layers.0.conv.in_proj.weight",
            vec![3 * hidden, hidden],
            &in_proj,
        );
        add_raw(
            "model.layers.0.conv.out_proj.weight",
            vec![hidden, hidden],
            &out_proj,
        );
        add_raw(
            "model.layers.0.conv.conv.weight",
            vec![hidden, kernel],
            &conv_w,
        );
        let g = vec![0.02f32; inter * hidden];
        let d = vec![0.02f32; hidden * inter];
        add_raw(
            "model.layers.0.mlp.gate_proj.weight",
            vec![inter, hidden],
            &g,
        );
        add_raw("model.layers.0.mlp.up_proj.weight", vec![inter, hidden], &g);
        add_raw(
            "model.layers.0.mlp.down_proj.weight",
            vec![hidden, inter],
            &d,
        );
        // Layer 1: full attention
        add_raw("model.layers.1.operator_norm.weight", vec![hidden], &n1);
        add_raw(
            "model.layers.1.post_attention_layernorm.weight",
            vec![hidden],
            &n1,
        );
        let wq = vec![0.02f32; q_dim * hidden];
        let wk = vec![0.02f32; k_dim * hidden];
        let wv = vec![0.02f32; k_dim * hidden];
        let wo = vec![0.02f32; hidden * q_dim];
        add_raw(
            "model.layers.1.self_attn.q_proj.weight",
            vec![q_dim, hidden],
            &wq,
        );
        add_raw(
            "model.layers.1.self_attn.k_proj.weight",
            vec![k_dim, hidden],
            &wk,
        );
        add_raw(
            "model.layers.1.self_attn.v_proj.weight",
            vec![k_dim, hidden],
            &wv,
        );
        add_raw(
            "model.layers.1.self_attn.o_proj.weight",
            vec![hidden, q_dim],
            &wo,
        );
        add_raw(
            "model.layers.1.mlp.gate_proj.weight",
            vec![inter, hidden],
            &g,
        );
        add_raw("model.layers.1.mlp.up_proj.weight", vec![inter, hidden], &g);
        add_raw(
            "model.layers.1.mlp.down_proj.weight",
            vec![hidden, inter],
            &d,
        );
        add_raw("model.norm.weight", vec![hidden], &n1);
        add_raw("lm_head.weight", vec![vocab, hidden], &emb);

        let cfg = json!({
            "format": "aria-quant-bundle",
            "format_version": 2,
            "quantization": "test",
            "group_size_default": 32,
            "hadamard_seed": 0,
            "model": {
                "hidden_size": hidden,
                "num_layers": layers,
                "num_attention_heads": n_heads,
                "num_kv_heads": n_kv,
                "head_dim": head_dim,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "context_length": 32,
                "rope_theta": 10000.0,
                "conv_l_cache": kernel,
                "layer_types": ["conv", "full_attention"]
            },
            "tensors": tensors
        });
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();

        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("lfm/lfm2-350m")
            .build()
            .unwrap();
        let gen = s
            .generate(
                &[1, 2, 3],
                &GenerateOpts {
                    max_tokens: 2,
                    temperature: 0.0,
                },
            )
            .unwrap();
        assert_eq!(gen.tokens.len(), 2);
    }

    #[test]
    fn moe_topk_experts_generate() {
        let dir = tempfile::tempdir().unwrap();
        let hidden = 8usize;
        let layers = 1usize;
        let inter = 16usize;
        let vocab = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 4usize;
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv * head_dim;
        let n_experts = 4usize;
        let top_k = 2usize;

        let mut tensors = serde_json::Map::new();
        let mut bin = Vec::new();
        let mut add_raw = |name: &str, shape: Vec<usize>, data: &[f32]| {
            let offset = bin.len();
            for &v in data {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let nbytes = data.len() * 4;
            let mut meta = serde_json::Map::new();
            meta.insert("kind".into(), json!("raw"));
            meta.insert("dtype".into(), json!("f32"));
            meta.insert("shape".into(), json!(shape));
            meta.insert("offsets".into(), json!({ "data": [offset, nbytes] }));
            tensors.insert(name.to_string(), Value::Object(meta));
        };
        let emb: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.01).collect();
        add_raw("model.embed_tokens.weight", vec![vocab, hidden], &emb);
        let n1 = vec![1.0f32; hidden];
        add_raw("model.layers.0.input_layernorm.weight", vec![hidden], &n1);
        add_raw(
            "model.layers.0.post_attention_layernorm.weight",
            vec![hidden],
            &n1,
        );
        let wq = vec![0.02f32; q_dim * hidden];
        let wk = vec![0.02f32; k_dim * hidden];
        let wv = vec![0.02f32; k_dim * hidden];
        let wo = vec![0.02f32; hidden * q_dim];
        add_raw(
            "model.layers.0.self_attn.q_proj.weight",
            vec![q_dim, hidden],
            &wq,
        );
        add_raw(
            "model.layers.0.self_attn.k_proj.weight",
            vec![k_dim, hidden],
            &wk,
        );
        add_raw(
            "model.layers.0.self_attn.v_proj.weight",
            vec![k_dim, hidden],
            &wv,
        );
        add_raw(
            "model.layers.0.self_attn.o_proj.weight",
            vec![hidden, q_dim],
            &wo,
        );
        let router: Vec<f32> = (0..n_experts * hidden)
            .map(|i| ((i % n_experts) as f32) * 0.1)
            .collect();
        add_raw(
            "model.layers.0.block_sparse_moe.gate.weight",
            vec![n_experts, hidden],
            &router,
        );
        let g = vec![0.02f32; inter * hidden];
        let d = vec![0.02f32; hidden * inter];
        for e in 0..n_experts {
            add_raw(
                &format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight"),
                vec![inter, hidden],
                &g,
            );
            add_raw(
                &format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight"),
                vec![inter, hidden],
                &g,
            );
            add_raw(
                &format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight"),
                vec![hidden, inter],
                &d,
            );
        }
        add_raw("model.norm.weight", vec![hidden], &n1);
        add_raw("lm_head.weight", vec![vocab, hidden], &emb);

        let cfg = json!({
            "format": "aria-quant-bundle",
            "format_version": 2,
            "quantization": "test",
            "group_size_default": 32,
            "hadamard_seed": 0,
            "model": {
                "hidden_size": hidden,
                "num_layers": layers,
                "num_attention_heads": n_heads,
                "num_kv_heads": n_kv,
                "head_dim": head_dim,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "context_length": 32,
                "rope_theta": 10000.0,
                "num_experts": n_experts,
                "num_experts_per_tok": top_k
            },
            "tensors": tensors
        });
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();

        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("inkling/inkling-small")
            .build()
            .unwrap();
        assert_eq!(s.arch(), ArchClass::TextMoE);
        assert_eq!(s.graph_hook_name(), "text_moe_decoder");
        let gen = s
            .generate(
                &[1, 2],
                &GenerateOpts {
                    max_tokens: 2,
                    temperature: 0.0,
                },
            )
            .unwrap();
        assert_eq!(gen.tokens.len(), 2);
    }

    #[test]
    fn tiny_q4_codebook_weights_carry_hdm_seed() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let b = load_bundle(dir.path()).unwrap();
        let w = b.weight_loaded("blk.0.attn_q.weight").unwrap();
        assert!(w.hdm_seed.is_some(), "rotated codebook must expose HDM seed");
        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("gemma/gemma-4-e2b-it")
            .build()
            .unwrap();
        let gen = s
            .generate(
                &[1, 2],
                &GenerateOpts {
                    max_tokens: 2,
                    temperature: 0.0,
                },
            )
            .unwrap();
        assert_eq!(gen.tokens.len(), 2);
    }

    #[test]
    fn gemma_hidden_act_geglu_and_qk_norm() {
        let dir = tempfile::tempdir().unwrap();
        let hidden = 8usize;
        let layers = 1usize;
        let inter = 16usize;
        let vocab = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 4usize;
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv * head_dim;
        let p = "model.language_model";

        let mut tensors = serde_json::Map::new();
        let mut bin = Vec::new();
        let mut add_raw = |name: &str, shape: Vec<usize>, data: &[f32]| {
            let offset = bin.len();
            for &v in data {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let nbytes = data.len() * 4;
            let mut meta = serde_json::Map::new();
            meta.insert("kind".into(), json!("raw"));
            meta.insert("dtype".into(), json!("f32"));
            meta.insert("shape".into(), json!(shape));
            meta.insert("offsets".into(), json!({ "data": [offset, nbytes] }));
            tensors.insert(name.to_string(), Value::Object(meta));
        };
        let emb: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.01).collect();
        add_raw(&format!("{p}.embed_tokens.weight"), vec![vocab, hidden], &emb);
        let n1 = vec![1.0f32; hidden];
        let qn = vec![1.0f32; q_dim];
        let kn = vec![1.0f32; k_dim];
        let wq = vec![0.01f32; q_dim * hidden];
        let wk = vec![0.01f32; k_dim * hidden];
        let wv = vec![0.01f32; k_dim * hidden];
        let wo = vec![0.01f32; hidden * q_dim];
        add_raw(
            &format!("{p}.layers.0.input_layernorm.weight"),
            vec![hidden],
            &n1,
        );
        add_raw(
            &format!("{p}.layers.0.pre_feedforward_layernorm.weight"),
            vec![hidden],
            &n1,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.q_proj.weight"),
            vec![q_dim, hidden],
            &wq,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.k_proj.weight"),
            vec![k_dim, hidden],
            &wk,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.v_proj.weight"),
            vec![k_dim, hidden],
            &wv,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.o_proj.weight"),
            vec![hidden, q_dim],
            &wo,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.q_norm.weight"),
            vec![q_dim],
            &qn,
        );
        add_raw(
            &format!("{p}.layers.0.self_attn.k_norm.weight"),
            vec![k_dim],
            &kn,
        );
        let g = vec![0.01f32; inter * hidden];
        let d = vec![0.01f32; hidden * inter];
        add_raw(
            &format!("{p}.layers.0.mlp.gate_proj.weight"),
            vec![inter, hidden],
            &g,
        );
        add_raw(
            &format!("{p}.layers.0.mlp.up_proj.weight"),
            vec![inter, hidden],
            &g,
        );
        add_raw(
            &format!("{p}.layers.0.mlp.down_proj.weight"),
            vec![hidden, inter],
            &d,
        );
        add_raw(&format!("{p}.norm.weight"), vec![hidden], &n1);
        add_raw("lm_head.weight", vec![vocab, hidden], &emb);

        let cfg = json!({
            "format": "aria-quant-bundle",
            "format_version": 2,
            "quantization": "test",
            "group_size_default": 32,
            "hadamard_seed": 0,
            "model": {
                "hidden_size": hidden,
                "num_layers": layers,
                "num_attention_heads": n_heads,
                "num_kv_heads": n_kv,
                "head_dim": head_dim,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "context_length": 32,
                "rope_theta": 10000.0,
                "hidden_act": "gelu_pytorch_tanh"
            },
            "tensors": tensors
        });
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();

        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("gemma/gemma-4-e2b-it")
            .build()
            .unwrap();
        assert_eq!(
            s.config().hidden_act.as_deref(),
            Some("gelu_pytorch_tanh")
        );
        let gen = s
            .generate(
                &[1, 2],
                &GenerateOpts {
                    max_tokens: 2,
                    temperature: 0.0,
                },
            )
            .unwrap();
        assert_eq!(gen.tokens.len(), 2);
    }

    #[test]
    fn gated_deltanet_and_full_attn_generate() {
        let dir = tempfile::tempdir().unwrap();
        let hidden = 8usize;
        let inter = 16usize;
        let vocab = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 4usize;
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv * head_dim;
        let n_lin = 2usize;
        let hk = 4usize;
        let hv = 4usize;
        let key_dim = n_lin * hk;
        let value_dim = n_lin * hv;
        let conv_k = 4usize;
        let conv_dim = key_dim * 2 + value_dim;

        let mut tensors = serde_json::Map::new();
        let mut bin = Vec::new();
        let mut add_raw = |name: &str, shape: Vec<usize>, data: &[f32]| {
            let offset = bin.len();
            for &v in data {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let nbytes = data.len() * 4;
            let mut meta = serde_json::Map::new();
            meta.insert("kind".into(), json!("raw"));
            meta.insert("dtype".into(), json!("f32"));
            meta.insert("shape".into(), json!(shape));
            meta.insert("offsets".into(), json!({ "data": [offset, nbytes] }));
            tensors.insert(name.to_string(), Value::Object(meta));
        };
        let emb: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.01).collect();
        add_raw("model.embed_tokens.weight", vec![vocab, hidden], &emb);
        let n1 = vec![1.0f32; hidden];
        add_raw("model.layers.0.input_layernorm.weight", vec![hidden], &n1);
        add_raw(
            "model.layers.0.post_attention_layernorm.weight",
            vec![hidden],
            &n1,
        );
        let qkvz = vec![0.02f32; (2 * key_dim + 2 * value_dim) * hidden];
        let ba = vec![0.1f32; 2 * n_lin * hidden];
        let conv = vec![0.05f32; conv_dim * conv_k];
        let a_log = vec![0.5f32; n_lin];
        let dt = vec![1.0f32; n_lin];
        let outp = vec![0.02f32; hidden * value_dim];
        add_raw(
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            vec![2 * key_dim + 2 * value_dim, hidden],
            &qkvz,
        );
        add_raw(
            "model.layers.0.linear_attn.in_proj_ba.weight",
            vec![2 * n_lin, hidden],
            &ba,
        );
        add_raw(
            "model.layers.0.linear_attn.conv1d.weight",
            vec![conv_dim, conv_k],
            &conv,
        );
        add_raw("model.layers.0.linear_attn.A_log", vec![n_lin], &a_log);
        add_raw("model.layers.0.linear_attn.dt_bias", vec![n_lin], &dt);
        add_raw(
            "model.layers.0.linear_attn.out_proj.weight",
            vec![hidden, value_dim],
            &outp,
        );
        let g = vec![0.02f32; inter * hidden];
        let d = vec![0.02f32; hidden * inter];
        add_raw("model.layers.0.mlp.gate_proj.weight", vec![inter, hidden], &g);
        add_raw("model.layers.0.mlp.up_proj.weight", vec![inter, hidden], &g);
        add_raw("model.layers.0.mlp.down_proj.weight", vec![hidden, inter], &d);

        add_raw("model.layers.1.input_layernorm.weight", vec![hidden], &n1);
        add_raw(
            "model.layers.1.post_attention_layernorm.weight",
            vec![hidden],
            &n1,
        );
        let wq = vec![0.02f32; q_dim * hidden];
        let wk = vec![0.02f32; k_dim * hidden];
        let wv = vec![0.02f32; k_dim * hidden];
        let wo = vec![0.02f32; hidden * q_dim];
        add_raw(
            "model.layers.1.self_attn.q_proj.weight",
            vec![q_dim, hidden],
            &wq,
        );
        add_raw(
            "model.layers.1.self_attn.k_proj.weight",
            vec![k_dim, hidden],
            &wk,
        );
        add_raw(
            "model.layers.1.self_attn.v_proj.weight",
            vec![k_dim, hidden],
            &wv,
        );
        add_raw(
            "model.layers.1.self_attn.o_proj.weight",
            vec![hidden, q_dim],
            &wo,
        );
        add_raw("model.layers.1.mlp.gate_proj.weight", vec![inter, hidden], &g);
        add_raw("model.layers.1.mlp.up_proj.weight", vec![inter, hidden], &g);
        add_raw("model.layers.1.mlp.down_proj.weight", vec![hidden, inter], &d);
        add_raw("model.norm.weight", vec![hidden], &n1);
        add_raw("lm_head.weight", vec![vocab, hidden], &emb);

        let cfg = json!({
            "format": "aria-quant-bundle",
            "format_version": 2,
            "quantization": "test",
            "hadamard_seed": 0,
            "model": {
                "hidden_size": hidden,
                "num_layers": 2,
                "num_attention_heads": n_heads,
                "num_kv_heads": n_kv,
                "head_dim": head_dim,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "context_length": 32,
                "rope_theta": 10000.0,
                "layer_types": ["linear_attention", "full_attention"]
            },
            "tensors": tensors
        });
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();
        let mut s = SessionBuilder::new()
            .model(dir.path())
            .family("qwen/qwen3.5-2b")
            .build()
            .unwrap();
        let gen = s
            .generate(
                &[1, 2, 3],
                &GenerateOpts {
                    max_tokens: 2,
                    temperature: 0.0,
                },
            )
            .unwrap();
        assert_eq!(gen.tokens.len(), 2);
    }

    #[test]
    fn vision_and_action_consume_bundle_weights() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let cfg_path = dir.path().join("config.json");
        let mut cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        let hidden = cfg["model"]["hidden_size"].as_u64().unwrap() as usize;
        let mut tensors = cfg["tensors"].as_object().cloned().unwrap();
        let mut bin = std::fs::read(dir.path().join("weight.bin")).unwrap();
        let mut add_raw = |name: &str, shape: Vec<usize>, data: &[f32]| {
            let offset = bin.len();
            for &v in data {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            let nbytes = data.len() * 4;
            tensors.insert(
                name.to_string(),
                json!({
                    "kind": "raw",
                    "dtype": "f32",
                    "shape": shape,
                    "offsets": { "data": [offset, nbytes] }
                }),
            );
        };
        let vis = vec![0.1f32; hidden * 3];
        add_raw("mm_projector.weight", vec![hidden, 3], &vis);
        let act_dim = 7usize;
        let act = vec![0.05f32; act_dim * hidden];
        add_raw("action_head.weight", vec![act_dim, hidden], &act);
        cfg["tensors"] = Value::Object(tensors);
        std::fs::write(&cfg_path, cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();

        let s = SessionBuilder::new()
            .model(dir.path())
            .family("lfm/lfm2-vl-450m")
            .build()
            .unwrap();
        let rgb = vec![10u8; 3 * 4 * 4];
        let pref = s.vision_prefix(&rgb, 4, 4).unwrap();
        assert_eq!(pref.len(), hidden);

        let vla = SessionBuilder::new()
            .model(dir.path())
            .family("openvla/openvla-7b")
            .build()
            .unwrap();
        let a = vla.predict_action("move", act_dim).unwrap();
        assert_eq!(a.len(), act_dim);
    }

    #[test]
    fn load_real_hf_named_bundle_if_present() {
        // Optional local smoke: ARIA_SMOKE_BUNDLE=/path/to/qwen3-0.6b_q4
        let Ok(path) = std::env::var("ARIA_SMOKE_BUNDLE") else {
            return;
        };
        let path = std::path::Path::new(&path);
        if !path.join("config.json").is_file() {
            return;
        }
        let s = SessionBuilder::new()
            .model(path)
            .family("qwen/qwen3-0.6b")
            .build()
            .expect("HF-named qwen bundle should materialize");
        assert!(s.config().num_layers > 0);
        assert!(s.config().hidden_size > 0);
    }
}
