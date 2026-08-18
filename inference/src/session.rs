use crate::bundle::{load_bundle, Bundle};
use crate::family::{graph_hook, require_runnable, ArchClass, Family};
use crate::multimodal::{action_head, asr_transcribe_pcm16le, vision_encode};
use crate::tensor_names::{
    attn_k_names, attn_norm_names, attn_o_names, attn_q_names, attn_v_names, emb_names,
    ffn_down_names, ffn_gate_names, ffn_norm_names, ffn_up_names, output_names, output_norm_names,
};
use aria_kernel::{attention, linear, rms_norm, rope, swiglu, EngineError};
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

struct LayerWeights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

struct ModelWeights {
    emb: Vec<f32>,
    layers: Vec<LayerWeights>,
    output_norm: Vec<f32>,
    output: Vec<f32>,
}

pub struct Session {
    family: Family,
    bundle: Bundle,
    weights: ModelWeights,
    conf: crate::bundle::ModelConfig,
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
        let weights = materialize(&bundle)?;
        let conf = bundle.model.clone();
        Ok(Session {
            family,
            bundle,
            weights,
            conf,
        })
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn materialize(b: &Bundle) -> Result<ModelWeights, EngineError> {
    fn any_owned(b: &Bundle, names: &[String]) -> Result<Vec<f32>, EngineError> {
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        b.weight_f32_any(&refs)
    }

    let m = &b.model;
    let mut layers = Vec::with_capacity(m.num_layers);
    // Gemma-4 E2B/E4B: last `num_kv_shared_layers` omit k/v; reuse prior layer KV.
    let mut prev_wk: Option<Vec<f32>> = None;
    let mut prev_wv: Option<Vec<f32>> = None;
    for layer in 0..m.num_layers {
        let wk = match any_owned(b, &attn_k_names(layer)) {
            Ok(w) => {
                prev_wk = Some(w.clone());
                w
            }
            Err(e) => prev_wk.clone().ok_or_else(|| {
                EngineError::Format(format!(
                    "missing k_proj for layer {layer} and no prior KV to share ({e})"
                ))
            })?,
        };
        let wv = match any_owned(b, &attn_v_names(layer)) {
            Ok(w) => {
                prev_wv = Some(w.clone());
                w
            }
            Err(e) => prev_wv.clone().ok_or_else(|| {
                EngineError::Format(format!(
                    "missing v_proj for layer {layer} and no prior KV to share ({e})"
                ))
            })?,
        };
        layers.push(LayerWeights {
            attn_norm: any_owned(b, &attn_norm_names(layer))?,
            ffn_norm: any_owned(b, &ffn_norm_names(layer))?,
            wq: any_owned(b, &attn_q_names(layer))?,
            wk,
            wv,
            wo: any_owned(b, &attn_o_names(layer))?,
            gate: any_owned(b, &ffn_gate_names(layer))?,
            up: any_owned(b, &ffn_up_names(layer))?,
            down: any_owned(b, &ffn_down_names(layer))?,
        });
    }
    let emb_n = emb_names();
    let out_norm_n = output_norm_names();
    let out_n = output_names();
    Ok(ModelWeights {
        emb: b.weight_f32_any(&emb_n)?,
        layers,
        output_norm: b.weight_f32_any(&out_norm_n)?,
        output: b.weight_f32_any(&out_n)?,
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
    pub fn generate(&mut self, prompt: &[u32], opts: &GenerateOpts) -> Result<Generation, EngineError> {
        if opts.max_tokens == 0 {
            return Err(EngineError::InvalidParam("max_tokens must be > 0".into()));
        }
        let mut tokens: Vec<u32> = prompt.to_vec();
        if tokens.is_empty() {
            tokens.push(1);
        }
        let mut generated = Vec::new();
        for _ in 0..opts.max_tokens {
            let logits = self.forward(&tokens)?;
            let next = if opts.temperature <= 0.0 {
                argmax(&logits)
            } else {
                // Stage A: still greedy for determinism; temperature reserved.
                argmax(&logits)
            };
            generated.push(next);
            tokens.push(next);
            if next == 0 {
                break;
            }
        }
        let text = generated
            .iter()
            .map(|t| format!("<{t}>"))
            .collect::<Vec<_>>()
            .join("");
        Ok(Generation {
            tokens: generated,
            text,
        })
    }

    /// Naive whitespace / char tokenizer for demos (no HF tokenizer required).
    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        let vocab = self.conf.vocab_size as u32;
        if text.is_empty() {
            return vec![1];
        }
        text.bytes()
            .map(|b| (b as u32) % vocab.max(1))
            .collect()
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
            let row = &self.weights.emb[tid * hidden..(tid + 1) * hidden];
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

    /// Stage C VL: encode image and blend into a one-token visual prefix embedding.
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
        vision_encode(rgb, height, width, self.conf.hidden_size)
    }

    /// Stage C VLA action vector from last hidden proxy (embedding of prompt).
    pub fn predict_action(&self, prompt: &str, action_dim: usize) -> Result<Vec<f32>, EngineError> {
        if self.family.arch != ArchClass::VLA {
            return Err(EngineError::Unsupported(format!(
                "predict_action requires VLA, got {:?}",
                self.family.arch
            )));
        }
        let h = self.embed_text(prompt)?;
        action_head(&h, action_dim)
    }

    /// Stage C ASR stub bound to session vocab.
    pub fn transcribe_pcm16le(&self, pcm: &[u8]) -> Result<String, EngineError> {
        asr_transcribe_pcm16le(pcm, self.conf.vocab_size as u32)
    }

    fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        let hidden = self.conf.hidden_size;
        let n_heads = self.conf.num_attention_heads;
        let n_kv = self.conf.num_kv_heads;
        let vocab = self.conf.vocab_size;
        if hidden == 0 {
            return Err(EngineError::ShapeMismatch("hidden_size is 0".into()));
        }
        if self.weights.emb.len() < vocab.saturating_mul(hidden)
            || !self.weights.emb.len().is_multiple_of(hidden)
        {
            return Err(EngineError::ShapeMismatch(format!(
                "embedding length {} not compatible with vocab={vocab} hidden={hidden}",
                self.weights.emb.len()
            )));
        }
        let mut k_caches: Vec<Vec<f32>> = (0..self.conf.num_layers).map(|_| Vec::new()).collect();
        let mut v_caches: Vec<Vec<f32>> = (0..self.conf.num_layers).map(|_| Vec::new()).collect();

        let mut x = vec![0.0f32; hidden];
        for (pos, &tok) in tokens.iter().enumerate() {
            let tid = (tok as usize) % vocab;
            x.copy_from_slice(&self.weights.emb[tid * hidden..(tid + 1) * hidden]);

            for (li, layer) in self.weights.layers.iter().enumerate() {
                if layer.wq.len() % hidden != 0
                    || layer.wk.len() % hidden != 0
                    || layer.wv.len() % hidden != 0
                {
                    return Err(EngineError::ShapeMismatch(
                        "attn proj weight not divisible by hidden_size".into(),
                    ));
                }
                let q_dim = layer.wq.len() / hidden;
                let k_dim = layer.wk.len() / hidden;
                let v_dim = layer.wv.len() / hidden;
                if n_heads == 0 || !q_dim.is_multiple_of(n_heads) {
                    return Err(EngineError::ShapeMismatch(
                        "q_dim not divisible by num_attention_heads".into(),
                    ));
                }
                // Gemma-4: head_dim may differ from hidden/n_heads (e.g. 256 vs 192).
                let head_dim = q_dim / n_heads;
                if k_dim != n_kv * head_dim || v_dim != n_kv * head_dim {
                    return Err(EngineError::ShapeMismatch(format!(
                        "kv dims {k_dim}/{v_dim} != n_kv*head_dim {}",
                        n_kv * head_dim
                    )));
                }
                if layer.wo.len() != hidden * q_dim {
                    return Err(EngineError::ShapeMismatch(
                        "attn output proj weight shape mismatch".into(),
                    ));
                }

                let xn = rms_norm(&x, &layer.attn_norm, 1e-6)?;
                let mut q = linear(&xn, &layer.wq, q_dim, hidden)?;
                let mut k = linear(&xn, &layer.wk, k_dim, hidden)?;
                let v = linear(&xn, &layer.wv, v_dim, hidden)?;
                rope(&mut q, head_dim, pos, self.conf.rope_theta)?;
                rope(&mut k, head_dim, pos, self.conf.rope_theta)?;
                k_caches[li].extend_from_slice(&k);
                v_caches[li].extend_from_slice(&v);
                let attn = attention(&q, &k_caches[li], &v_caches[li], n_heads, n_kv, head_dim)?;
                let ao = linear(&attn, &layer.wo, hidden, q_dim)?;
                for i in 0..hidden {
                    x[i] += ao[i];
                }
                let xn2 = rms_norm(&x, &layer.ffn_norm, 1e-6)?;
                // Per-layer intermediate (Gemma-4 KV-shared layers use 2× MLP width).
                if layer.gate.len() % hidden != 0 {
                    return Err(EngineError::ShapeMismatch(format!(
                        "layer {li} gate len {} not divisible by hidden {hidden}",
                        layer.gate.len()
                    )));
                }
                let inter = layer.gate.len() / hidden;
                if inter == 0 {
                    return Err(EngineError::ShapeMismatch(format!(
                        "layer {li} inferred intermediate_size is 0"
                    )));
                }
                if layer.up.len() != inter * hidden {
                    return Err(EngineError::ShapeMismatch(format!(
                        "layer {li} up len {} != inter*hidden {inter}*{hidden}",
                        layer.up.len()
                    )));
                }
                if layer.down.len() != hidden * inter {
                    return Err(EngineError::ShapeMismatch(format!(
                        "layer {li} down len {} != hidden*inter {hidden}*{inter}",
                        layer.down.len()
                    )));
                }
                let gate = linear(&xn2, &layer.gate, inter, hidden)?;
                let up = linear(&xn2, &layer.up, inter, hidden)?;
                let h = swiglu(&gate, &up)?;
                let down = linear(&h, &layer.down, hidden, inter)?;
                for i in 0..hidden {
                    x[i] += down[i];
                }
            }
        }
        let xn = rms_norm(&x, &self.weights.output_norm, 1e-6)?;
        if self.weights.output.len() % hidden != 0 {
            return Err(EngineError::ShapeMismatch(format!(
                "lm_head len {} not divisible by hidden {hidden}",
                self.weights.output.len()
            )));
        }
        let out_rows = self.weights.output.len() / hidden;
        linear(&xn, &self.weights.output, out_rows, hidden)
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
    use crate::family::{arch_class_representatives, require_stage_b};
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
        // and omits k/v projections — must still generate without shape mismatch.
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
            if matches!(arch, ArchClass::VL | ArchClass::VLA) {
                continue; // stage C reps tested separately
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
        let pref = s.vision_prefix(&rgb, 4, 4).unwrap();
        assert_eq!(pref.len(), s.config().hidden_size);

        let vla = SessionBuilder::new()
            .model(dir.path())
            .family("openvla/openvla-7b")
            .build()
            .unwrap();
        let act = vla.predict_action("move", 7).unwrap();
        assert_eq!(act.len(), 7);
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
    fn moe_family_hook() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let s = SessionBuilder::new()
            .model(dir.path())
            .family("lfm/lfm2-8b-a1b")
            .build()
            .unwrap();
        assert_eq!(s.arch(), ArchClass::TextMoE);
        assert!(s.family().is_moe());
        assert_eq!(s.graph_hook_name(), "text_moe_decoder_stub");
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
