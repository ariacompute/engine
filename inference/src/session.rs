use crate::bundle::{load_bundle, Bundle};
use crate::family::{graph_hook, require_runnable, ArchClass, Family};
use crate::multimodal::{action_head, asr_transcribe_pcm16le, vision_encode};
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
    let m = &b.model;
    let mut layers = Vec::with_capacity(m.num_layers);
    for layer in 0..m.num_layers {
        layers.push(LayerWeights {
            attn_norm: b.weight_f32(&format!("blk.{layer}.attn_norm.weight"))?,
            ffn_norm: b.weight_f32(&format!("blk.{layer}.ffn_norm.weight"))?,
            wq: b.weight_f32(&format!("blk.{layer}.attn_q.weight"))?,
            wk: b.weight_f32(&format!("blk.{layer}.attn_k.weight"))?,
            wv: b.weight_f32(&format!("blk.{layer}.attn_v.weight"))?,
            wo: b.weight_f32(&format!("blk.{layer}.attn_output.weight"))?,
            gate: b.weight_f32(&format!("blk.{layer}.ffn_gate.weight"))?,
            up: b.weight_f32(&format!("blk.{layer}.ffn_up.weight"))?,
            down: b.weight_f32(&format!("blk.{layer}.ffn_down.weight"))?,
        });
    }
    Ok(ModelWeights {
        emb: b.weight_f32("token_embd.weight")?,
        layers,
        output_norm: b.weight_f32("output_norm.weight")?,
        output: b.weight_f32("output.weight")?,
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
        let head_dim = hidden / n_heads;
        let vocab = self.conf.vocab_size;
        let mut k_caches: Vec<Vec<f32>> = (0..self.conf.num_layers).map(|_| Vec::new()).collect();
        let mut v_caches: Vec<Vec<f32>> = (0..self.conf.num_layers).map(|_| Vec::new()).collect();

        let mut x = vec![0.0f32; hidden];
        for (pos, &tok) in tokens.iter().enumerate() {
            let tid = (tok as usize) % vocab;
            x.copy_from_slice(&self.weights.emb[tid * hidden..(tid + 1) * hidden]);

            for (li, layer) in self.weights.layers.iter().enumerate() {
                let xn = rms_norm(&x, &layer.attn_norm, 1e-6)?;
                let mut q = linear(&xn, &layer.wq, hidden, hidden)?;
                let mut k = linear(&xn, &layer.wk, hidden, hidden)?;
                let v = linear(&xn, &layer.wv, hidden, hidden)?;
                rope(&mut q, head_dim, pos, self.conf.rope_theta)?;
                rope(&mut k, head_dim, pos, self.conf.rope_theta)?;
                k_caches[li].extend_from_slice(&k);
                v_caches[li].extend_from_slice(&v);
                let attn = attention(&q, &k_caches[li], &v_caches[li], n_heads, n_kv, head_dim)?;
                let ao = linear(&attn, &layer.wo, hidden, hidden)?;
                for i in 0..hidden {
                    x[i] += ao[i];
                }
                let xn2 = rms_norm(&x, &layer.ffn_norm, 1e-6)?;
                let gate = linear(&xn2, &layer.gate, self.conf.intermediate_size, hidden)?;
                let up = linear(&xn2, &layer.up, self.conf.intermediate_size, hidden)?;
                let h = swiglu(&gate, &up)?;
                let down = linear(&h, &layer.down, hidden, self.conf.intermediate_size)?;
                for i in 0..hidden {
                    x[i] += down[i];
                }
            }
        }
        let xn = rms_norm(&x, &self.weights.output_norm, 1e-6)?;
        linear(&xn, &self.weights.output, vocab, hidden)
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
    cache.values().all(|v| v.len() % kv_dim == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{arch_class_representatives, require_stage_b};
    use crate::fixture::write_tiny_q4_bundle;

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
}
