//! MiniCPM5-2B (Llama) candle load, optional PEFT merge, last-token letter logits.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use serde::Deserialize;
use tokenizers::Tokenizer;

use ariacompute_core::contract::{DD_BASE_MODEL, DD_BASE_REVISION, MAX_TOKENS_DD};
use ariacompute_core::error::{AfmError, Result};
use ariacompute_core::typed::softmax;

use crate::checkpoint::DecoderCheckpoint;
use crate::prompt::prompt_text;
use crate::semif::LETTERS;

const LORA_TARGETS: &[&str] = &["q_proj", "k_proj", "v_proj", "o_proj"];

#[derive(Debug, Deserialize)]
struct AdapterConfig {
    #[serde(default)]
    r: usize,
    #[serde(default)]
    lora_alpha: f64,
    #[serde(default)]
    target_modules: Vec<String>,
}

pub struct MiniCpmForward {
    llama: Llama,
    cache: Cache,
    device: Device,
    tokenizer: Tokenizer,
    letter_ids: [u32; 16],
    bos_id: Option<u32>,
}

impl MiniCpmForward {
    pub fn load(ckpt: &DecoderCheckpoint) -> Result<Self> {
        let device = Device::Cpu;
        let base = locate_base_dir(ckpt)?;
        let tokenizer = load_tokenizer(ckpt, &base)?;
        let letter_ids = letter_token_ids(&tokenizer)?;
        let bos_id = tokenizer.token_to_id("<s>");

        let weight_files = collect_weight_files(&base)?;
        if weight_files.is_empty() {
            return Err(missing_weights(ckpt));
        }
        let mut tensors = load_weight_map(&weight_files, &device)?;
        maybe_tie_lm_head(&mut tensors)?;
        if let Some(adapter_dir) = &ckpt.adapter_dir {
            merge_lora_into(&mut tensors, adapter_dir, &device)?;
        }

        let cfg_text = fs::read_to_string(base.join("config.json")).map_err(|e| {
            AfmError::msg(format!("read {}: {e}", base.join("config.json").display()))
        })?;
        let llama_cfg: LlamaConfig = serde_json::from_str(&cfg_text)?;
        let mut config = llama_cfg.into_config(false);
        // MiniCPM5 lists 131072 positions; Cache preallocates that many — cap to the SemIf budget.
        if config.max_position_embeddings > MAX_TOKENS_DD {
            config.max_position_embeddings = MAX_TOKENS_DD;
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let llama = Llama::load(vb, &config)
            .map_err(|e| AfmError::msg(format!("MiniCPM Llama::load: {e}")))?;
        let cache = Cache::new(false, DType::F32, &config, &device)
            .map_err(|e| AfmError::msg(format!("MiniCPM Cache: {e}")))?;
        Ok(Self {
            llama,
            cache,
            device,
            tokenizer,
            letter_ids,
            bos_id,
        })
    }

    pub fn letter_logits(&mut self, row: &serde_json::Value) -> Result<(Vec<String>, Vec<f32>)> {
        let prompt = prompt_text(row)?;
        let ids = encode_prompt(&self.tokenizer, &prompt, self.bos_id)?;
        if ids.len() > MAX_TOKENS_DD {
            return Err(AfmError::msg(format!(
                "decoder prompt is {} tokens (max {MAX_TOKENS_DD})",
                ids.len()
            )));
        }
        let seq = ids.len();
        let x = Tensor::new(ids.as_slice(), &self.device)
            .map_err(candle_err)?
            .reshape((1, seq))
            .map_err(candle_err)?;
        let logits = self
            .llama
            .forward(&x, 0, &mut self.cache)
            .map_err(|e| AfmError::msg(format!("MiniCPM forward: {e}")))?;
        let last = last_token_logits(&logits)?;
        let options = row
            .get("options")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AfmError::msg("semif row missing options"))?;
        let n = options.len();
        let mut letter_logits = Vec::with_capacity(n);
        let mut option_ids = Vec::with_capacity(n);
        for (i, opt) in options.iter().enumerate() {
            let tid = self.letter_ids[i] as usize;
            let logit = last
                .get(tid)
                .copied()
                .ok_or_else(|| AfmError::msg(format!("letter logit OOB token {tid}")))?;
            letter_logits.push(logit);
            let id = opt
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            option_ids.push(id);
        }
        let probs = softmax(&letter_logits, 1.0);
        Ok((option_ids, probs))
    }
}

fn candle_err(e: candle_core::Error) -> AfmError {
    AfmError::msg(e.to_string())
}

fn last_token_logits(logits: &Tensor) -> Result<Vec<f32>> {
    let dims = logits.dims();
    let t = match dims.len() {
        1 => logits.clone(),
        2 => logits.i(0).map_err(candle_err)?,
        3 => {
            let seq = dims[1];
            logits.i((0, seq - 1)).map_err(candle_err)?
        }
        n => {
            return Err(AfmError::msg(format!(
                "MiniCPM logits rank {n} unsupported"
            )))
        }
    };
    t.to_vec1::<f32>().map_err(candle_err)
}

fn encode_prompt(tok: &Tokenizer, prompt: &str, bos_id: Option<u32>) -> Result<Vec<u32>> {
    let enc = tok
        .encode(prompt, false)
        .map_err(|e| AfmError::msg(format!("tokenize MiniCPM prompt: {e}")))?;
    let mut ids = enc.get_ids().to_vec();
    if let Some(bos) = bos_id {
        if ids.first().copied() != Some(bos) {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
}

fn letter_token_ids(tok: &Tokenizer) -> Result<[u32; 16]> {
    let mut out = [0u32; 16];
    for (i, ch) in LETTERS.chars().enumerate() {
        let s = ch.to_string();
        if let Some(id) = tok.token_to_id(&s) {
            out[i] = id;
            continue;
        }
        let enc = tok
            .encode(s.as_str(), false)
            .map_err(|e| AfmError::msg(format!("tokenize letter {s}: {e}")))?;
        let ids = enc.get_ids();
        if ids.len() != 1 {
            return Err(AfmError::msg(format!(
                "letter {s} is not a single token ({} ids)",
                ids.len()
            )));
        }
        out[i] = ids[0];
    }
    Ok(out)
}

fn load_tokenizer(ckpt: &DecoderCheckpoint, base: &Path) -> Result<Tokenizer> {
    let mut dirs = Vec::new();
    if let Some(root) = &ckpt.root {
        dirs.push(root.clone());
        if let Some(ad) = &ckpt.adapter_dir {
            if ad != root {
                dirs.push(ad.clone());
            }
        }
    }
    dirs.push(base.to_path_buf());
    for dir in dirs {
        for rel in ["tokenizer.json", "tokenizer/tokenizer.json"] {
            let path = dir.join(rel);
            if path.is_file() {
                return Tokenizer::from_file(&path)
                    .map_err(|e| AfmError::msg(format!("tokenizer load {}: {e}", path.display())));
            }
        }
    }
    Err(AfmError::msg(format!(
        "tokenizer.json not found next to decoder checkpoint or {}",
        base.display()
    )))
}

pub fn locate_base_dir(ckpt: &DecoderCheckpoint) -> Result<PathBuf> {
    if let Ok(p) = std::env::var("AFM_DD_BASE") {
        let path = PathBuf::from(p);
        if looks_like_llama_dir(&path) {
            return Ok(path);
        }
        return Err(AfmError::msg(format!(
            "AFM_DD_BASE is not a MiniCPM safetensors dir: {}",
            path.display()
        )));
    }
    if let Some(root) = &ckpt.root {
        if looks_like_llama_dir(root) {
            return Ok(root.clone());
        }
        let base = root.join("base");
        if looks_like_llama_dir(&base) {
            return Ok(base);
        }
        let merged = root.join("merged");
        if looks_like_llama_dir(&merged) {
            return Ok(merged);
        }
    }
    let repo = if ckpt.config.base_model.is_empty() {
        DD_BASE_MODEL
    } else {
        ckpt.config.base_model.as_str()
    };
    let rev = if ckpt.config.base_revision.is_empty() {
        DD_BASE_REVISION
    } else {
        ckpt.config.base_revision.as_str()
    };
    if let Some(hub) = hf_snapshot(repo, rev) {
        if looks_like_llama_dir(&hub) {
            return Ok(hub);
        }
    }
    Err(missing_weights(ckpt))
}

pub fn missing_weights(ckpt: &DecoderCheckpoint) -> AfmError {
    AfmError::msg(format!(
        "decoder MiniCPM weights not found (base={}, rev={}). \
Run `aria-engine download afm-dd` (fills PEFT + base/ safetensors), or set AFM_DD_BASE \
to a local MiniCPM5-2B snapshot. Engine candle does not load GGUF. \
Golden tests can use score_from_semif_out.",
        ckpt.config.base_model, ckpt.config.base_revision
    ))
}

fn looks_like_llama_dir(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join("config.json").is_file() {
        return false;
    }
    !collect_weight_files(dir).unwrap_or_default().is_empty()
}

fn collect_weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(&index)?)?;
        let mut files = Vec::new();
        if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
            let mut names: Vec<&str> = map.values().filter_map(|x| x.as_str()).collect();
            names.sort_unstable();
            names.dedup();
            for name in names {
                let p = dir.join(name);
                if p.is_file() {
                    files.push(p);
                }
            }
        }
        return Ok(files);
    }
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let mut files = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for ent in rd.flatten() {
            let name = ent.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".safetensors")
                && !name.starts_with("adapter")
                && !name.contains("lora")
            {
                files.push(ent.path());
            }
        }
    }
    files.sort();
    Ok(files)
}

fn load_weight_map(files: &[PathBuf], device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for f in files {
        let part = candle_core::safetensors::load(f, device)
            .map_err(|e| AfmError::msg(format!("safetensors {}: {e}", f.display())))?;
        out.extend(part);
    }
    if !out.keys().any(|k| k.contains("layers.0.self_attn")) {
        return Err(AfmError::msg(
            "MiniCPM safetensors have no model.layers.0.self_attn.* keys",
        ));
    }
    Ok(out)
}

fn maybe_tie_lm_head(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    if tensors.contains_key("lm_head.weight") {
        return Ok(());
    }
    if let Some(emb) = tensors.get("model.embed_tokens.weight").cloned() {
        tensors.insert("lm_head.weight".into(), emb);
    }
    Ok(())
}

fn merge_lora_into(
    tensors: &mut HashMap<String, Tensor>,
    adapter_dir: &Path,
    device: &Device,
) -> Result<()> {
    let cfg_path = adapter_dir.join("adapter_config.json");
    let cfg: AdapterConfig = if cfg_path.is_file() {
        serde_json::from_str(&fs::read_to_string(&cfg_path)?)?
    } else {
        AdapterConfig {
            r: 16,
            lora_alpha: 32.0,
            target_modules: LORA_TARGETS.iter().map(|s| (*s).to_string()).collect(),
        }
    };
    let r = if cfg.r == 0 { 16 } else { cfg.r };
    let alpha = if cfg.lora_alpha == 0.0 {
        32.0
    } else {
        cfg.lora_alpha
    };
    let scale = alpha / r as f64;
    let adapter_file = ["adapter_model.safetensors", "adapter_model.safetensor"]
        .iter()
        .map(|n| adapter_dir.join(n))
        .find(|p| p.is_file())
        .ok_or_else(|| {
            AfmError::msg(format!(
                "PEFT adapter_model.safetensors missing in {}",
                adapter_dir.display()
            ))
        })?;
    let adapter = candle_core::safetensors::load(&adapter_file, device)
        .map_err(|e| AfmError::msg(format!("adapter {}: {e}", adapter_file.display())))?;

    let mut pairs: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();
    for (k, t) in adapter {
        let Some(base_key) = lora_base_key(&k) else {
            continue;
        };
        if !cfg.target_modules.is_empty()
            && !cfg
                .target_modules
                .iter()
                .any(|m| base_key.contains(&format!(".{m}.")))
            && !cfg
                .target_modules
                .iter()
                .any(|m| base_key.ends_with(&format!(".{m}.weight")))
        {
            continue;
        }
        let entry = pairs.entry(base_key).or_insert((None, None));
        if k.contains("lora_A") {
            entry.0 = Some(t);
        } else if k.contains("lora_B") {
            entry.1 = Some(t);
        }
    }
    for (base_key, (a, b)) in pairs {
        let (Some(a), Some(b)) = (a, b) else {
            continue;
        };
        let Some(w) = tensors.get(&base_key) else {
            return Err(AfmError::msg(format!(
                "LoRA target {base_key} missing from MiniCPM base"
            )));
        };
        let a = a.to_dtype(w.dtype()).map_err(candle_err)?;
        let b = b.to_dtype(w.dtype()).map_err(candle_err)?;
        let delta = b.matmul(&a).map_err(candle_err)?;
        let delta = (delta * scale).map_err(candle_err)?;
        let merged = w.add(&delta).map_err(candle_err)?;
        tensors.insert(base_key, merged);
    }
    Ok(())
}

/// Map PEFT `...q_proj.lora_A.weight` → `model.layers.N.self_attn.q_proj.weight`.
fn lora_base_key(adapter_key: &str) -> Option<String> {
    let mut k = adapter_key.to_string();
    for prefix in [
        "base_model.model.model.",
        "base_model.model.",
        "model.model.",
    ] {
        if let Some(rest) = k.strip_prefix(prefix) {
            k = format!("model.{rest}");
            break;
        }
    }
    let k = k.replace(".lora_A.default.weight", ".weight");
    let k = k.replace(".lora_B.default.weight", ".weight");
    let k = k.replace(".lora_A.weight", ".weight");
    let k = k.replace(".lora_B.weight", ".weight");
    if k.contains("lora_") {
        return None;
    }
    Some(k)
}

fn hf_snapshot(repo: &str, rev: &str) -> Option<PathBuf> {
    let cache_root = std::env::var("HF_HUB_CACHE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HF_HOME")
                .ok()
                .map(|h| PathBuf::from(h).join("hub"))
        })
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        })?;
    let ds = format!("models--{}", repo.replace('/', "--"));
    let snap = cache_root.join(ds).join("snapshots").join(rev);
    snap.is_dir().then_some(snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn lora_key_strips_peft_prefix() {
        assert_eq!(
            lora_base_key("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
                .as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            lora_base_key("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight")
                .as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
    }

    #[test]
    fn lora_merge_math() {
        let dev = Device::Cpu;
        let w = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &dev).unwrap();
        let a = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let b = Tensor::from_vec(vec![2.0f32, 0.0], (2, 1), &dev).unwrap();
        let scale = 32.0 / 16.0;
        let delta = b.matmul(&a).unwrap();
        let merged = (&w + (delta * scale).unwrap()).unwrap();
        let v = merged.to_vec2::<f32>().unwrap();
        assert!((v[0][0] - 5.0).abs() < 1e-5);
        assert!((v[1][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn missing_base_is_actionable() {
        let ckpt = DecoderCheckpoint::open(None::<&str>).unwrap();
        let msg = missing_weights(&ckpt).to_string();
        assert!(msg.contains("AFM_DD_BASE"));
        assert!(msg.contains("aria-engine download afm-dd"));
        assert!(msg.contains("score_from_semif_out"));
    }
}
