//! MiniCPM weight discovery, loading and PEFT merge (no model instantiation).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use ariacompute_core::contract::{DD_BASE_MODEL, DD_BASE_REVISION};
use ariacompute_core::error::{AfmError, Result};

use crate::checkpoint::DecoderCheckpoint;

pub(crate) const LORA_TARGETS: &[&str] = &["q_proj", "k_proj", "v_proj", "o_proj"];

#[derive(Debug, Deserialize)]
pub(crate) struct AdapterConfig {
    #[serde(default)]
    pub r: usize,
    #[serde(default)]
    pub lora_alpha: f64,
    #[serde(default)]
    pub target_modules: Vec<String>,
}

pub(crate) fn candle_err(e: candle_core::Error) -> AfmError {
    AfmError::msg(e.to_string())
}

pub(crate) fn to_f32(t: Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 {
        Ok(t)
    } else {
        t.to_dtype(DType::F32).map_err(candle_err)
    }
}

/// Map PEFT `...q_proj.lora_A.weight` → `model.layers.N.self_attn.q_proj.weight`.
pub(crate) fn lora_base_key(adapter_key: &str) -> Option<String> {
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

pub(crate) fn looks_like_llama_dir(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join("config.json").is_file() {
        return false;
    }
    !collect_weight_files(dir).unwrap_or_default().is_empty()
}

pub(crate) fn collect_weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
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

pub(crate) fn load_weight_map(
    files: &[PathBuf],
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for f in files {
        let part = candle_core::safetensors::load(f, device)
            .map_err(|e| AfmError::msg(format!("safetensors {}: {e}", f.display())))?;
        for (k, t) in part {
            // Hub MiniCPM ships BF16; candle CPU matmul only supports F32/F16.
            out.insert(k, to_f32(t)?);
        }
    }
    if !out.keys().any(|k| k.contains("layers.0.self_attn")) {
        return Err(AfmError::msg(
            "MiniCPM safetensors have no model.layers.0.self_attn.* keys",
        ));
    }
    Ok(out)
}

pub(crate) fn maybe_tie_lm_head(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    if tensors.contains_key("lm_head.weight") {
        return Ok(());
    }
    if let Some(emb) = tensors.get("model.embed_tokens.weight").cloned() {
        tensors.insert("lm_head.weight".into(), emb);
    }
    Ok(())
}

pub(crate) fn merge_lora_into(
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
        let t = to_f32(t)?;
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
        let w = to_f32(w.clone())?;
        let a = to_f32(a)?;
        let b = to_f32(b)?;
        let delta = b.matmul(&a).map_err(candle_err)?;
        let delta = (delta * scale).map_err(candle_err)?;
        let merged = w.add(&delta).map_err(candle_err)?;
        tensors.insert(base_key, merged);
    }
    Ok(())
}

pub(crate) fn hf_cache_root() -> Option<PathBuf> {
    std::env::var("HF_HUB_CACHE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HF_HOME")
                .ok()
                .map(|h| PathBuf::from(h).join("hub"))
        })
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        })
}

pub(crate) fn hf_snapshot(repo: &str, rev: &str) -> Option<PathBuf> {
    hf_snapshot_in(&hf_cache_root()?, repo, rev)
}

/// Resolve `<cache>/models--org--name/snapshots/<rev>` under an explicit root.
pub(crate) fn hf_snapshot_in(cache_root: &Path, repo: &str, rev: &str) -> Option<PathBuf> {
    let ds = format!("models--{}", repo.replace('/', "--"));
    let snap = cache_root.join(ds).join("snapshots").join(rev);
    snap.is_dir().then_some(snap)
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use serial_test::serial;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn dev() -> Device {
        Device::Cpu
    }

    fn tensor(values: Vec<f32>, shape: (usize, usize)) -> Tensor {
        Tensor::from_vec(values, shape, &dev()).unwrap()
    }

    fn save_tensors(path: &Path, map: HashMap<&str, Tensor>) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        candle_core::safetensors::save(&map, path).unwrap();
    }

    fn llama_dir(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("config.json"), "{}").unwrap();
        save_tensors(
            &dir.join("model.safetensors"),
            HashMap::from([(
                "model.layers.0.self_attn.q_proj.weight",
                tensor(vec![1.0, 0.0, 0.0, 1.0], (2, 2)),
            )]),
        );
    }

    fn clear_env() {
        std::env::remove_var("AFM_DD_BASE");
        std::env::remove_var("HF_HUB_CACHE");
        std::env::remove_var("HF_HOME");
    }

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
        assert_eq!(
            lora_base_key("base_model.model.layers.1.self_attn.v_proj.lora_A.default.weight")
                .as_deref(),
            Some("model.layers.1.self_attn.v_proj.weight")
        );
        assert_eq!(
            lora_base_key("model.model.layers.2.mlp.down_proj.lora_B.default.weight").as_deref(),
            Some("model.layers.2.mlp.down_proj.weight")
        );
        assert_eq!(
            lora_base_key("model.layers.0.other.lora_embedding.weight"),
            None
        );
    }

    #[test]
    fn lora_merge_math() {
        let w = tensor(vec![1.0, 0.0, 0.0, 1.0], (2, 2));
        let a = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev()).unwrap();
        let b = Tensor::from_vec(vec![2.0f32, 0.0], (2, 1), &dev()).unwrap();
        let scale = 32.0 / 16.0;
        let delta = b.matmul(&a).unwrap();
        let merged = (&w + (delta * scale).unwrap()).unwrap();
        let v = merged.to_vec2::<f32>().unwrap();
        assert!((v[0][0] - 5.0).abs() < 1e-5);
        assert!((v[1][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn to_f32_is_idempotent_on_f32_and_converts_f16() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &dev()).unwrap();
        assert_eq!(to_f32(t).unwrap().dtype(), DType::F32);
        let half = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &dev())
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        assert_eq!(to_f32(half).unwrap().dtype(), DType::F32);
    }

    #[test]
    fn missing_base_is_actionable() {
        let ckpt = DecoderCheckpoint::open(None::<&str>).unwrap();
        let msg = missing_weights(&ckpt).to_string();
        assert!(msg.contains("AFM_DD_BASE"));
        assert!(msg.contains("aria-engine download afm-dd"));
        assert!(msg.contains("score_from_semif_out"));
    }

    #[test]
    fn collect_prefers_index_then_single_then_scan() {
        let dir = tempdir().unwrap();
        assert!(collect_weight_files(dir.path()).unwrap().is_empty());

        // scan: picks *.safetensors, skips adapters/lora
        fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"x").unwrap();
        fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"x").unwrap();
        fs::write(dir.path().join("adapter_model.safetensors"), b"x").unwrap();
        fs::write(dir.path().join("lora_weights.safetensors"), b"x").unwrap();
        let files = collect_weight_files(dir.path()).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files[0].to_string_lossy().contains("00001"));

        // single file wins over scan
        let single = tempdir().unwrap();
        fs::write(single.path().join("model.safetensors"), b"x").unwrap();
        fs::write(single.path().join("extra.safetensors"), b"x").unwrap();
        assert_eq!(collect_weight_files(single.path()).unwrap().len(), 1);

        // index lists only files that exist, deduped and sorted
        let indexed = tempdir().unwrap();
        fs::write(indexed.path().join("part-a.safetensors"), b"x").unwrap();
        fs::write(indexed.path().join("part-b.safetensors"), b"x").unwrap();
        fs::write(
            indexed.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"part-b.safetensors","b":"part-a.safetensors","c":"part-a.safetensors","d":"missing.safetensors"}}"#,
        )
        .unwrap();
        let files = collect_weight_files(indexed.path()).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files[0].to_string_lossy().ends_with("part-a.safetensors"));
    }

    #[test]
    fn candle_err_wraps_the_message() {
        let err = candle_err(candle_core::Error::Msg("boom".into()));
        assert!(err.to_string().contains("boom"));
    }

    #[test]
    fn collect_weight_files_tolerates_a_missing_dir() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("gone");
        assert!(collect_weight_files(&missing).unwrap().is_empty());
    }

    #[test]
    fn merge_skips_tensors_that_are_not_lora_pairs() {
        let dir = tempdir().unwrap();
        save_tensors(
            &dir.path().join("adapter_model.safetensors"),
            HashMap::from([
                (
                    "base_model.model.model.layers.0.norm.weight",
                    tensor(vec![1.0, 1.0], (1, 2)),
                ),
                // a lora_* tensor that is not an A/B pair is skipped outright
                (
                    "base_model.model.model.layers.0.q_proj.lora_magnitude_vector",
                    tensor(vec![1.0, 1.0], (1, 2)),
                ),
            ]),
        );
        let mut base = HashMap::from([(
            "model.layers.0.norm.weight".to_string(),
            tensor(vec![2.0, 2.0], (1, 2)),
        )]);
        merge_lora_into(&mut base, dir.path(), &dev()).unwrap();
        // untouched: the adapter key is not a lora_A/lora_B pair
        assert_eq!(
            base["model.layers.0.norm.weight"]
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.0, 2.0]
        );
    }

    #[test]
    fn index_without_weight_map_yields_no_files() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"metadata":{}}"#,
        )
        .unwrap();
        assert!(collect_weight_files(dir.path()).unwrap().is_empty());
    }

    #[test]
    fn broken_index_json_is_an_error() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors.index.json"), "{oops").unwrap();
        assert!(collect_weight_files(dir.path()).is_err());
    }

    #[test]
    fn load_weight_map_converts_and_validates() {
        let dir = tempdir().unwrap();
        llama_dir(dir.path());
        let files = collect_weight_files(dir.path()).unwrap();
        let map = load_weight_map(&files, &dev()).unwrap();
        assert!(map.contains_key("model.layers.0.self_attn.q_proj.weight"));

        let empty = tempdir().unwrap();
        save_tensors(
            &empty.path().join("model.safetensors"),
            HashMap::from([(
                "model.embed_tokens.weight",
                tensor(vec![1.0, 2.0, 3.0, 4.0], (2, 2)),
            )]),
        );
        let files = collect_weight_files(empty.path()).unwrap();
        let err = load_weight_map(&files, &dev()).unwrap_err();
        assert!(err.to_string().contains("layers.0.self_attn"));
    }

    #[test]
    fn load_weight_map_reports_broken_safetensors() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("model.safetensors"), b"not safetensors").unwrap();
        let files = vec![dir.path().join("model.safetensors")];
        let err = load_weight_map(&files, &dev()).unwrap_err();
        assert!(err.to_string().contains("safetensors"));
    }

    #[test]
    fn tie_lm_head_from_embeddings() {
        let mut map = HashMap::from([(
            "model.embed_tokens.weight".to_string(),
            tensor(vec![1.0, 2.0, 3.0, 4.0], (2, 2)),
        )]);
        maybe_tie_lm_head(&mut map).unwrap();
        assert!(map.contains_key("lm_head.weight"));

        let mut with_head = HashMap::from([
            ("lm_head.weight".to_string(), tensor(vec![0.0], (1, 1))),
            (
                "model.embed_tokens.weight".to_string(),
                tensor(vec![1.0], (1, 1)),
            ),
        ]);
        maybe_tie_lm_head(&mut with_head).unwrap();
        assert_eq!(with_head["lm_head.weight"].dims(), &[1, 1]);

        let mut no_emb = HashMap::new();
        maybe_tie_lm_head(&mut no_emb).unwrap();
        assert!(no_emb.is_empty());
    }

    #[test]
    fn merge_lora_applies_delta_and_skips_unpaired() {
        let dir = tempdir().unwrap();
        let adapter = dir.path().join("adapter");
        save_tensors(
            &adapter.join("adapter_model.safetensors"),
            HashMap::from([
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev()).unwrap(),
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    Tensor::from_vec(vec![2.0f32, 0.0], (2, 1), &dev()).unwrap(),
                ),
                // unpaired lora_B for o_proj is skipped
                (
                    "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
                    Tensor::from_vec(vec![1.0f32], (1, 1), &dev()).unwrap(),
                ),
                // not a target module → filtered out
                (
                    "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
                    Tensor::from_vec(vec![1.0f32], (1, 1), &dev()).unwrap(),
                ),
                // not a lora key at all
                ("some.other.weight", tensor(vec![1.0], (1, 1))),
            ]),
        );
        let mut base = HashMap::from([(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            tensor(vec![1.0, 0.0, 0.0, 1.0], (2, 2)),
        )]);
        merge_lora_into(&mut base, &adapter, &dev()).unwrap();
        let merged = base["model.layers.0.self_attn.q_proj.weight"]
            .to_vec2::<f32>()
            .unwrap();
        assert!((merged[0][0] - 5.0).abs() < 1e-5);
        assert!(base.len() == 1);
    }

    #[test]
    fn merge_lora_uses_defaults_when_config_missing_or_zero() {
        let dir = tempdir().unwrap();
        save_tensors(
            &dir.path().join("adapter_model.safetensors"),
            HashMap::from([
                (
                    "model.layers.0.self_attn.k_proj.lora_A.weight",
                    Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev()).unwrap(),
                ),
                (
                    "model.layers.0.self_attn.k_proj.lora_B.weight",
                    Tensor::from_vec(vec![2.0f32, 0.0], (2, 1), &dev()).unwrap(),
                ),
            ]),
        );
        let mut base = HashMap::from([(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            tensor(vec![0.0, 0.0, 0.0, 0.0], (2, 2)),
        )]);
        merge_lora_into(&mut base, dir.path(), &dev()).unwrap();
        let merged = base["model.layers.0.self_attn.k_proj.weight"]
            .to_vec2::<f32>()
            .unwrap();
        // scale = alpha/r = 32/16 = 2 → 2 * 2 = 4
        assert!((merged[0][0] - 4.0).abs() < 1e-5);

        // explicit zero r/alpha fall back to the same defaults
        fs::write(
            dir.path().join("adapter_config.json"),
            r#"{"r":0,"lora_alpha":0.0,"target_modules":[]}"#,
        )
        .unwrap();
        let mut base = HashMap::from([(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            tensor(vec![0.0, 0.0, 0.0, 0.0], (2, 2)),
        )]);
        merge_lora_into(&mut base, dir.path(), &dev()).unwrap();
        let merged = base["model.layers.0.self_attn.k_proj.weight"]
            .to_vec2::<f32>()
            .unwrap();
        assert!((merged[0][0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn merge_lora_reports_missing_adapter_and_missing_targets() {
        let dir = tempdir().unwrap();
        let err = merge_lora_into(&mut HashMap::new(), dir.path(), &dev()).unwrap_err();
        assert!(err
            .to_string()
            .contains("adapter_model.safetensors missing"));

        let missing_target = tempdir().unwrap();
        save_tensors(
            &missing_target.path().join("adapter_model.safetensors"),
            HashMap::from([
                (
                    "model.layers.0.self_attn.v_proj.lora_A.weight",
                    Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev()).unwrap(),
                ),
                (
                    "model.layers.0.self_attn.v_proj.lora_B.weight",
                    Tensor::from_vec(vec![1.0f32, 0.0], (2, 1), &dev()).unwrap(),
                ),
            ]),
        );
        let err = merge_lora_into(&mut HashMap::new(), missing_target.path(), &dev()).unwrap_err();
        assert!(err.to_string().contains("missing from MiniCPM base"));

        // `adapter_model.safetensor` (typo-tolerant alias) is also accepted
        let alias = tempdir().unwrap();
        save_tensors(
            &alias.path().join("adapter_model.safetensor"),
            HashMap::from([("some.other.weight", tensor(vec![1.0], (1, 1)))]),
        );
        merge_lora_into(&mut HashMap::new(), alias.path(), &dev()).unwrap();
    }

    #[test]
    fn merge_lora_reports_broken_adapter_config() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("adapter_config.json"), "{not json").unwrap();
        let err = merge_lora_into(&mut HashMap::new(), dir.path(), &dev()).unwrap_err();
        assert!(matches!(err, AfmError::Json(_)));
    }

    #[test]
    fn looks_like_llama_dir_needs_config_and_weights() {
        let dir = tempdir().unwrap();
        assert!(!looks_like_llama_dir(dir.path()));
        fs::write(dir.path().join("config.json"), "{}").unwrap();
        assert!(!looks_like_llama_dir(dir.path()));
        save_tensors(
            &dir.path().join("model.safetensors"),
            HashMap::from([("x.weight", tensor(vec![1.0], (1, 1)))]),
        );
        assert!(looks_like_llama_dir(dir.path()));
        assert!(!looks_like_llama_dir(&dir.path().join("missing")));
    }

    #[test]
    #[serial]
    fn locate_base_dir_prefers_env_then_root_then_base_then_merged() {
        clear_env();
        let root = tempdir().unwrap();
        let ckpt = DecoderCheckpoint::open(Some(root.path())).unwrap();
        // nothing on disk → actionable error
        let err = locate_base_dir(&ckpt).unwrap_err();
        assert!(err
            .to_string()
            .contains("decoder MiniCPM weights not found"));

        // root > base > merged
        llama_dir(root.path());
        assert_eq!(locate_base_dir(&ckpt).unwrap(), root.path().to_path_buf());

        let only_base = tempdir().unwrap();
        llama_dir(&only_base.path().join("base"));
        let ckpt_base = DecoderCheckpoint::open(Some(only_base.path())).unwrap();
        assert_eq!(
            locate_base_dir(&ckpt_base).unwrap(),
            only_base.path().join("base")
        );

        let only_merged = tempdir().unwrap();
        llama_dir(&only_merged.path().join("merged"));
        let ckpt_merged = DecoderCheckpoint::open(Some(only_merged.path())).unwrap();
        assert_eq!(
            locate_base_dir(&ckpt_merged).unwrap(),
            only_merged.path().join("merged")
        );

        // env override wins
        std::env::set_var("AFM_DD_BASE", root.path());
        assert_eq!(
            locate_base_dir(&ckpt_merged).unwrap(),
            root.path().to_path_buf()
        );

        // env pointing at a non-MiniCPM dir is an explicit error
        std::env::set_var("AFM_DD_BASE", only_merged.path().join("nope"));
        let err = locate_base_dir(&ckpt).unwrap_err();
        assert!(err.to_string().contains("AFM_DD_BASE is not a MiniCPM"));
        clear_env();
    }

    #[test]
    #[serial]
    fn locate_base_dir_falls_back_to_hf_snapshot() {
        clear_env();
        let cache = tempdir().unwrap();
        std::env::set_var("HF_HUB_CACHE", cache.path());
        let repo = "openbmb/MiniCPM5-2B";
        let rev = "abc123";
        let snap = cache
            .path()
            .join("models--openbmb--MiniCPM5-2B")
            .join("snapshots")
            .join(rev);
        llama_dir(&snap);
        let hub = hf_snapshot(repo, rev).unwrap();
        assert_eq!(hub, snap);

        let ckpt = DecoderCheckpoint {
            root: None,
            config: crate::checkpoint::DdConfig {
                base_model: repo.into(),
                base_revision: rev.into(),
                ..crate::checkpoint::DdConfig::default()
            },
            adapter_dir: None,
        };
        assert_eq!(locate_base_dir(&ckpt).unwrap(), snap);

        // HF_HOME/hub is the second choice
        clear_env();
        let hf_home = tempdir().unwrap();
        std::env::set_var("HF_HOME", hf_home.path());
        let snap2 = hf_home
            .path()
            .join("hub")
            .join("models--openbmb--MiniCPM5-2B")
            .join("snapshots")
            .join(rev);
        llama_dir(&snap2);
        assert_eq!(hf_snapshot(repo, rev).unwrap(), snap2);

        // an empty config falls back to the pinned default model/revision
        let default_snap = hf_home
            .path()
            .join("hub")
            .join(format!("models--{}", DD_BASE_MODEL.replace('/', "--")))
            .join("snapshots")
            .join(DD_BASE_REVISION);
        llama_dir(&default_snap);
        let default_ckpt = DecoderCheckpoint {
            root: None,
            config: crate::checkpoint::DdConfig {
                base_model: String::new(),
                base_revision: String::new(),
                ..crate::checkpoint::DdConfig::default()
            },
            adapter_dir: None,
        };
        assert_eq!(locate_base_dir(&default_ckpt).unwrap(), default_snap);

        // a snapshot directory without safetensors is not a usable base
        std::env::set_var("HF_HOME", hf_home.path());
        let bare_rev = "no-weights";
        fs::create_dir_all(
            hf_home
                .path()
                .join("hub")
                .join("models--openbmb--MiniCPM5-2B")
                .join("snapshots")
                .join(bare_rev),
        )
        .unwrap();
        let bare = DecoderCheckpoint {
            root: None,
            config: crate::checkpoint::DdConfig {
                base_model: repo.into(),
                base_revision: bare_rev.into(),
                ..crate::checkpoint::DdConfig::default()
            },
            adapter_dir: None,
        };
        let err = locate_base_dir(&bare).unwrap_err();
        assert!(err
            .to_string()
            .contains("decoder MiniCPM weights not found"));
        clear_env();
    }
}
