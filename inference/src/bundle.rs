use crate::pack::unpack_indices;
use aria_kernel::{
    dequant_lookup_group, hadamard_blocked_rows_tiles, pow2_tile_sizes, EngineError,
};
use half::f16;
use memmap2::Mmap;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const BUNDLE_FORMAT: &str = "aria-quant-bundle";

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<usize>,
    #[serde(default)]
    pub use_double_wide_mlp: Option<bool>,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    /// LFM2 short-conv kernel / cache length (HF `conv_L_cache`, default 3).
    #[serde(default)]
    pub conv_l_cache: Option<usize>,
    /// Sliding-window attention length (HF `sliding_window`). Required for gemma-4.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Gemma-4 global p-RoPE fraction (HF `partial_rotary_factor`). Required for gemma-4.
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    /// Gemma-4 full-attention head dim (HF `global_head_dim`). Required for gemma-4.
    #[serde(default)]
    pub global_head_dim: Option<usize>,
}

fn default_rope() -> f32 {
    10000.0
}

#[derive(Debug, Deserialize)]
struct BundleConfig {
    format: String,
    format_version: u32,
    #[allow(dead_code)]
    quantization: String,
    #[serde(default)]
    group_size_default: usize,
    #[serde(default)]
    hadamard_seed: Option<i64>,
    model: ModelConfig,
    tensors: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct QuantTensor {
    pub bits: u8,
    pub group_size: usize,
    pub shape: (usize, usize),
    pub row_pad: usize,
    pub codebook_share: String,
    pub packed_indices: Vec<u8>,
    pub codebook: Vec<f32>,
    pub codebook_shape: Vec<usize>,
    pub hadamard: Value,
}

#[derive(Debug, Clone)]
pub enum TensorData {
    Codebook(QuantTensor),
    Raw {
        dtype: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    },
}

pub struct Bundle {
    pub path: PathBuf,
    pub model: ModelConfig,
    pub quantization: String,
    pub group_size_default: usize,
    pub hadamard_seed: Option<i64>,
    pub tensors: HashMap<String, TensorData>,
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
}

impl std::fmt::Debug for Bundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bundle")
            .field("path", &self.path)
            .field("quantization", &self.quantization)
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

fn read_slice(mmap: &Mmap, start: usize, len: usize) -> Result<&[u8], EngineError> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| EngineError::Format("offset overflow".into()))?;
    if end > mmap.len() {
        return Err(EngineError::Format(format!(
            "offset [{start},{len}] out of range (bin size {})",
            mmap.len()
        )));
    }
    Ok(&mmap[start..end])
}

fn offset_pair(v: &Value, key: &str) -> Result<(usize, usize), EngineError> {
    let arr = v
        .get(key)
        .and_then(|x| x.as_array())
        .ok_or_else(|| EngineError::Format(format!("missing offset {key}")))?;
    if arr.len() != 2 {
        return Err(EngineError::Format(format!("bad offset {key}")));
    }
    let s = arr[0]
        .as_u64()
        .ok_or_else(|| EngineError::Format("offset start".into()))? as usize;
    let l = arr[1]
        .as_u64()
        .ok_or_else(|| EngineError::Format("offset len".into()))? as usize;
    Ok((s, l))
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, EngineError> {
    if !bytes.len().is_multiple_of(2) {
        return Err(EngineError::Format("f16 byte length odd".into()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.as_chunks::<2>().0 {
        let h = f16::from_le_bytes(*c);
        out.push(h.to_f32());
    }
    Ok(out)
}

fn f32_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, EngineError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(EngineError::Format("f32 byte length not aligned".into()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*c));
    }
    Ok(out)
}

pub fn load_bundle(path: impl AsRef<Path>) -> Result<Bundle, EngineError> {
    let path = path.as_ref();
    let cfg_path = path.join("config.json");
    let bin_path = path.join("weight.bin");
    if !cfg_path.is_file() {
        return Err(EngineError::Format(format!(
            "missing config.json in {}",
            path.display()
        )));
    }
    if !bin_path.is_file() {
        return Err(EngineError::Format(format!(
            "missing weight.bin in {}",
            path.display()
        )));
    }
    let cfg_text = std::fs::read_to_string(&cfg_path)?;
    let cfg: BundleConfig =
        serde_json::from_str(&cfg_text).map_err(|e| EngineError::Format(e.to_string()))?;
    if cfg.format != BUNDLE_FORMAT {
        return Err(EngineError::Format(format!(
            "unsupported format {:?}",
            cfg.format
        )));
    }
    if cfg.format_version != 1 && cfg.format_version != 2 {
        return Err(EngineError::Format(format!(
            "unsupported format_version {}",
            cfg.format_version
        )));
    }
    let file = File::open(&bin_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let mmap = Arc::new(mmap);

    let mut tensors = HashMap::new();
    for (name, meta) in &cfg.tensors {
        let kind = meta
            .get("kind")
            .and_then(|v| v.as_str())
            .ok_or_else(|| EngineError::Format(format!("tensor {name} missing kind")))?;
        let offsets = meta
            .get("offsets")
            .ok_or_else(|| EngineError::Format(format!("tensor {name} missing offsets")))?;
        match kind {
            "codebook" => {
                let bits = meta
                    .get("bits")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| EngineError::Quant("bits".into()))?
                    as u8;
                if !matches!(bits, 1 | 2 | 3 | 4 | 8) {
                    return Err(EngineError::Quant(format!("unsupported bits {bits}")));
                }
                let group_size =
                    meta.get("group_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(cfg.group_size_default as u64) as usize;
                let shape = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| EngineError::Format("shape".into()))?;
                if shape.len() != 2 {
                    return Err(EngineError::Format("codebook shape must be [K,N]".into()));
                }
                let k = shape[0].as_u64().unwrap() as usize;
                let n = shape[1].as_u64().unwrap() as usize;
                let row_pad = meta.get("row_pad").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let share = meta
                    .get("codebook_share")
                    .and_then(|v| v.as_str())
                    .unwrap_or("group")
                    .to_string();
                let (ps, pl) = offset_pair(offsets, "packed_indices")?;
                let (cs, cl) = offset_pair(offsets, "codebook")?;
                let packed = read_slice(&mmap, ps, pl)?.to_vec();
                let cb_raw = read_slice(&mmap, cs, cl)?;
                let codebook = f16_bytes_to_f32(cb_raw)?;
                let kc = 1usize << bits;
                let codebook_shape = if share == "group" {
                    if !codebook.len().is_multiple_of(kc) {
                        return Err(EngineError::ShapeMismatch("bad group codebook size".into()));
                    }
                    vec![codebook.len() / kc, kc]
                } else {
                    if n * kc == 0 || codebook.len() % (n * kc) != 0 {
                        return Err(EngineError::ShapeMismatch(
                            "bad channel codebook size".into(),
                        ));
                    }
                    let g = codebook.len() / (n * kc);
                    vec![g, n, kc]
                };
                let hadamard = meta
                    .get("hadamard")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));
                tensors.insert(
                    name.clone(),
                    TensorData::Codebook(QuantTensor {
                        bits,
                        group_size,
                        shape: (k, n),
                        row_pad,
                        codebook_share: share,
                        packed_indices: packed,
                        codebook,
                        codebook_shape,
                        hadamard,
                    }),
                );
            }
            "raw" => {
                let dtype = meta
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("f16")
                    .to_string();
                let shape: Vec<usize> = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| EngineError::Format("raw shape".into()))?
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                let (ds, dl) = offset_pair(offsets, "data")?;
                let raw = read_slice(&mmap, ds, dl)?;
                let data = if dtype == "f32" {
                    f32_bytes_to_f32(raw)?
                } else {
                    f16_bytes_to_f32(raw)?
                };
                tensors.insert(name.clone(), TensorData::Raw { dtype, shape, data });
            }
            other => {
                return Err(EngineError::Format(format!(
                    "unknown tensor kind {other:?}"
                )));
            }
        }
    }

    Ok(Bundle {
        path: path.to_path_buf(),
        model: cfg.model,
        quantization: cfg.quantization,
        group_size_default: cfg.group_size_default,
        hadamard_seed: cfg.hadamard_seed,
        tensors,
        mmap,
    })
}

/// Rotated-space reconstruction (matches Python `dequantize`).
pub fn dequantize(t: &QuantTensor) -> Result<Vec<f32>, EngineError> {
    let (k0, n) = t.shape;
    let gs = t.group_size;
    let kc = 1usize << t.bits;
    if t.codebook_share == "group" {
        if t.codebook_shape.len() != 2 {
            return Err(EngineError::ShapeMismatch(
                "group codebook must be 2D".into(),
            ));
        }
        let num_groups = t.codebook_shape[0];
        let k_work = num_groups * gs;
        let expected = k_work * n;
        let indices = unpack_indices(&t.packed_indices, expected, t.bits)?;
        dequant_lookup_group(&indices, &t.codebook, num_groups, gs, n, kc, k0)
    } else {
        // channel share
        if t.codebook_shape.len() != 3 {
            return Err(EngineError::ShapeMismatch(
                "channel codebook must be 3D".into(),
            ));
        }
        let num_groups = t.codebook_shape[0];
        let k_work = num_groups * gs;
        let expected = k_work * n;
        let indices = unpack_indices(&t.packed_indices, expected, t.bits)?;
        let mut out = vec![0.0f32; k_work * n];
        for g in 0..num_groups {
            for r in 0..gs {
                let row = g * gs + r;
                for j in 0..n {
                    let idx = indices[row * n + j] as usize;
                    let base = (g * n + j) * kc;
                    out[row * n + j] = t.codebook[base + idx];
                }
            }
        }
        out.truncate(k0 * n);
        Ok(out)
    }
}

impl Bundle {
    pub fn weight_f32(&self, name: &str) -> Result<Vec<f32>, EngineError> {
        Ok(self.weight_loaded(name)?.data)
    }

    /// Load a weight in **original space** (Python `reconstruct_weight`).
    ///
    /// Codebook tensors are stored rotated (`W_rot = H@S@W` on axis 0). Fused
    /// `hdm_linear` unrotates `y = W_rot @ x`, which is valid for dense GEMM but
    /// **not** for embedding row gather (`e = W[token]`). Unrotating the full
    /// matrix here makes lookup, tied `lm_head`, and `linear` all match HF inject.
    pub fn weight_loaded(&self, name: &str) -> Result<LoadedWeight, EngineError> {
        match self.tensors.get(name) {
            Some(TensorData::Codebook(q)) => {
                let t0 = std::time::Instant::now();
                let mut data = dequantize(q)?;
                crate::profile::load_profile_add_dequant(crate::profile::elapsed_ms(t0));
                let applied = q
                    .hadamard
                    .get("applied")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if applied {
                    let (k0, n) = q.shape;
                    if data.len() != k0 * n {
                        return Err(EngineError::ShapeMismatch(format!(
                            "dequant {name} len {} != shape {k0}*{n}",
                            data.len()
                        )));
                    }
                    let seed = self
                        .hadamard_seed
                        .or_else(|| q.hadamard.get("seed").and_then(|v| v.as_i64()));
                    if k0 > 1 {
                        let t1 = std::time::Instant::now();
                        let tiles = hadamard_tile_sizes_from_meta(&q.hadamard, k0)?;
                        hadamard_blocked_rows_tiles(&mut data, k0, n, seed, true, &tiles)?;
                        crate::profile::load_profile_add_unrotate(crate::profile::elapsed_ms(t1));
                    }
                }
                Ok(LoadedWeight {
                    data,
                    hdm_seed: None,
                })
            }
            Some(TensorData::Raw { data, .. }) => Ok(LoadedWeight {
                data: data.clone(),
                hdm_seed: None,
            }),
            None => Err(EngineError::Format(format!("missing tensor {name}"))),
        }
    }

    /// Load the first tensor that exists among `names` (HF vs tiny/blk aliases).
    pub fn weight_f32_any(&self, names: &[&str]) -> Result<Vec<f32>, EngineError> {
        Ok(self.weight_loaded_any(names)?.data)
    }

    pub fn weight_loaded_any(&self, names: &[&str]) -> Result<LoadedWeight, EngineError> {
        let mut tried = Vec::with_capacity(names.len());
        for name in names {
            match self.weight_loaded(name) {
                Ok(v) => return Ok(v),
                Err(EngineError::Format(_)) => tried.push(*name),
                Err(e) => return Err(e),
            }
        }
        Err(EngineError::Format(format!(
            "missing tensor (tried {})",
            tried.join(", ")
        )))
    }
}

/// Tile sizes from Python `hadamard.blocks` (`[{start,size},…]`); greedy pow2 if absent/invalid.
fn hadamard_tile_sizes_from_meta(hadamard: &Value, rows: usize) -> Result<Vec<usize>, EngineError> {
    if let Some(blocks) = hadamard.get("blocks").and_then(|v| v.as_array()) {
        if !blocks.is_empty() {
            let mut sizes = Vec::with_capacity(blocks.len());
            let mut pos = 0usize;
            let mut ok = true;
            for b in blocks {
                let start = b.get("start").and_then(|x| x.as_u64()).map(|x| x as usize);
                let size = b.get("size").and_then(|x| x.as_u64()).map(|x| x as usize);
                match (start, size) {
                    (Some(s), Some(sz)) if s == pos && sz > 0 => {
                        sizes.push(sz);
                        pos = pos.saturating_add(sz);
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && pos == rows {
                return Ok(sizes);
            }
        }
    }
    pow2_tile_sizes(rows)
}

/// Dequantized original-space weight (codebook tensors already unrotated).
///
/// `hdm_seed` is kept for API compatibility; `weight_loaded` always clears it
/// so Session GEMM uses `linear` on reconstructed `W`.
#[derive(Debug, Clone)]
pub struct LoadedWeight {
    pub data: Vec<f32>,
    pub hdm_seed: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{
        make_channel_quant_tensor, make_group_quant_tensor, rel_rmse, write_tiny_q4_bundle,
    };
    use aria_kernel::hadamard_blocked_rows;
    use serde_json::json;

    #[test]
    fn load_and_dequant() {
        let dir = tempfile::tempdir().unwrap();
        let (rmse, _) = write_tiny_q4_bundle(dir.path()).unwrap();
        assert!(rmse < 0.5, "rmse {rmse}");
        let b = load_bundle(dir.path()).unwrap();
        assert_eq!(b.model.hidden_size, 64);
        let w = b.weight_f32("blk.0.attn_q.weight").unwrap();
        assert_eq!(w.len(), 64 * 64);
        // Core guarantee: hadamard.applied + codebook K=16 for q4.
        match b.tensors.get("blk.0.attn_q.weight").unwrap() {
            TensorData::Codebook(q) => {
                assert_eq!(q.bits, 4);
                assert_eq!(1usize << q.bits, q.codebook_shape[1]);
                assert_eq!(q.hadamard.get("applied"), Some(&json!(true)));
            }
            _ => panic!("expected codebook"),
        }
        assert!(matches!(
            b.tensors.get("blk.0.attn_norm.weight"),
            Some(TensorData::Raw { .. })
        ));
    }

    #[test]
    fn bad_format() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"format":"nope"}"#).unwrap();
        std::fs::write(dir.path().join("weight.bin"), b"").unwrap();
        let err = load_bundle(dir.path()).unwrap_err();
        assert!(matches!(err, EngineError::Format(_)));
    }

    #[test]
    fn missing_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            load_bundle(dir.path()),
            Err(EngineError::Format(_))
        ));
    }

    #[test]
    fn load_v2_blocked_hadamard_meta() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let cfg_text = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
        let cfg: serde_json::Value = serde_json::from_str(&cfg_text).unwrap();
        assert_eq!(cfg["format_version"], 2);
        assert_eq!(cfg["hadamard_seed"], 0);
        let b = load_bundle(dir.path()).unwrap();
        assert_eq!(b.hadamard_seed, Some(0));
        match b.tensors.get("blk.0.attn_q.weight").unwrap() {
            TensorData::Codebook(q) => {
                assert_eq!(q.hadamard.get("mode"), Some(&json!("blocked")));
                assert_eq!(q.hadamard.get("applied"), Some(&json!(true)));
                let blocks = q.hadamard["blocks"].as_array().expect("blocks");
                assert!(!blocks.is_empty());
                let k = q.shape.0;
                let covered: usize = blocks
                    .iter()
                    .map(|b| b["size"].as_u64().unwrap() as usize)
                    .sum();
                assert_eq!(covered, k);
                assert_eq!(blocks[0]["start"], 0);
                // First tile is largest power-of-two ≤ k.
                let first = blocks[0]["size"].as_u64().unwrap() as usize;
                assert!(first.is_power_of_two());
                assert!(first <= k);
                if k > first {
                    assert_eq!(blocks[1]["start"], first as u64);
                }
            }
            _ => panic!("expected codebook"),
        }
    }

    #[test]
    fn codebook_weight_loaded_unrotates_like_reconstruct() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let b = load_bundle(dir.path()).unwrap();
        let name = "blk.0.attn_q.weight";
        let q = match b.tensors.get(name).unwrap() {
            TensorData::Codebook(q) => q,
            _ => panic!("expected codebook"),
        };
        let mut expected = dequantize(q).unwrap();
        let (k0, n) = q.shape;
        let seed = b
            .hadamard_seed
            .or_else(|| q.hadamard.get("seed").and_then(|v| v.as_i64()));
        hadamard_blocked_rows(&mut expected, k0, n, seed, true).unwrap();
        let loaded = b.weight_loaded(name).unwrap();
        assert!(
            loaded.hdm_seed.is_none(),
            "reconstructed weights are original-space; Session uses linear()"
        );
        assert_eq!(loaded.data.len(), expected.len());
        for (a, e) in loaded.data.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "{a} vs {e}");
        }
        // Rotated dequant row must differ from reconstructed row (axis-0 mix).
        let rotated = dequantize(q).unwrap();
        let row = n;
        let rot_norm: f32 = rotated[..row]
            .iter()
            .zip(loaded.data[..row].iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            rot_norm.sqrt() > 1e-4,
            "embedding/linear rows must change under blocked unrotate"
        );
    }

    #[test]
    fn load_accepts_format_version_1() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let cfg_path = dir.path().join("config.json");
        let mut cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        cfg["format_version"] = json!(1);
        // Legacy fixtures may omit blocked mode; loader must still accept v1.
        if let Some(tensors) = cfg["tensors"].as_object_mut() {
            for meta in tensors.values_mut() {
                if meta.get("kind") == Some(&json!("codebook")) {
                    if let Some(h) = meta.get_mut("hadamard") {
                        if let Some(o) = h.as_object_mut() {
                            o.remove("mode");
                            o.remove("blocks");
                        }
                    }
                }
            }
        }
        std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg).unwrap()).unwrap();
        let b = load_bundle(dir.path()).unwrap();
        assert!(!b.tensors.is_empty());
    }

    #[test]
    fn load_rejects_format_version_3() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_q4_bundle(dir.path()).unwrap();
        let cfg_path = dir.path().join("config.json");
        let mut cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        cfg["format_version"] = json!(3);
        std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg).unwrap()).unwrap();
        let err = load_bundle(dir.path()).unwrap_err();
        assert!(matches!(err, EngineError::Format(_)));
        let msg = format!("{err}");
        assert!(msg.contains("format_version"), "{msg}");
    }

    /// Mirror model `test_quant.test_dequant_error_bounds` (linspace stand-in; Spec-ish bands).
    #[test]
    fn dequant_error_bounds_group() {
        let mut rng = 0u64;
        let mut randn = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((rng >> 33) as f32) / (u32::MAX as f32);
            (u - 0.5) * 2.0
        };
        let k = 64usize;
        let n = 16usize;
        let mut w = vec![0.0f32; k * n];
        for v in &mut w {
            *v = randn();
        }
        // Linspace is coarser than Lloyd-Max; keep Spec-aligned upper bands.
        let bounds = [(8u8, 0.25f32), (4, 0.45), (3, 0.60), (2, 0.85), (1, 1.20)];
        for (bits, lim) in bounds {
            let t = make_group_quant_tensor(&w, k, n, 32, bits);
            assert_eq!(t.codebook_shape[1], 1usize << bits);
            assert_eq!(t.hadamard.get("applied"), Some(&json!(true)));
            let recon = dequantize(&t).unwrap();
            assert_eq!(recon.len(), k * n);
            let err = rel_rmse(&w, &recon);
            assert!(err <= lim, "q{bits} group rel_rmse={err} > {lim}");
        }
    }

    #[test]
    fn dequant_channel_q4_tighter() {
        let mut rng = 1u64;
        let mut randn = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((rng >> 33) as f32) / (u32::MAX as f32);
            (u - 0.5) * 2.0
        };
        let k = 64usize;
        let n = 16usize;
        let mut w = vec![0.0f32; k * n];
        for v in &mut w {
            *v = randn();
        }
        let t = make_channel_quant_tensor(&w, k, n, 32, 4);
        assert_eq!(t.codebook_shape, vec![2, n, 16]);
        let g = make_group_quant_tensor(&w, k, n, 32, 4);
        assert!(t.codebook.len() > g.codebook.len() * 8);
        let recon = dequantize(&t).unwrap();
        let err = rel_rmse(&w, &recon);
        assert!(err <= 0.35, "q4 channel rel_rmse={err}");
    }

    #[test]
    fn dequant_bad_share_shape() {
        let mut t = make_group_quant_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2, 2, 4);
        t.codebook_share = "channel".into(); // shape still 2D → error
        assert!(matches!(dequantize(&t), Err(EngineError::ShapeMismatch(_))));
    }

    /// Optional golden path: `ARIA_TINY_BUNDLE` → model `--tiny` export directory.
    #[test]
    fn load_aria_tiny_bundle_from_env() {
        let Ok(path) = std::env::var("ARIA_TINY_BUNDLE") else {
            return;
        };
        let b = load_bundle(&path).expect("ARIA_TINY_BUNDLE must be a valid aria-quant-bundle");
        assert_eq!(b.quantization.chars().next(), Some('q'));
        assert!(b.model.hidden_size > 0);
        assert!(!b.tensors.is_empty());
        // At least one codebook tensor dequants to declared shape.
        let (name, q) = b
            .tensors
            .iter()
            .find_map(|(n, t)| match t {
                TensorData::Codebook(q) => Some((n, q)),
                _ => None,
            })
            .expect("bundle has codebook tensors");
        let recon = dequantize(q).unwrap();
        assert_eq!(recon.len(), q.shape.0 * q.shape.1, "{name}");
        assert_eq!(q.hadamard.get("applied"), Some(&json!(true)), "{name}");
    }

    #[test]
    fn hadamard_tiles_prefer_bundle_blocks() {
        let greedy = pow2_tile_sizes(10).unwrap();
        assert_eq!(greedy, vec![8, 2]);
        let meta = json!({
            "applied": true,
            "mode": "blocked",
            "blocks": [{"start": 0, "size": 8}, {"start": 8, "size": 2}]
        });
        assert_eq!(hadamard_tile_sizes_from_meta(&meta, 10).unwrap(), greedy);
        let bad = json!({"blocks": [{"start": 0, "size": 4}]});
        assert_eq!(hadamard_tile_sizes_from_meta(&bad, 10).unwrap(), greedy);
        let empty = json!({});
        assert_eq!(hadamard_tile_sizes_from_meta(&empty, 10).unwrap(), greedy);
    }

    #[test]
    fn gemma4_embed_and_ple_codebook_row_gather() {
        // Mirrors model quantize_weight: axis-0 rotate, group codebook, LSB pack,
        // shape [vocab, cols] — engine must unrotate then gather token rows.
        use half::f16;
        let vocab = 10usize;
        let hidden = 8usize;
        let packed_ple = 12usize; // 3 layers * 4
        let gs = 8usize;
        let seed = Some(0i64);

        let mut emb: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32) * 0.01 - 0.05)
            .collect();
        let mut ple: Vec<f32> = (0..vocab * packed_ple)
            .map(|i| (i as f32) * 0.003 - 0.02)
            .collect();
        let emb_orig = emb.clone();
        let ple_orig = ple.clone();
        hadamard_blocked_rows(&mut emb, vocab, hidden, seed, false).unwrap();
        hadamard_blocked_rows(&mut ple, vocab, packed_ple, seed, false).unwrap();

        let write_cb = |name: &str,
                        w_rot: &[f32],
                        k: usize,
                        n: usize,
                        bin: &mut Vec<u8>,
                        tensors: &mut serde_json::Map<String, Value>| {
            let t = make_group_quant_tensor(w_rot, k, n, gs, 4);
            let pi_s = bin.len();
            bin.extend_from_slice(&t.packed_indices);
            let pi_l = bin.len() - pi_s;
            let cb_s = bin.len();
            for &v in &t.codebook {
                bin.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
            let cb_l = bin.len() - cb_s;
            let mut blocks = Vec::new();
            let mut start = 0usize;
            for sz in pow2_tile_sizes(k).unwrap() {
                blocks.push(json!({"start": start, "size": sz}));
                start += sz;
            }
            tensors.insert(
                name.to_string(),
                json!({
                    "kind": "codebook",
                    "bits": 4,
                    "group_size": gs,
                    "shape": [k, n],
                    "row_pad": 0,
                    "codebook_share": "group",
                    "hadamard": {
                        "applied": true,
                        "axis": 0,
                        "seed": 0,
                        "mode": "blocked",
                        "blocks": blocks
                    },
                    "offsets": {
                        "packed_indices": [pi_s, pi_l],
                        "codebook": [cb_s, cb_l]
                    }
                }),
            );
        };

        let dir = tempfile::tempdir().unwrap();
        let mut bin = Vec::new();
        let mut tensors = serde_json::Map::new();
        write_cb(
            "model.language_model.embed_tokens.weight",
            &emb,
            vocab,
            hidden,
            &mut bin,
            &mut tensors,
        );
        write_cb(
            "model.language_model.embed_tokens_per_layer.weight",
            &ple,
            vocab,
            packed_ple,
            &mut bin,
            &mut tensors,
        );
        let cfg = json!({
            "format": "aria-quant-bundle",
            "format_version": 2,
            "quantization": "q4",
            "group_size_default": gs,
            "hadamard_seed": 0,
            "model": {
                "hidden_size": hidden,
                "num_layers": 3,
                "num_attention_heads": 2,
                "num_kv_heads": 1,
                "intermediate_size": 16,
                "vocab_size": vocab,
                "context_length": 32,
                "rope_theta": 10000.0
            },
            "tensors": tensors
        });
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(dir.path().join("weight.bin"), &bin).unwrap();

        let b = load_bundle(dir.path()).unwrap();
        let loaded_emb = b
            .weight_loaded("model.language_model.embed_tokens.weight")
            .unwrap();
        let loaded_ple = b
            .weight_loaded("model.language_model.embed_tokens_per_layer.weight")
            .unwrap();
        assert_eq!(loaded_emb.data.len(), vocab * hidden);
        assert_eq!(loaded_ple.data.len(), vocab * packed_ple);
        // Quant is lossy; unrotate+gather must match reconstruct (dequant+unrotate).
        for (name, cols, orig, loaded) in [
            (
                "embed",
                hidden,
                emb_orig.as_slice(),
                loaded_emb.data.as_slice(),
            ),
            (
                "ple",
                packed_ple,
                ple_orig.as_slice(),
                loaded_ple.data.as_slice(),
            ),
        ] {
            let q = match b.tensors.get(match name {
                "embed" => "model.language_model.embed_tokens.weight",
                _ => "model.language_model.embed_tokens_per_layer.weight",
            }) {
                Some(TensorData::Codebook(q)) => q,
                _ => panic!("{name}"),
            };
            let mut recon = dequantize(q).unwrap();
            hadamard_blocked_rows(&mut recon, vocab, cols, seed, true).unwrap();
            for (a, e) in loaded.iter().zip(recon.iter()) {
                assert!((a - e).abs() < 1e-5, "{name} {a} vs {e}");
            }
            for tid in [0usize, 2, 9] {
                let row_l = &loaded[tid * cols..(tid + 1) * cols];
                let row_r = &recon[tid * cols..(tid + 1) * cols];
                for (a, e) in row_l.iter().zip(row_r.iter()) {
                    assert!((a - e).abs() < 1e-5, "{name} tid={tid}");
                }
                let orig_row = &orig[tid * cols..(tid + 1) * cols];
                assert!(
                    orig_row.iter().any(|v| v.abs() > 1e-6),
                    "{name} orig row {tid} unexpectedly zero"
                );
            }
        }
    }
}
