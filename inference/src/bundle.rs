use crate::pack::unpack_indices;
use aria_kernel::{dequant_lookup_group, EngineError};
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
    if bytes.len() % 2 != 0 {
        return Err(EngineError::Format("f16 byte length odd".into()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        let h = f16::from_le_bytes([c[0], c[1]]);
        out.push(h.to_f32());
    }
    Ok(out)
}

fn f32_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, EngineError> {
    if bytes.len() % 4 != 0 {
        return Err(EngineError::Format("f32 byte length not aligned".into()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
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
    let cfg: BundleConfig = serde_json::from_str(&cfg_text)
        .map_err(|e| EngineError::Format(e.to_string()))?;
    if cfg.format != BUNDLE_FORMAT {
        return Err(EngineError::Format(format!(
            "unsupported format {:?}",
            cfg.format
        )));
    }
    if cfg.format_version != 1 {
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
                    .ok_or_else(|| EngineError::Quant("bits".into()))? as u8;
                if !matches!(bits, 1 | 2 | 3 | 4 | 8) {
                    return Err(EngineError::Quant(format!("unsupported bits {bits}")));
                }
                let group_size = meta
                    .get("group_size")
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
                let row_pad = meta
                    .get("row_pad")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
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
                    if codebook.len() % kc != 0 {
                        return Err(EngineError::ShapeMismatch(
                            "bad group codebook size".into(),
                        ));
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
                tensors.insert(
                    name.clone(),
                    TensorData::Raw { dtype, shape, data },
                );
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
        match self.tensors.get(name) {
            Some(TensorData::Codebook(q)) => dequantize(q),
            Some(TensorData::Raw { data, .. }) => Ok(data.clone()),
            None => Err(EngineError::Format(format!("missing tensor {name}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::write_tiny_q4_bundle;

    #[test]
    fn load_and_dequant() {
        let dir = tempfile::tempdir().unwrap();
        let (rmse, _) = write_tiny_q4_bundle(dir.path()).unwrap();
        assert!(rmse < 0.5, "rmse {rmse}");
        let b = load_bundle(dir.path()).unwrap();
        assert_eq!(b.model.hidden_size, 64);
        let w = b.weight_f32("blk.0.attn_q.weight").unwrap();
        assert_eq!(w.len(), 64 * 64);
    }

    #[test]
    fn bad_format() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"format":"nope"}"#).unwrap();
        std::fs::write(dir.path().join("weight.bin"), b"").unwrap();
        let err = load_bundle(dir.path()).unwrap_err();
        assert!(matches!(err, EngineError::Format(_)));
    }
}
