//! Synthetic tiny Aria bundles / QuantTensors for contract tests (mirrors model tiny layout).

use crate::bundle::QuantTensor;
use crate::pack::pack_indices;
use aria_kernel::EngineError;
use half::f16;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

fn write_f16_slice(buf: &mut Vec<u8>, data: &[f32]) {
    for &v in data {
        buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
}

fn blocked_hadamard_meta(k: usize, seed: i64) -> Value {
    let mut blocks = Vec::new();
    let mut rem = k;
    let mut start = 0usize;
    while rem > 0 {
        let mut b = 1usize;
        while (b << 1) <= rem {
            b <<= 1;
        }
        blocks.push(json!({ "start": start, "size": b }));
        start += b;
        rem -= b;
    }
    json!({
        "applied": true,
        "axis": 0,
        "seed": seed,
        "mode": "blocked",
        "blocks": blocks,
        "row_pad": 0
    })
}

/// Relative RMSE: sqrt(mean((a-b)^2)) / (rms(a)+eps). Matches model `test_quant._rel_rmse`.
pub fn rel_rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut mse = 0.0f64;
    let mut ms = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = f64::from(x) - f64::from(y);
        mse += d * d;
        ms += f64::from(x) * f64::from(x);
    }
    let n = a.len() as f64;
    let rmse = (mse / n).sqrt();
    let rms = (ms / n).sqrt() + 1e-12;
    (rmse / rms) as f32
}

fn fill_linspace_codebook(codebook: &mut [f32], kc: usize, mn: f32, mx: f32) {
    if kc == 1 {
        codebook[0] = mn;
        return;
    }
    let denom = (kc as f32) - 1.0;
    for (c, slot) in codebook.iter_mut().enumerate().take(kc) {
        *slot = mn + (mx - mn) * (c as f32) / denom;
    }
}

/// Linspace group-share quant (test stand-in for Lloyd-Max).
pub fn make_group_quant_tensor(
    w: &[f32],
    k: usize,
    n: usize,
    group_size: usize,
    bits: u8,
) -> QuantTensor {
    assert_eq!(w.len(), k * n);
    assert!((1..=4).contains(&bits) || bits == 8);
    let kc = 1usize << bits;
    let num_groups = k.div_ceil(group_size);
    let k_work = num_groups * group_size;
    let mut codebook = vec![0.0f32; num_groups * kc];
    let mut indices = vec![0u8; k_work * n];
    for g in 0..num_groups {
        let mut vals = Vec::new();
        for r in 0..group_size {
            let row = g * group_size + r;
            if row < k {
                for j in 0..n {
                    vals.push(w[row * n + j]);
                }
            }
        }
        let (mn, mx) = vals
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                (a.min(v), b.max(v))
            });
        let mn = if mn.is_finite() { mn } else { 0.0 };
        let mx = if mx.is_finite() { mx } else { 0.0 };
        fill_linspace_codebook(&mut codebook[g * kc..(g + 1) * kc], kc, mn, mx);
        for r in 0..group_size {
            let row = g * group_size + r;
            for j in 0..n {
                let v = if row < k { w[row * n + j] } else { 0.0 };
                let mut best = 0u8;
                let mut best_d = f32::INFINITY;
                for c in 0..kc {
                    let d = (v - codebook[g * kc + c]).abs();
                    if d < best_d {
                        best_d = d;
                        best = c as u8;
                    }
                }
                indices[row * n + j] = best;
            }
        }
    }
    let packed = pack_indices(&indices, bits).expect("pack");
    QuantTensor {
        bits,
        group_size,
        shape: (k, n),
        row_pad: 0,
        codebook_share: "group".into(),
        packed_indices: packed,
        codebook,
        codebook_shape: vec![num_groups, kc],
        hadamard: blocked_hadamard_meta(k, 0),
    }
}

/// Linspace channel-share quant.
pub fn make_channel_quant_tensor(
    w: &[f32],
    k: usize,
    n: usize,
    group_size: usize,
    bits: u8,
) -> QuantTensor {
    assert_eq!(w.len(), k * n);
    let kc = 1usize << bits;
    let num_groups = k.div_ceil(group_size);
    let k_work = num_groups * group_size;
    let mut codebook = vec![0.0f32; num_groups * n * kc];
    let mut indices = vec![0u8; k_work * n];
    for g in 0..num_groups {
        for j in 0..n {
            let mut vals = Vec::new();
            for r in 0..group_size {
                let row = g * group_size + r;
                if row < k {
                    vals.push(w[row * n + j]);
                }
            }
            let (mn, mx) = vals
                .iter()
                .copied()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                    (a.min(v), b.max(v))
                });
            let mn = if mn.is_finite() { mn } else { 0.0 };
            let mx = if mx.is_finite() { mx } else { 0.0 };
            let base = (g * n + j) * kc;
            fill_linspace_codebook(&mut codebook[base..base + kc], kc, mn, mx);
            for r in 0..group_size {
                let row = g * group_size + r;
                let v = if row < k { w[row * n + j] } else { 0.0 };
                let mut best = 0u8;
                let mut best_d = f32::INFINITY;
                for c in 0..kc {
                    let d = (v - codebook[base + c]).abs();
                    if d < best_d {
                        best_d = d;
                        best = c as u8;
                    }
                }
                indices[row * n + j] = best;
            }
        }
    }
    let packed = pack_indices(&indices, bits).expect("pack");
    QuantTensor {
        bits,
        group_size,
        shape: (k, n),
        row_pad: 0,
        codebook_share: "channel".into(),
        packed_indices: packed,
        codebook,
        codebook_shape: vec![num_groups, n, kc],
        hadamard: blocked_hadamard_meta(k, 0),
    }
}

fn quantize_group_q4(
    w: &[f32],
    k: usize,
    n: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, usize) {
    let t = make_group_quant_tensor(w, k, n, group_size, 4);
    (t.packed_indices, t.codebook, t.codebook_shape[0])
}

fn dequant_ref(
    indices_packed: &[u8],
    codebook: &[f32],
    num_groups: usize,
    gs: usize,
    n: usize,
    k0: usize,
) -> Vec<f32> {
    use crate::pack::unpack_indices;
    let k_work = num_groups * gs;
    let idx = unpack_indices(indices_packed, k_work * n, 4).unwrap();
    let kc = 16;
    let mut out = vec![0.0f32; k_work * n];
    for g in 0..num_groups {
        for r in 0..gs {
            let row = g * gs + r;
            for j in 0..n {
                let id = idx[row * n + j] as usize;
                out[row * n + j] = codebook[g * kc + id];
            }
        }
    }
    out.truncate(k0 * n);
    out
}

struct Writer {
    bin: Vec<u8>,
    tensors: BTreeMap<String, Value>,
    gs: usize,
}

impl Writer {
    fn add_raw_1d(&mut self, name: &str, data: &[f32]) {
        let start = self.bin.len();
        write_f16_slice(&mut self.bin, data);
        let len = self.bin.len() - start;
        self.tensors.insert(
            name.to_string(),
            json!({
                "kind": "raw",
                "dtype": "f16",
                "shape": [data.len()],
                "offsets": { "data": [start, len] }
            }),
        );
    }

    fn add_codebook_2d(&mut self, name: &str, w: &[f32], k: usize, n: usize) -> Vec<f32> {
        // Match Python quantize: rotate then pack; load path unrotates (reconstruct_weight).
        let mut w_rot = w.to_vec();
        aria_kernel::hadamard_blocked_rows(&mut w_rot, k, n, Some(0), false)
            .expect("fixture hadamard rotate");
        let (packed, codebook, num_groups) = quantize_group_q4(&w_rot, k, n, self.gs);
        let pi_start = self.bin.len();
        self.bin.extend_from_slice(&packed);
        let pi_len = self.bin.len() - pi_start;
        let cb_start = self.bin.len();
        write_f16_slice(&mut self.bin, &codebook);
        let cb_len = self.bin.len() - cb_start;
        let mut recon = dequant_ref(&packed, &codebook, num_groups, self.gs, n, k);
        aria_kernel::hadamard_blocked_rows(&mut recon, k, n, Some(0), true)
            .expect("fixture hadamard unrotate");
        self.tensors.insert(
            name.to_string(),
            json!({
                "kind": "codebook",
                "bits": 4,
                "group_size": self.gs,
                "shape": [k, n],
                "row_pad": 0,
                "codebook_share": "group",
                "hadamard": blocked_hadamard_meta(k, 0),
                "offsets": {
                    "packed_indices": [pi_start, pi_len],
                    "codebook": [cb_start, cb_len]
                }
            }),
        );
        recon
    }
}

/// Write tiny q4 bundle. Returns (RMSE vs original attn_q, path).
pub fn write_tiny_q4_bundle(out: &Path) -> Result<(f32, std::path::PathBuf), EngineError> {
    fs::create_dir_all(out)?;
    let vocab = 128usize;
    let hidden = 64usize;
    let layers = 2usize;
    let inter = 128usize;
    let heads = 4usize;
    let gs = 32usize;

    let mut rng = 1u64;
    let mut randn = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((rng >> 33) as f32) / (u32::MAX as f32);
        (u - 0.5) * 0.04
    };

    let mut w = Writer {
        bin: Vec::new(),
        tensors: BTreeMap::new(),
        gs,
    };

    let mut emb = vec![0.0f32; vocab * hidden];
    for v in &mut emb {
        *v = randn();
    }
    w.add_codebook_2d("token_embd.weight", &emb, vocab, hidden);

    let mut attn_q0 = vec![0.0f32; hidden * hidden];
    for v in &mut attn_q0 {
        *v = randn();
    }
    let recon_q = w.add_codebook_2d("blk.0.attn_q.weight", &attn_q0, hidden, hidden);
    let mut mse = 0.0f32;
    for i in 0..attn_q0.len() {
        let d = attn_q0[i] - recon_q[i];
        mse += d * d;
    }
    let rmse = (mse / attn_q0.len() as f32).sqrt();

    for layer in 0..layers {
        if layer != 0 {
            let mut mat = vec![0.0f32; hidden * hidden];
            for v in &mut mat {
                *v = randn();
            }
            w.add_codebook_2d(&format!("blk.{layer}.attn_q.weight"), &mat, hidden, hidden);
        }
        for part in ["attn_k", "attn_v", "attn_output"] {
            let mut mat = vec![0.0f32; hidden * hidden];
            for v in &mut mat {
                *v = randn();
            }
            w.add_codebook_2d(&format!("blk.{layer}.{part}.weight"), &mat, hidden, hidden);
        }
        for (part, rows, cols) in [
            ("ffn_gate", inter, hidden),
            ("ffn_up", inter, hidden),
            ("ffn_down", hidden, inter),
        ] {
            let mut mat = vec![0.0f32; rows * cols];
            for v in &mut mat {
                *v = randn();
            }
            w.add_codebook_2d(&format!("blk.{layer}.{part}.weight"), &mat, rows, cols);
        }
        let mut n1 = vec![0.0f32; hidden];
        let mut n2 = vec![0.0f32; hidden];
        for i in 0..hidden {
            n1[i] = 1.0 + randn();
            n2[i] = 1.0 + randn();
        }
        w.add_raw_1d(&format!("blk.{layer}.attn_norm.weight"), &n1);
        w.add_raw_1d(&format!("blk.{layer}.ffn_norm.weight"), &n2);
    }

    let mut on = vec![1.0f32; hidden];
    for v in &mut on {
        *v += randn();
    }
    w.add_raw_1d("output_norm.weight", &on);

    let mut ow = vec![0.0f32; vocab * hidden];
    for v in &mut ow {
        *v = randn();
    }
    w.add_codebook_2d("output.weight", &ow, vocab, hidden);

    fs::write(out.join("weight.bin"), &w.bin)?;
    let config = json!({
        "format": "aria-quant-bundle",
        "format_version": 2,
        "quantization": "q4",
        "group_size_default": gs,
        "hadamard_seed": 0,
        "model": {
            "hidden_size": hidden,
            "num_layers": layers,
            "num_attention_heads": heads,
            "num_kv_heads": heads,
            "intermediate_size": inter,
            "vocab_size": vocab,
            "context_length": 64,
            "rope_theta": 10000.0
        },
        "tensors": w.tensors
    });
    fs::write(
        out.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )?;
    Ok((rmse, out.to_path_buf()))
}
