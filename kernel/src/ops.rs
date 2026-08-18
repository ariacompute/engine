use crate::{EngineError, SimdMode};

/// C = A @ B^T style? We use row-major: out[m,n] = sum_k a[m,k] * b[k,n]
/// with `a`: [m,k], `b`: [k,n].
pub fn matmul(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_rows: usize,
    b_cols: usize,
    _mode: SimdMode,
) -> Result<Vec<f32>, EngineError> {
    if a_cols != b_rows {
        return Err(EngineError::ShapeMismatch(format!(
            "matmul inner dim {a_cols} != {b_rows}"
        )));
    }
    if a.len() != a_rows * a_cols || b.len() != b_rows * b_cols {
        return Err(EngineError::ShapeMismatch(
            "matmul buffer length does not match shape".into(),
        ));
    }
    let mut out = vec![0.0f32; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut s = 0.0f32;
            for k in 0..a_cols {
                s += a[i * a_cols + k] * b[k * b_cols + j];
            }
            out[i * b_cols + j] = s;
        }
    }
    Ok(out)
}

/// y = x @ W^T where W is [out_features, in_features] row-major (GGUF-style).
pub fn linear(x: &[f32], w: &[f32], out_f: usize, in_f: usize) -> Result<Vec<f32>, EngineError> {
    if !x.len().is_multiple_of(in_f) {
        return Err(EngineError::ShapeMismatch(format!(
            "linear x len {} not divisible by in_f {in_f}",
            x.len()
        )));
    }
    if w.len() != out_f * in_f {
        return Err(EngineError::ShapeMismatch(format!(
            "linear weight length mismatch: got {} want out_f*in_f={out_f}*{in_f}={}",
            w.len(),
            out_f * in_f
        )));
    }
    let batch = x.len() / in_f;
    let mut out = vec![0.0f32; batch * out_f];
    for b in 0..batch {
        for o in 0..out_f {
            let mut s = 0.0f32;
            let wr = &w[o * in_f..(o + 1) * in_f];
            let xr = &x[b * in_f..(b + 1) * in_f];
            for i in 0..in_f {
                s += xr[i] * wr[i];
            }
            out[b * out_f + o] = s;
        }
    }
    Ok(out)
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, EngineError> {
    let n = weight.len();
    if n == 0 || !x.len().is_multiple_of(n) {
        return Err(EngineError::ShapeMismatch(
            "rms_norm: x length must be multiple of weight len".into(),
        ));
    }
    let batch = x.len() / n;
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batch {
        let base = b * n;
        let mut ms = 0.0f32;
        for i in 0..n {
            ms += x[base + i] * x[base + i];
        }
        let scale = (ms / n as f32 + eps).sqrt().recip();
        for i in 0..n {
            out[base + i] = x[base + i] * scale * weight[i];
        }
    }
    Ok(out)
}

pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut o = logits.to_vec();
    softmax_inplace(&mut o);
    o
}

/// Apply RoPE to interleaved q/k pairs for one token (head_dim even).
pub fn rope(x: &mut [f32], head_dim: usize, pos: usize, theta: f32) -> Result<(), EngineError> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(EngineError::ShapeMismatch(
            "rope head_dim must be positive even".into(),
        ));
    }
    if !x.len().is_multiple_of(head_dim) {
        return Err(EngineError::ShapeMismatch(
            "rope x len not divisible by head_dim".into(),
        ));
    }
    let n_heads = x.len() / head_dim;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..(head_dim / 2) {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (c, s) = (angle.cos(), angle.sin());
            let u = x[base + 2 * i];
            let v = x[base + 2 * i + 1];
            x[base + 2 * i] = u * c - v * s;
            x[base + 2 * i + 1] = u * s + v * c;
        }
    }
    Ok(())
}

/// Causal attention for single query step against KV cache.
/// q: [n_heads * head_dim], k_cache/v_cache: [seq, n_kv_heads * head_dim]
pub fn attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, EngineError> {
    if n_heads == 0 || head_dim == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads)
    {
        return Err(EngineError::ShapeMismatch(
            "attention invalid head configuration".into(),
        ));
    }
    let kv_dim = n_kv_heads * head_dim;
    if q.len() != n_heads * head_dim {
        return Err(EngineError::ShapeMismatch("attention q shape".into()));
    }
    if k_cache.len() != v_cache.len() || !k_cache.len().is_multiple_of(kv_dim) {
        return Err(EngineError::ShapeMismatch("attention kv cache shape".into()));
    }
    let seq = k_cache.len() / kv_dim;
    let rep = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv_h = h / rep;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0f32; seq];
        for t in 0..seq {
            let kh = &k_cache[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += qh[i] * kh[i];
            }
            scores[t] = dot * scale;
        }
        softmax_inplace(&mut scores);
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq {
            let vh = &v_cache[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
            for i in 0..head_dim {
                oh[i] += scores[t] * vh[i];
            }
        }
    }
    Ok(out)
}

pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, EngineError> {
    if gate.len() != up.len() {
        return Err(EngineError::ShapeMismatch("swiglu length mismatch".into()));
    }
    Ok(gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let s = 1.0 / (1.0 + (-g).exp());
            s * g * u
        })
        .collect())
}

/// In-place orthogonal FWHT on length = power of two (scale 1/sqrt(n)).
pub fn fwht(x: &mut [f32]) -> Result<(), EngineError> {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(EngineError::ShapeMismatch(
            "fwht length must be power of two".into(),
        ));
    }
    if n == 1 {
        return Ok(());
    }
    let mut h = 1usize;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..(i + h) {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
    Ok(())
}

/// Greedy largest-pow2 tiling of row count (e.g. 10 → [8, 2]).
pub fn pow2_tile_sizes(k: usize) -> Result<Vec<usize>, EngineError> {
    if k == 0 {
        return Err(EngineError::ShapeMismatch(
            "pow2_tile_sizes expects k>=1".into(),
        ));
    }
    let mut sizes = Vec::new();
    let mut rem = k;
    while rem > 0 {
        let mut b = 1usize;
        while (b << 1) <= rem {
            b <<= 1;
        }
        sizes.push(b);
        rem -= b;
    }
    Ok(sizes)
}

/// Portable ±1 signs matching Python `portable_block_signs`.
pub fn portable_block_signs(seed: i64, start: usize, size: usize) -> Vec<f32> {
    let mut signs = vec![0.0f32; size];
    let mut state = (seed as u64) ^ ((start as u64).wrapping_mul(0x9E3779B97F4A7C15));
    for s in signs.iter_mut() {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        *s = if (z & 1) == 0 { 1.0 } else { -1.0 };
    }
    signs
}

/// Apply blocked Hadamard on rows of a row-major `[rows, cols]` matrix.
/// `inverse=false` → per-block `H@S`; `inverse=true` → `S@H`.
pub fn hadamard_blocked_rows(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    seed: Option<i64>,
    inverse: bool,
) -> Result<(), EngineError> {
    if rows == 0 || cols == 0 || data.len() != rows * cols {
        return Err(EngineError::ShapeMismatch(
            "hadamard_blocked_rows shape mismatch".into(),
        ));
    }
    let sizes = pow2_tile_sizes(rows)?;
    let mut start = 0usize;
    for &sz in &sizes {
        let signs = seed.map(|s| portable_block_signs(s, start, sz));
        // Process column-chunks to bound stack/temp if needed; here full width.
        let mut work = vec![0.0f32; sz * cols];
        for r in 0..sz {
            let src = (start + r) * cols;
            work[r * cols..(r + 1) * cols].copy_from_slice(&data[src..src + cols]);
        }
        // Apply along axis 0 for each column: collect column into contiguous buffer.
        let mut colbuf = vec![0.0f32; sz];
        for c in 0..cols {
            for r in 0..sz {
                colbuf[r] = work[r * cols + c];
            }
            if let Some(ref sg) = signs {
                if !inverse {
                    for r in 0..sz {
                        colbuf[r] *= sg[r];
                    }
                    fwht(&mut colbuf)?;
                } else {
                    fwht(&mut colbuf)?;
                    for r in 0..sz {
                        colbuf[r] *= sg[r];
                    }
                }
            } else if sz > 1 {
                fwht(&mut colbuf)?;
            }
            for r in 0..sz {
                work[r * cols + c] = colbuf[r];
            }
        }
        for r in 0..sz {
            let dst = (start + r) * cols;
            data[dst..dst + cols].copy_from_slice(&work[r * cols..(r + 1) * cols]);
        }
        start += sz;
    }
    Ok(())
}

/// Blocked unrotate on a length-`rows` vector (treat as `[rows, 1]`).
pub fn hadamard_blocked_vec(
    x: &mut [f32],
    seed: Option<i64>,
    inverse: bool,
) -> Result<(), EngineError> {
    let rows = x.len();
    hadamard_blocked_rows(x, rows, 1, seed, inverse)
}

/// Codebook lookup dequant (group share): indices [k_work, n], codebook [g, kc].
pub fn dequant_lookup_group(
    indices: &[u8],
    codebook: &[f32],
    num_groups: usize,
    group_size: usize,
    n: usize,
    kc: usize,
    k0: usize,
) -> Result<Vec<f32>, EngineError> {
    let k_work = num_groups * group_size;
    if indices.len() != k_work * n {
        return Err(EngineError::ShapeMismatch(
            "dequant indices length mismatch".into(),
        ));
    }
    if codebook.len() != num_groups * kc {
        return Err(EngineError::Quant("dequant codebook length mismatch".into()));
    }
    let mut out = vec![0.0f32; k_work * n];
    for g in 0..num_groups {
        let cb = &codebook[g * kc..(g + 1) * kc];
        for r in 0..group_size {
            let row = g * group_size + r;
            for j in 0..n {
                let idx = indices[row * n + j] as usize;
                if idx >= kc {
                    return Err(EngineError::Quant(format!("index {idx} >= kc {kc}")));
                }
                out[row * n + j] = cb[idx];
            }
        }
    }
    out.truncate(k0 * n);
    Ok(out)
}

/// Blocked matmul used as Neon / SIMD-friendly path (portable; aarch64 may specialize later).
pub fn matmul_blocked(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_rows: usize,
    b_cols: usize,
    block: usize,
) -> Result<Vec<f32>, EngineError> {
    if a_cols != b_rows {
        return Err(EngineError::ShapeMismatch(format!(
            "matmul inner dim {a_cols} != {b_rows}"
        )));
    }
    if a.len() != a_rows * a_cols || b.len() != b_rows * b_cols {
        return Err(EngineError::ShapeMismatch(
            "matmul buffer length does not match shape".into(),
        ));
    }
    let block = block.max(1);
    let mut out = vec![0.0f32; a_rows * b_cols];
    for i0 in (0..a_rows).step_by(block) {
        for j0 in (0..b_cols).step_by(block) {
            for k0 in (0..a_cols).step_by(block) {
                let i_max = (i0 + block).min(a_rows);
                let j_max = (j0 + block).min(b_cols);
                let k_max = (k0 + block).min(a_cols);
                for i in i0..i_max {
                    for j in j0..j_max {
                        let mut s = out[i * b_cols + j];
                        for k in k0..k_max {
                            s += a[i * a_cols + k] * b[k * b_cols + j];
                        }
                        out[i * b_cols + j] = s;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Dispatch scalar vs Neon (blocked) paths. Neon is available on all targets for parity tests;
/// on `aarch64` this is the production SIMD entry (intrinsics may replace the body later).
pub fn matmul_dispatch(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_rows: usize,
    b_cols: usize,
    mode: SimdMode,
) -> Result<Vec<f32>, EngineError> {
    match mode {
        SimdMode::Scalar => matmul(a, a_rows, a_cols, b, b_rows, b_cols, SimdMode::Scalar),
        SimdMode::Neon => {
            #[cfg(target_arch = "aarch64")]
            {
                // Prefer blocked layout matching NEON tile width; values must match scalar.
                matmul_blocked(a, a_rows, a_cols, b, b_rows, b_cols, 8)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                // Emulated Neon path for CI parity on x86_64.
                matmul_blocked(a, a_rows, a_cols, b, b_rows, b_cols, 8)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_ok() {
        let a = [1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let c = matmul(&a, 2, 2, &b, 2, 2, SimdMode::Scalar).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_shape_err() {
        let err = matmul(&[1.0], 1, 1, &[1.0, 2.0], 2, 1, SimdMode::Scalar).unwrap_err();
        assert!(matches!(err, EngineError::ShapeMismatch(_)));
    }

    #[test]
    fn rms_and_softmax() {
        let w = [1.0f32, 1.0];
        let y = rms_norm(&[3.0, 4.0], &w, 1e-6).unwrap();
        // rms = sqrt((9+16)/2) = sqrt(12.5); y0 = 3/rms
        let rms = (12.5f32).sqrt();
        assert!((y[0] - 3.0 / rms).abs() < 1e-4);
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fwht_roundtrip_ish() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let orig = x;
        fwht(&mut x).unwrap();
        fwht(&mut x).unwrap();
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn pow2_tiles() {
        assert_eq!(pow2_tile_sizes(10).unwrap(), vec![8, 2]);
        assert_eq!(pow2_tile_sizes(3072).unwrap(), vec![2048, 1024]);
        assert_eq!(pow2_tile_sizes(64).unwrap(), vec![64]);
        assert_eq!(pow2_tile_sizes(1).unwrap(), vec![1]);
        assert_eq!(pow2_tile_sizes(151936).unwrap()[0], 131072);
        assert!(matches!(
            pow2_tile_sizes(0),
            Err(EngineError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn blocked_roundtrip_non_pow2() {
        let rows = 10usize;
        let cols = 3usize;
        let mut w: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let orig = w.clone();
        hadamard_blocked_rows(&mut w, rows, cols, Some(7), false).unwrap();
        // Second forward is not the inverse.
        let mut twice = w.clone();
        hadamard_blocked_rows(&mut twice, rows, cols, Some(7), false).unwrap();
        let mut err_wrong = 0.0f32;
        for (a, b) in twice.iter().zip(orig.iter()) {
            err_wrong += (a - b).abs();
        }
        assert!(err_wrong > 1.0, "second forward unexpectedly near identity");
        hadamard_blocked_rows(&mut w, rows, cols, Some(7), true).unwrap();
        for (a, b) in w.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn blocked_roundtrip_unsigned_and_pow2() {
        let rows = 16usize;
        let cols = 5usize;
        let mut w: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.03).collect();
        let orig = w.clone();
        hadamard_blocked_rows(&mut w, rows, cols, None, false).unwrap();
        hadamard_blocked_rows(&mut w, rows, cols, None, true).unwrap();
        for (a, b) in w.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn blocked_matches_python_golden() {
        // W[i,j] = i*3+j)*0.1 - 0.5; seed=7 — from model.common.hadamard
        let rows = 10usize;
        let cols = 3usize;
        let mut w: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        hadamard_blocked_rows(&mut w, rows, cols, Some(7), false).unwrap();
        // Clippy-safe f32 literals (Python float32 rounded to representable precision).
        let golden: [f32; 30] = [
            0.919239, 0.989949, 1.060_66, 0.919239, 0.989950, 1.060_66, -0.919239, -0.989949,
            -1.060_66, 0.777817, std::f32::consts::FRAC_1_SQRT_2, 0.636396, -0.919239, -0.989949,
            -1.060_66, -0.070711, -0.141421, -0.212132, 1.343503, std::f32::consts::SQRT_2,
            1.484924, -0.636396, -0.848528, -1.060_66, -2.899138, -3.040559, -3.181_98, 0.212132,
            0.212132, 0.212132,
        ];
        assert_eq!(w.len(), golden.len());
        for (a, b) in w.iter().zip(golden.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        assert_eq!(
            portable_block_signs(7, 0, 8),
            vec![-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]
        );
        assert_eq!(portable_block_signs(7, 8, 2), vec![-1.0, -1.0]);
    }

    #[test]
    fn portable_signs_stable() {
        let a = portable_block_signs(0, 0, 8);
        let b = portable_block_signs(0, 0, 8);
        assert_eq!(a, b);
        // Golden from model.common.hadamard.portable_block_signs(0, 0, 8)
        let golden = [-1.0f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        assert_eq!(a, golden);
    }

    #[test]
    fn hadamard_blocked_shape_errors() {
        let mut w = [1.0f32, 2.0];
        assert!(matches!(
            hadamard_blocked_rows(&mut w, 2, 2, None, false),
            Err(EngineError::ShapeMismatch(_))
        ));
        assert!(matches!(
            hadamard_blocked_rows(&mut [], 0, 1, None, false),
            Err(EngineError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn hadamard_blocked_vec_roundtrip() {
        let mut x: Vec<f32> = (0..10).map(|i| (i as f32) * 0.2 - 1.0).collect();
        let orig = x.clone();
        hadamard_blocked_vec(&mut x, Some(11), false).unwrap();
        hadamard_blocked_vec(&mut x, Some(11), true).unwrap();
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn dequant_group() {
        // 1 group, gs=2, n=2, kc=2
        let indices = [0u8, 1, 1, 0];
        let codebook = [10.0f32, 20.0];
        let out = dequant_lookup_group(&indices, &codebook, 1, 2, 2, 2, 2).unwrap();
        assert_eq!(out, vec![10.0, 20.0, 20.0, 10.0]);
    }

    #[test]
    fn neon_scalar_matmul_parity() {
        let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 - 0.5).collect();
        let s = matmul_dispatch(&a, 8, 8, &b, 8, 8, SimdMode::Scalar).unwrap();
        let n = matmul_dispatch(&a, 8, 8, &b, 8, 8, SimdMode::Neon).unwrap();
        assert_eq!(s.len(), n.len());
        for (x, y) in s.iter().zip(n.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn shape_errors() {
        assert!(matches!(
            rms_norm(&[1.0, 2.0], &[1.0, 1.0, 1.0], 1e-6),
            Err(EngineError::ShapeMismatch(_))
        ));
        assert!(matches!(
            linear(&[1.0], &[1.0, 2.0], 1, 1),
            Err(EngineError::ShapeMismatch(_))
        ));
        assert!(matches!(
            rope(&mut [1.0, 2.0, 3.0], 3, 0, 10000.0),
            Err(EngineError::ShapeMismatch(_))
        ));
        assert!(matches!(
            attention(&[1.0], &[1.0], &[1.0], 1, 1, 2),
            Err(EngineError::ShapeMismatch(_))
        ));
        assert!(matches!(
            fwht(&mut [1.0, 2.0, 3.0]),
            Err(EngineError::ShapeMismatch(_))
        ));
    }
}
