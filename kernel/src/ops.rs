use crate::{EngineError, SimdMode};
use rayon::prelude::*;

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

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0.0f32;
        for i in 0..a.len() {
            s += a[i] * b[i];
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut s = tmp.iter().sum::<f32>();
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(0.0);
    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, va, vb);
        i += 4;
    }
    let mut tmp = [0.0f32; 4];
    vst1q_f32(tmp.as_mut_ptr(), acc);
    let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Multi-threaded `linear` (AVX2/FMA or NEON dots). Numerically close to [`linear`].
pub fn linear_cpu(x: &[f32], w: &[f32], out_f: usize, in_f: usize) -> Result<Vec<f32>, EngineError> {
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
    if batch == 0 || out_f == 0 {
        return Ok(out);
    }
    // Parallelize over every (batch, out_feature) pair so decode lm_head
    // (batch=1, out_f≈vocab) and prefill FFN both scale across cores.
    out.par_iter_mut().enumerate().for_each(|(idx, slot)| {
        let b = idx / out_f;
        let o = idx % out_f;
        *slot = dot_f32(
            &w[o * in_f..(o + 1) * in_f],
            &x[b * in_f..(b + 1) * in_f],
        );
    });
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
    if n_heads == 0 || head_dim == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(EngineError::ShapeMismatch(
            "attention invalid head configuration".into(),
        ));
    }
    let kv_dim = n_kv_heads * head_dim;
    if q.len() != n_heads * head_dim {
        return Err(EngineError::ShapeMismatch("attention q shape".into()));
    }
    if k_cache.len() != v_cache.len() || !k_cache.len().is_multiple_of(kv_dim) {
        return Err(EngineError::ShapeMismatch(
            "attention kv cache shape".into(),
        ));
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

/// Causal attention for a batch of queries `[seq_q, n_heads*head_dim]`.
/// When `seq_q == seq_kv`, query `t` attends to keys `0..=t` (prefill).
/// When `seq_q == 1`, equivalent to [`attention`] (decode).
pub fn attention_causal(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, EngineError> {
    let q_dim = n_heads * head_dim;
    if q_dim == 0 || !q.len().is_multiple_of(q_dim) {
        return Err(EngineError::ShapeMismatch("attention_causal q shape".into()));
    }
    let seq_q = q.len() / q_dim;
    if seq_q == 1 {
        return attention(q, k_cache, v_cache, n_heads, n_kv_heads, head_dim);
    }
    let kv_dim = n_kv_heads * head_dim;
    let seq_kv = k_cache.len() / kv_dim;
    let causal = seq_q == seq_kv;
    let mut out = vec![0.0f32; q.len()];
    for tq in 0..seq_q {
        let q_tok = &q[tq * q_dim..(tq + 1) * q_dim];
        let k_end = if causal { tq + 1 } else { seq_kv };
        let attn = attention(
            q_tok,
            &k_cache[..k_end * kv_dim],
            &v_cache[..k_end * kv_dim],
            n_heads,
            n_kv_heads,
            head_dim,
        )?;
        out[tq * q_dim..(tq + 1) * q_dim].copy_from_slice(&attn);
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

/// Causal depthwise short-conv one-token step (LFM2).
///
/// `w` layout: `[hidden * kernel]`, channel-major, `w[c*kernel + 0]` taps the oldest
/// sample. `state` is `[hidden * (kernel-1)]` (oldest→newest); updated in place.
pub fn short_conv_step(
    x: &[f32],
    w: &[f32],
    state: &mut [f32],
    hidden: usize,
    kernel: usize,
) -> Result<Vec<f32>, EngineError> {
    if hidden == 0 || kernel == 0 {
        return Err(EngineError::ShapeMismatch(
            "short_conv_step: hidden and kernel must be > 0".into(),
        ));
    }
    if x.len() != hidden {
        return Err(EngineError::ShapeMismatch(
            "short_conv_step: x len != hidden".into(),
        ));
    }
    if w.len() != hidden * kernel {
        return Err(EngineError::ShapeMismatch(
            "short_conv_step: weight len != hidden*kernel".into(),
        ));
    }
    let hist = kernel.saturating_sub(1);
    if state.len() != hidden * hist {
        return Err(EngineError::ShapeMismatch(
            "short_conv_step: state len != hidden*(kernel-1)".into(),
        ));
    }
    let mut out = vec![0.0f32; hidden];
    for c in 0..hidden {
        let mut acc = 0.0f32;
        let wbase = c * kernel;
        let sbase = c * hist;
        for k in 0..hist {
            acc += w[wbase + k] * state[sbase + k];
        }
        acc += w[wbase + hist] * x[c];
        out[c] = acc;
        if hist > 0 {
            for k in 0..(hist - 1) {
                state[sbase + k] = state[sbase + k + 1];
            }
            state[sbase + hist - 1] = x[c];
        }
    }
    Ok(out)
}

/// Softmax (or sigmoid) top-k MoE routing. Returns (expert_ids, normalized weights).
pub fn moe_topk_route(
    logits: &[f32],
    top_k: usize,
    use_sigmoid: bool,
) -> Result<(Vec<usize>, Vec<f32>), EngineError> {
    let n = logits.len();
    if n == 0 || top_k == 0 {
        return Err(EngineError::InvalidParam(
            "moe_topk_route: num_experts and top_k must be > 0".into(),
        ));
    }
    let k = top_k.min(n);
    let scores: Vec<f32> = if use_sigmoid {
        logits.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
    } else {
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
        let s: f32 = exps.iter().sum();
        if s > 0.0 {
            for e in &mut exps {
                *e /= s;
            }
        }
        exps
    };
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    let mut weights: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    Ok((idx, weights))
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Elementwise SiLU (DeltaNet conv activation).
pub fn silu_vec(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// Numerically stable softplus.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn l2_normalize_inplace(x: &mut [f32]) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += *v * *v;
    }
    let n = (ss + 1e-6).sqrt();
    if n > 0.0 {
        for v in x.iter_mut() {
            *v /= n;
        }
    }
}

/// Bundled args for [`gated_delta_step`] (avoids clippy `too_many_arguments`).
pub struct GatedDeltaStep<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub g: &'a [f32],
    pub beta: &'a [f32],
    pub state: &'a mut [f32],
    pub n_heads: usize,
    pub dk: usize,
    pub dv: usize,
}

/// One-token Gated DeltaNet recurrence (Qwen3.5 / Bonsai linear attention).
///
/// `state` is `[n_heads * dk * dv]` (per-head `S[dk, dv]`). `q`/`k` are
/// `[n_heads * dk]`, `v` `[n_heads * dv]`, `g`/`beta` `[n_heads]`.
pub fn gated_delta_step(p: GatedDeltaStep<'_>) -> Result<Vec<f32>, EngineError> {
    let GatedDeltaStep {
        q,
        k,
        v,
        g,
        beta,
        state,
        n_heads,
        dk,
        dv,
    } = p;
    if n_heads == 0 || dk == 0 || dv == 0 {
        return Err(EngineError::ShapeMismatch(
            "gated_delta_step: heads/dk/dv must be > 0".into(),
        ));
    }
    if q.len() != n_heads * dk
        || k.len() != n_heads * dk
        || v.len() != n_heads * dv
        || g.len() != n_heads
        || beta.len() != n_heads
        || state.len() != n_heads * dk * dv
    {
        return Err(EngineError::ShapeMismatch(
            "gated_delta_step: q/k/v/g/beta/state shape mismatch".into(),
        ));
    }
    let mut qq = q.to_vec();
    let mut kk = k.to_vec();
    for h in 0..n_heads {
        l2_normalize_inplace(&mut qq[h * dk..(h + 1) * dk]);
        l2_normalize_inplace(&mut kk[h * dk..(h + 1) * dk]);
    }
    let scale = (dk as f32).sqrt().recip();
    let mut out = vec![0.0f32; n_heads * dv];
    for h in 0..n_heads {
        let sbase = h * dk * dv;
        let gh = g[h];
        for i in 0..dk * dv {
            state[sbase + i] *= gh;
        }
        let mut kv_mem = vec![0.0f32; dv];
        for i in 0..dk {
            let kv = kk[h * dk + i];
            for j in 0..dv {
                kv_mem[j] += state[sbase + i * dv + j] * kv;
            }
        }
        for j in 0..dv {
            let delta = (v[h * dv + j] - kv_mem[j]) * beta[h];
            for i in 0..dk {
                state[sbase + i * dv + j] += kk[h * dk + i] * delta;
            }
        }
        for j in 0..dv {
            let mut o = 0.0f32;
            for i in 0..dk {
                o += state[sbase + i * dv + j] * qq[h * dk + i];
            }
            out[h * dv + j] = o * scale;
        }
    }
    Ok(out)
}

/// GeGLU: gelu(gate) * up (Gemma `gelu_pytorch_tanh` approximates with tanh form).
pub fn geglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, EngineError> {
    if gate.len() != up.len() {
        return Err(EngineError::ShapeMismatch("geglu length mismatch".into()));
    }
    Ok(gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| gelu_pytorch_tanh(*g) * u)
        .collect())
}

/// Match transformers `gelu_pytorch_tanh` (used by Gemma GeGLU).
pub fn gelu_pytorch_tanh(x: f32) -> f32 {
    // Match transformers gelu_pytorch_tanh.
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Gemma-style RMSNorm: x * rrms * (1 + weight).
pub fn rms_norm_gemma(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, EngineError> {
    let n = weight.len();
    if n == 0 || !x.len().is_multiple_of(n) {
        return Err(EngineError::ShapeMismatch(
            "rms_norm_gemma: x length must be multiple of weight len".into(),
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
        let rrms = (ms / n as f32 + eps).sqrt().recip();
        for i in 0..n {
            out[base + i] = x[base + i] * rrms * (1.0 + weight[i]);
        }
    }
    Ok(out)
}

/// HF Llama/Qwen/Gemma RoPE: rotate half of the head dims as a contiguous block.
pub fn rope_half(
    x: &mut [f32],
    head_dim: usize,
    pos: usize,
    theta: f32,
) -> Result<(), EngineError> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(EngineError::ShapeMismatch(
            "rope_half head_dim must be positive even".into(),
        ));
    }
    if !x.len().is_multiple_of(head_dim) {
        return Err(EngineError::ShapeMismatch(
            "rope_half x len not divisible by head_dim".into(),
        ));
    }
    let half = head_dim / 2;
    let n_heads = x.len() / head_dim;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (c, s) = (angle.cos(), angle.sin());
            let u = x[base + i];
            let v = x[base + i + half];
            x[base + i] = u * c - v * s;
            x[base + i + half] = u * s + v * c;
        }
    }
    Ok(())
}

/// y = W_rot @ x followed by blocked unrotate on each out_f row (HDM fused path).
///
/// Equivalent to `linear(x, unrotate(W_rot))` for dense GEMM. **Not** valid for
/// embedding row gather: axis-0 Hadamard mixes vocab rows, so `W_rot[token]` is
/// not the token vector. Session reconstructs the full matrix at load instead.
pub fn hdm_linear(
    x: &[f32],
    w_rot: &[f32],
    out_f: usize,
    in_f: usize,
    hadamard_seed: Option<i64>,
) -> Result<Vec<f32>, EngineError> {
    let mut y = linear(x, w_rot, out_f, in_f)?;
    if out_f == 0 || !y.len().is_multiple_of(out_f) {
        return Err(EngineError::ShapeMismatch(
            "hdm_linear output not divisible by out_f".into(),
        ));
    }
    let batch = y.len() / out_f;
    for b in 0..batch {
        let sl = b * out_f..(b + 1) * out_f;
        hadamard_blocked_vec(&mut y[sl], hadamard_seed, true)?;
    }
    Ok(y)
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
        let col_out: Result<Vec<Vec<f32>>, EngineError> = (0..cols)
            .into_par_iter()
            .map(|c| {
                let mut colbuf = vec![0.0f32; sz];
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
                Ok(colbuf)
            })
            .collect();
        let col_out = col_out?;
        for c in 0..cols {
            for r in 0..sz {
                work[r * cols + c] = col_out[c][r];
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
        return Err(EngineError::Quant(
            "dequant codebook length mismatch".into(),
        ));
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
        SimdMode::Neon | SimdMode::Avx2 => matmul_blocked(a, a_rows, a_cols, b, b_rows, b_cols, 8),
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
        let mut w: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 0.5).collect();
        hadamard_blocked_rows(&mut w, rows, cols, Some(7), false).unwrap();
        // Clippy-safe f32 literals (Python float32 rounded to representable precision).
        let golden: [f32; 30] = [
            0.919239,
            0.989949,
            1.060_66,
            0.919239,
            0.989950,
            1.060_66,
            -0.919239,
            -0.989949,
            -1.060_66,
            0.777817,
            std::f32::consts::FRAC_1_SQRT_2,
            0.636396,
            -0.919239,
            -0.989949,
            -1.060_66,
            -0.070711,
            -0.141421,
            -0.212132,
            1.343503,
            std::f32::consts::SQRT_2,
            1.484924,
            -0.636396,
            -0.848528,
            -1.060_66,
            -2.899138,
            -3.040559,
            -3.181_98,
            0.212132,
            0.212132,
            0.212132,
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
    fn short_conv_step_and_moe_route() {
        let hidden = 2usize;
        let kernel = 3usize;
        // Channel-major: each channel w=[0,0,1] taps newest only.
        let w = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut state = vec![0.0f32; hidden * (kernel - 1)];
        let x = [3.0f32, 5.0];
        let y = short_conv_step(&x, &w, &mut state, hidden, kernel).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-5);
        assert!((y[1] - 5.0).abs() < 1e-5);
        // After one step hist=[0, x]; w_old=[1,0,0] taps oldest → 0.
        let w_old = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let y2 = short_conv_step(&[1.0, 2.0], &w_old, &mut state, hidden, kernel).unwrap();
        assert!((y2[0] - 0.0).abs() < 1e-5);
        assert!((y2[1] - 0.0).abs() < 1e-5);

        let (ids, ws) = moe_topk_route(&[0.1, 2.0, 0.5, -1.0], 2, false).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 1);
        assert!((ws.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let (ids_s, _) = moe_topk_route(&[0.0, 10.0, 0.0], 1, true).unwrap();
        assert_eq!(ids_s, vec![1]);
    }

    #[test]
    fn gated_delta_step_updates_state() {
        let n_heads = 1usize;
        let dk = 2usize;
        let dv = 2usize;
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0];
        let v = [0.5f32, -0.25];
        let g = [0.9f32];
        let beta = [1.0f32];
        let mut s = vec![0.0f32; n_heads * dk * dv];
        let o1 = gated_delta_step(GatedDeltaStep {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            state: &mut s,
            n_heads,
            dk,
            dv,
        })
        .unwrap();
        assert_eq!(o1.len(), dv);
        let s_after = s.clone();
        let o2 = gated_delta_step(GatedDeltaStep {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            state: &mut s,
            n_heads,
            dk,
            dv,
        })
        .unwrap();
        assert!(s != s_after || (o2[0] - o1[0]).abs() > 0.0);
    }

    #[test]
    fn geglu_and_rms_norm_gemma() {
        let gate = [0.5f32, -1.0];
        let up = [2.0f32, 3.0];
        let y = geglu(&gate, &up).unwrap();
        assert!((y[0] - gelu_pytorch_tanh(0.5) * 2.0).abs() < 1e-5);
        assert!((y[1] - gelu_pytorch_tanh(-1.0) * 3.0).abs() < 1e-5);
        let x = [1.0f32, -1.0, 2.0, 0.0];
        let w = [0.1f32, 0.2];
        let n = rms_norm_gemma(&x, &w, 1e-6).unwrap();
        assert_eq!(n.len(), 4);
        // (1+w) vs plain rms_norm weight multiply differs.
        let plain = rms_norm(&x, &w, 1e-6).unwrap();
        assert!((n[0] - plain[0]).abs() > 1e-4 || (n[1] - plain[1]).abs() > 1e-4);
    }

    #[test]
    fn rope_half_layout() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        rope_half(&mut x, 4, 1, 10000.0).unwrap();
        // At pos=1, half=2: rotate (1,3) and (2,4) as pairs across half.
        let freq0 = 1.0 / 10000f32.powf(0.0);
        let (c0, s0) = (freq0.cos(), freq0.sin());
        let freq1 = 1.0 / 10000f32.powf(2.0 / 4.0);
        let (c1, s1) = (freq1.cos(), freq1.sin());
        assert!((x[0] - (1.0 * c0 - 3.0 * s0)).abs() < 1e-5);
        assert!((x[2] - (1.0 * s0 + 3.0 * c0)).abs() < 1e-5);
        assert!((x[1] - (2.0 * c1 - 4.0 * s1)).abs() < 1e-5);
        assert!((x[3] - (2.0 * s1 + 4.0 * c1)).abs() < 1e-5);
        assert!(matches!(
            rope_half(&mut [1.0, 2.0, 3.0], 3, 0, 10000.0),
            Err(EngineError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn hdm_linear_matches_unrotated_weight() {
        let out_f = 8usize;
        let in_f = 4usize;
        let seed = Some(7i64);
        let mut w_orig: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.05 - 0.2).collect();
        let x: Vec<f32> = (0..in_f).map(|i| (i as f32) * 0.1).collect();
        let y_ref = linear(&x, &w_orig, out_f, in_f).unwrap();
        hadamard_blocked_rows(&mut w_orig, out_f, in_f, seed, false).unwrap();
        let y = hdm_linear(&x, &w_orig, out_f, in_f, seed).unwrap();
        for (a, b) in y.iter().zip(y_ref.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn linear_cpu_matches_scalar_linear() {
        let out_f = 7usize;
        let in_f = 5usize;
        let w: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let x: Vec<f32> = (0..in_f * 3).map(|i| (i as f32) * 0.03).collect();
        let a = linear(&x, &w, out_f, in_f).unwrap();
        let b = linear_cpu(&x, &w, out_f, in_f).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn attention_causal_matches_stepwise() {
        let n_heads = 2usize;
        let n_kv = 1usize;
        let head_dim = 4usize;
        let seq = 3usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let q: Vec<f32> = (0..seq * q_dim).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..seq * kv_dim).map(|i| (i as f32) * 0.02).collect();
        let v: Vec<f32> = (0..seq * kv_dim).map(|i| (i as f32) * 0.03).collect();
        let batched = attention_causal(&q, &k, &v, n_heads, n_kv, head_dim).unwrap();
        let mut step = Vec::new();
        for t in 0..seq {
            let qi = &q[t * q_dim..(t + 1) * q_dim];
            let a = attention(
                qi,
                &k[..(t + 1) * kv_dim],
                &v[..(t + 1) * kv_dim],
                n_heads,
                n_kv,
                head_dim,
            )
            .unwrap();
            step.extend_from_slice(&a);
        }
        for (a, b) in batched.iter().zip(step.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn embedding_row_gather_needs_full_matrix_unrotate() {
        let vocab = 8usize;
        let hidden = 4usize;
        let seed = Some(7i64);
        let tid = 3usize;
        let mut w: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32) * 0.05 - 0.2)
            .collect();
        let orig = w[tid * hidden..(tid + 1) * hidden].to_vec();
        hadamard_blocked_rows(&mut w, vocab, hidden, seed, false).unwrap();
        let rotated_row = &w[tid * hidden..(tid + 1) * hidden];
        let drift: f32 = orig
            .iter()
            .zip(rotated_row.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            drift > 1e-3,
            "axis-0 Hadamard must mix vocab rows; gather of W_rot[token] is wrong"
        );
        hadamard_blocked_rows(&mut w, vocab, hidden, seed, true).unwrap();
        for (a, b) in orig.iter().zip(w[tid * hidden..(tid + 1) * hidden].iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
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
