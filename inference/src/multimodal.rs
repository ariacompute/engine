//! Stage C multimodal helpers: vision prefix, ASR stub, VLA action head, RAG pack.

use aria_kernel::EngineError;

/// Flatten RGB bytes (u8) into a mean-pooled f32 vector of `hidden` dims (deterministic stub).
pub fn vision_encode(rgb: &[u8], height: usize, width: usize, hidden: usize) -> Result<Vec<f32>, EngineError> {
    if height == 0 || width == 0 || hidden == 0 {
        return Err(EngineError::InvalidParam("vision dims must be > 0".into()));
    }
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
    let mut out = vec![0.0f32; hidden];
    let pixels = height * width;
    for p in 0..pixels {
        let r = rgb[p * 3] as f32 / 255.0;
        let g = rgb[p * 3 + 1] as f32 / 255.0;
        let b = rgb[p * 3 + 2] as f32 / 255.0;
        let v = (r + g + b) / 3.0;
        out[p % hidden] += v;
    }
    let scale = 1.0 / (pixels as f32).max(1.0);
    for x in &mut out {
        *x *= scale;
    }
    Ok(out)
}

/// Minimal ASR: map PCM16 LE samples to a short token-id sequence (stub for stage C API).
pub fn asr_transcribe_pcm16le(pcm: &[u8], vocab: u32) -> Result<String, EngineError> {
    if pcm.len() < 2 {
        return Err(EngineError::InvalidParam("pcm too short".into()));
    }
    if !pcm.len().is_multiple_of(2) {
        return Err(EngineError::Format("pcm16 length must be even".into()));
    }
    let vocab = vocab.max(1);
    let mut words = Vec::new();
    for chunk in pcm.chunks(2).take(32) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        let tok = (s as i32).unsigned_abs() % vocab;
        words.push(format!("t{tok}"));
    }
    Ok(words.join(" "))
}

/// VLA action head: project hidden state to `action_dim` (linear stub).
pub fn action_head(hidden: &[f32], action_dim: usize) -> Result<Vec<f32>, EngineError> {
    if hidden.is_empty() || action_dim == 0 {
        return Err(EngineError::InvalidParam("action_head dims".into()));
    }
    let mut out = vec![0.0f32; action_dim];
    for (i, o) in out.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for (j, &h) in hidden.iter().enumerate() {
            s += h * (((i + 1) * (j + 1)) as f32 * 0.001);
        }
        *o = s.tanh();
    }
    Ok(out)
}

/// Pack retrieved RAG snippets into a system-style context string.
pub fn rag_pack_context(snippets: &[String], query: &str) -> String {
    let mut ctx = String::from("RAG context:\n");
    for (i, s) in snippets.iter().enumerate() {
        ctx.push_str(&format!("[{i}] {s}\n"));
    }
    ctx.push_str("Query: ");
    ctx.push_str(query);
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_and_asr_and_action() {
        let rgb = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let v = vision_encode(&rgb, 2, 2, 8).unwrap();
        assert_eq!(v.len(), 8);
        let t = asr_transcribe_pcm16le(&[0, 1, 2, 3, 4, 5], 128).unwrap();
        assert!(!t.is_empty());
        let a = action_head(&[0.5, -0.2, 0.1], 4).unwrap();
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn vision_shape_err() {
        assert!(matches!(
            vision_encode(&[1, 2], 2, 2, 4),
            Err(EngineError::ShapeMismatch(_))
        ));
    }
}
