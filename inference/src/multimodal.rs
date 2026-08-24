//! Stage C multimodal helpers: ASR stub + RAG pack.
//! Vision / VLA action require real bundle tensors — Session APIs return `Unsupported`
//! rather than RGB mean-pool / fake linear stubs.

use aria_kernel::EngineError;

/// Legacy helper kept for unit shape checks; Session no longer calls this for VL.
pub fn vision_encode(
    rgb: &[u8],
    height: usize,
    width: usize,
    hidden: usize,
) -> Result<Vec<f32>, EngineError> {
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
    Err(EngineError::Unsupported(
        "vision_encode: vision tower not implemented (refusing RGB mean-pool stub)".into(),
    ))
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

/// VLA action head: not implemented without bundle action weights.
pub fn action_head(_hidden: &[f32], action_dim: usize) -> Result<Vec<f32>, EngineError> {
    if action_dim == 0 {
        return Err(EngineError::InvalidParam("action_head dims".into()));
    }
    Err(EngineError::Unsupported(
        "action_head: action weights not implemented (refusing fake linear stub)".into(),
    ))
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
        assert!(matches!(
            vision_encode(&rgb, 2, 2, 8),
            Err(EngineError::Unsupported(_))
        ));
        let t = asr_transcribe_pcm16le(&[0, 1, 2, 3, 4, 5], 128).unwrap();
        assert!(!t.is_empty());
        assert!(matches!(
            action_head(&[0.5, -0.2, 0.1], 4),
            Err(EngineError::Unsupported(_))
        ));
    }

    #[test]
    fn vision_shape_err() {
        assert!(matches!(
            vision_encode(&[1, 2], 2, 2, 4),
            Err(EngineError::ShapeMismatch(_))
        ));
    }
}
