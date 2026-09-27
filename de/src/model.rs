//! Candle DecisionModel skeleton (ModernBERT encoder + MASK scorer).

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Typed decision head over MASK hidden states (Laya-aligned shape).
pub struct DecisionHead {
    pub type_emb: Tensor,
    pub scorer: Linear,
}

impl DecisionHead {
    pub fn new(vb: VarBuilder<'_>, hidden: usize, n_types: usize) -> CandleResult<Self> {
        let type_emb = vb.get((n_types, hidden), "type_emb")?;
        let scorer = linear(hidden, 1, vb.pp("scorer"))?;
        Ok(Self { type_emb, scorer })
    }

    /// logits: (B, K) from mask_h (B, K, H) + type id.
    pub fn forward(&self, mask_h: &Tensor, type_id: usize) -> CandleResult<Tensor> {
        let emb = self.type_emb.get(type_id)?;
        let h = mask_h.broadcast_add(&emb)?;
        let logits = self.scorer.forward(&h)?.squeeze(candle_core::D::Minus1)?;
        Ok(logits)
    }
}

pub fn cpu() -> Device {
    Device::Cpu
}
