//! Optional load / generate timings for `--profile`.

use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::time::Instant;

thread_local! {
    static LOAD_ON: Cell<bool> = const { Cell::new(false) };
    static LOAD: RefCell<LoadProfile> = const { RefCell::new(LoadProfile::empty()) };
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadProfile {
    pub mmap_ms: f64,
    pub dequant_ms: f64,
    pub unrotate_ms: f64,
    pub materialize_ms: f64,
    pub cuda_upload_ms: f64,
}

impl LoadProfile {
    const fn empty() -> Self {
        Self {
            mmap_ms: 0.0,
            dequant_ms: 0.0,
            unrotate_ms: 0.0,
            materialize_ms: 0.0,
            cuda_upload_ms: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct GenerateProfile {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub gemm_attn_ms: f64,
    pub gemm_ffn_ms: f64,
    pub gemm_lm_head_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineProfile {
    pub compute: String,
    pub load: LoadProfile,
    pub generate: Option<GenerateProfile>,
    pub ci_fail: bool,
}

pub fn load_profile_begin(enabled: bool) {
    LOAD_ON.with(|c| c.set(enabled));
    LOAD.with(|l| *l.borrow_mut() = LoadProfile::empty());
}

pub fn load_profile_enabled() -> bool {
    LOAD_ON.with(|c| c.get())
}

pub fn load_profile_add_dequant(ms: f64) {
    if !load_profile_enabled() {
        return;
    }
    LOAD.with(|l| l.borrow_mut().dequant_ms += ms);
}

pub fn load_profile_add_unrotate(ms: f64) {
    if !load_profile_enabled() {
        return;
    }
    LOAD.with(|l| l.borrow_mut().unrotate_ms += ms);
}

pub fn load_profile_set_mmap(ms: f64) {
    LOAD.with(|l| l.borrow_mut().mmap_ms = ms);
}

pub fn load_profile_set_materialize(ms: f64) {
    LOAD.with(|l| l.borrow_mut().materialize_ms = ms);
}

pub fn load_profile_set_cuda_upload(ms: f64) {
    LOAD.with(|l| l.borrow_mut().cuda_upload_ms = ms);
}

pub fn load_profile_take() -> LoadProfile {
    LOAD.with(|l| l.borrow().clone())
}

pub fn elapsed_ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_accum_when_enabled() {
        load_profile_begin(true);
        load_profile_add_dequant(1.5);
        load_profile_add_unrotate(2.5);
        let p = load_profile_take();
        assert!((p.dequant_ms - 1.5).abs() < 1e-9);
        assert!((p.unrotate_ms - 2.5).abs() < 1e-9);
        load_profile_begin(false);
        load_profile_add_dequant(9.0);
        assert!(load_profile_take().dequant_ms < 1e-9);
    }
}
