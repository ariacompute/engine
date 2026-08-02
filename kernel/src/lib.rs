//! Scalar (+ aarch64 NEON entry) kernels for Aria engine.

mod error;
mod ops;

pub use error::EngineError;
pub use ops::*;

/// Runtime SIMD selection. Tests force [`SimdMode::Scalar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimdMode {
    #[default]
    Scalar,
    Neon,
}

impl SimdMode {
    /// Prefer Neon on aarch64 unless forced otherwise.
    pub fn auto() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::Scalar
        }
    }
}
