//! Scalar (+ aarch64 NEON / x86 AVX2 + optional CUDA) kernels for Aria engine.

mod compute;
mod cuda_rt;
mod error;
mod ops;

pub use compute::{cpu_simd_label, resolve_compute, ComputeBackend, ComputePref};
pub use cuda_rt::CudaContext;
pub use error::EngineError;
pub use ops::*;

/// Runtime SIMD selection. Tests force [`SimdMode::Scalar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimdMode {
    #[default]
    Scalar,
    Neon,
    Avx2,
}

impl SimdMode {
    /// Prefer Neon on aarch64, AVX2 on x86_64 when detected, else Scalar.
    pub fn auto() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                Self::Avx2
            } else {
                Self::Scalar
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }
}
