//! Local compute preference (`auto|cpu|cuda`) for GEMM. Not a routing switch.

use crate::cuda_rt;
use crate::{EngineError, SimdMode};

/// CLI / config preference for local GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputePref {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl ComputePref {
    pub fn parse(raw: &str) -> Result<Self, EngineError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => Err(EngineError::InvalidParam(format!(
                "compute must be auto|cpu|cuda, got {other:?}"
            ))),
        }
    }
}

/// Resolved backend used by Session GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Cpu,
    Cuda,
}

pub fn cpu_simd_label() -> String {
    match SimdMode::auto() {
        SimdMode::Neon => "neon".into(),
        SimdMode::Avx2 => "avx2".into(),
        SimdMode::Scalar => "scalar".into(),
    }
}

/// Resolve preference. `Cuda` never silently falls back to CPU.
pub fn resolve_compute(pref: ComputePref) -> Result<(ComputeBackend, String), EngineError> {
    match pref {
        ComputePref::Cpu => Ok((
            ComputeBackend::Cpu,
            format!("cpu simd={}", cpu_simd_label()),
        )),
        ComputePref::Cuda => {
            let info = cuda_rt::device_info().map_err(|e| {
                EngineError::Unsupported(format!(
                    "compute=cuda requested but CUDA/cuBLAS is unavailable: {e}"
                ))
            })?;
            Ok((ComputeBackend::Cuda, format!("cuda {info}")))
        }
        ComputePref::Auto => match cuda_rt::device_info() {
            Ok(info) => Ok((ComputeBackend::Cuda, format!("cuda {info}"))),
            Err(_) => Ok((
                ComputeBackend::Cpu,
                format!("cpu simd={}", cpu_simd_label()),
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_compute_pref() {
        assert_eq!(ComputePref::parse("auto").unwrap(), ComputePref::Auto);
        assert_eq!(ComputePref::parse("CPU").unwrap(), ComputePref::Cpu);
        assert_eq!(ComputePref::parse("cuda").unwrap(), ComputePref::Cuda);
        assert_eq!(ComputePref::parse("gpu").unwrap(), ComputePref::Cuda);
        assert!(matches!(
            ComputePref::parse("tpu"),
            Err(EngineError::InvalidParam(_))
        ));
    }

    #[test]
    fn auto_never_errors() {
        let (backend, label) = resolve_compute(ComputePref::Auto).unwrap();
        assert!(!label.is_empty());
        match backend {
            ComputeBackend::Cpu => assert!(label.contains("cpu")),
            ComputeBackend::Cuda => assert!(label.contains("cuda")),
        }
    }

    #[test]
    fn explicit_cuda_fails_without_device() {
        if cuda_rt::device_info().is_err() {
            let err = resolve_compute(ComputePref::Cuda).unwrap_err();
            assert!(matches!(err, EngineError::Unsupported(_)), "{err}");
        }
    }
}
