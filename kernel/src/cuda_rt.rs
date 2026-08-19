//! Runtime CUDA/cuBLAS via libloading (no compile-time toolkit required).

use crate::EngineError;
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::Path;
use std::sync::Mutex;

const CUDA_SUCCESS: c_int = 0;
const CUBLAS_SUCCESS: c_int = 0;
const CUDA_MEMCPY_H2D: c_int = 1;
const CUDA_MEMCPY_D2H: c_int = 2;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;

type CudaGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> c_int;
type CudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type CudaFree = unsafe extern "C" fn(*mut c_void) -> c_int;
type CudaMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type CudaGetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;
type CublasCreate = unsafe extern "C" fn(*mut *mut c_void) -> c_int;
type CublasDestroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type CublasSgemm = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> c_int;

fn open_first(candidates: &[&str]) -> Result<Library, EngineError> {
    let mut last = String::from("no candidate loaded");
    for c in candidates {
        match unsafe { Library::new(c) } {
            Ok(lib) => return Ok(lib),
            Err(e) => last = format!("{c}: {e}"),
        }
        #[cfg(unix)]
        if let Ok(lib) = unsafe { Library::new(Path::new(c)) } {
            return Ok(lib);
        }
    }
    Err(EngineError::Unsupported(format!("libloading: {last}")))
}

fn nvidia_smi_name() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(line.to_string())
    }
}

/// Probe CUDA runtime + cuBLAS. Does not allocate a persistent handle.
pub fn device_info() -> Result<String, EngineError> {
    let cudart = open_first(&[
        "libcudart.so.12",
        "libcudart.so",
        "nvcudart.dll",
        "libcudart.dylib",
    ])?;
    let cublas = open_first(&[
        "libcublas.so.12",
        "libcublas.so",
        "cublas64_12.dll",
        "libcublas.dylib",
    ])?;
    let get_count: Symbol<CudaGetDeviceCount> = unsafe {
        cudart
            .get(b"cudaGetDeviceCount")
            .map_err(|e| EngineError::Unsupported(e.to_string()))?
    };
    let mut n: c_int = 0;
    let st = unsafe { get_count(&mut n) };
    if st != CUDA_SUCCESS || n <= 0 {
        return Err(EngineError::Unsupported(format!(
            "cudaGetDeviceCount status={st} count={n}"
        )));
    }
    // Touch a cublas symbol so missing SONAME fails here, not at first GEMM.
    let _create: Symbol<CublasCreate> = unsafe {
        cublas
            .get(b"cublasCreate_v2")
            .map_err(|e| EngineError::Unsupported(e.to_string()))?
    };
    let name = nvidia_smi_name().unwrap_or_else(|| format!("{n} device(s)"));
    Ok(name)
}

struct Fns {
    malloc: CudaMalloc,
    free: CudaFree,
    memcpy: CudaMemcpy,
    errstr: CudaGetErrorString,
    sgemm: CublasSgemm,
}

fn cuda_err(errstr: CudaGetErrorString, st: c_int, what: &str) -> EngineError {
    let msg = unsafe {
        let p = errstr(st);
        if p.is_null() {
            format!("{what} cuda status={st}")
        } else {
            format!("{what}: {}", CStr::from_ptr(p).to_string_lossy())
        }
    };
    EngineError::Unsupported(msg)
}

/// Persistent cuBLAS handle + device copies of host weight buffers.
pub struct CudaContext {
    _cudart: Library,
    _cublas: Library,
    handle: *mut c_void,
    fns: Fns,
    weights: Mutex<HashMap<usize, (*mut f32, usize)>>,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new() -> Result<Self, EngineError> {
        let cudart = open_first(&[
            "libcudart.so.12",
            "libcudart.so",
            "nvcudart.dll",
            "libcudart.dylib",
        ])?;
        let cublas = open_first(&[
            "libcublas.so.12",
            "libcublas.so",
            "cublas64_12.dll",
            "libcublas.dylib",
        ])?;
        let malloc: Symbol<CudaMalloc> = unsafe {
            cudart
                .get(b"cudaMalloc")
                .map_err(|e| EngineError::Unsupported(e.to_string()))?
        };
        let free: Symbol<CudaFree> = unsafe {
            cudart
                .get(b"cudaFree")
                .map_err(|e| EngineError::Unsupported(e.to_string()))?
        };
        let memcpy: Symbol<CudaMemcpy> = unsafe {
            cudart
                .get(b"cudaMemcpy")
                .map_err(|e| EngineError::Unsupported(e.to_string()))?
        };
        let errstr: Symbol<CudaGetErrorString> = unsafe {
            cudart
                .get(b"cudaGetErrorString")
                .map_err(|e| EngineError::Unsupported(e.to_string()))?
        };
        let create: Symbol<CublasCreate> = unsafe {
            cublas
                .get(b"cublasCreate_v2")
                .map_err(|e| EngineError::Unsupported(e.to_string()))?
        };
        let sgemm: Symbol<CublasSgemm> = unsafe {
            cublas
                .get(b"cublasSgemm_v2")
                .map_err(|e| EngineError::Unsupported(e.to_string()))?
        };
        let fns = Fns {
            malloc: *malloc,
            free: *free,
            memcpy: *memcpy,
            errstr: *errstr,
            sgemm: *sgemm,
        };
        let mut handle: *mut c_void = std::ptr::null_mut();
        let st = unsafe { create(&mut handle) };
        if st != CUBLAS_SUCCESS || handle.is_null() {
            return Err(EngineError::Unsupported(format!(
                "cublasCreate_v2 status={st}"
            )));
        }
        Ok(Self {
            _cudart: cudart,
            _cublas: cublas,
            handle,
            fns,
            weights: Mutex::new(HashMap::new()),
        })
    }

    pub fn upload(&self, host: &[f32]) -> Result<(), EngineError> {
        if host.is_empty() {
            return Ok(());
        }
        let key = host.as_ptr() as usize;
        {
            let map = self
                .weights
                .lock()
                .map_err(|e| EngineError::Unsupported(e.to_string()))?;
            if let Some((_, n)) = map.get(&key) {
                if *n == host.len() {
                    return Ok(());
                }
            }
        }
        let bytes = host.len() * 4;
        let mut dev: *mut c_void = std::ptr::null_mut();
        let st = unsafe { (self.fns.malloc)(&mut dev, bytes) };
        if st != CUDA_SUCCESS {
            return Err(cuda_err(self.fns.errstr, st, "cudaMalloc"));
        }
        let st = unsafe { (self.fns.memcpy)(dev, host.as_ptr().cast(), bytes, CUDA_MEMCPY_H2D) };
        if st != CUDA_SUCCESS {
            unsafe {
                (self.fns.free)(dev);
            }
            return Err(cuda_err(self.fns.errstr, st, "cudaMemcpy H2D"));
        }
        let mut map = self
            .weights
            .lock()
            .map_err(|e| EngineError::Unsupported(e.to_string()))?;
        if let Some((old, _)) = map.insert(key, (dev.cast(), host.len())) {
            unsafe {
                (self.fns.free)(old.cast());
            }
        }
        Ok(())
    }

    /// y = W @ x for row-major W `[out_f, in_f]`, x `[batch, in_f]`.
    pub fn linear(
        &self,
        x: &[f32],
        w: &[f32],
        out_f: usize,
        in_f: usize,
    ) -> Result<Vec<f32>, EngineError> {
        if in_f == 0 || !x.len().is_multiple_of(in_f) {
            return Err(EngineError::ShapeMismatch("cuda linear x".into()));
        }
        if w.len() != out_f * in_f {
            return Err(EngineError::ShapeMismatch("cuda linear w".into()));
        }
        self.upload(w)?;
        let batch = x.len() / in_f;
        let key = w.as_ptr() as usize;
        let map = self
            .weights
            .lock()
            .map_err(|e| EngineError::Unsupported(e.to_string()))?;
        let (w_dev, n) = map
            .get(&key)
            .copied()
            .ok_or_else(|| EngineError::Unsupported("cuda weight not uploaded".into()))?;
        if n != w.len() {
            return Err(EngineError::Unsupported("cuda weight size drift".into()));
        }
        drop(map);

        let x_bytes = x.len() * 4;
        let y_bytes = batch * out_f * 4;
        let mut x_dev: *mut c_void = std::ptr::null_mut();
        let mut y_dev: *mut c_void = std::ptr::null_mut();
        let st = unsafe { (self.fns.malloc)(&mut x_dev, x_bytes) };
        if st != CUDA_SUCCESS {
            return Err(cuda_err(self.fns.errstr, st, "cudaMalloc x"));
        }
        let st = unsafe { (self.fns.malloc)(&mut y_dev, y_bytes) };
        if st != CUDA_SUCCESS {
            unsafe {
                (self.fns.free)(x_dev);
            }
            return Err(cuda_err(self.fns.errstr, st, "cudaMalloc y"));
        }
        let st = unsafe { (self.fns.memcpy)(x_dev, x.as_ptr().cast(), x_bytes, CUDA_MEMCPY_H2D) };
        if st != CUDA_SUCCESS {
            unsafe {
                (self.fns.free)(x_dev);
                (self.fns.free)(y_dev);
            }
            return Err(cuda_err(self.fns.errstr, st, "cudaMemcpy x"));
        }

        let alpha = 1.0f32;
        let beta = 0.0f32;
        // See plan: W row-major [out,in] == col-major [in,out]; y = W_rm @ x => SGEMM(T, N).
        let m = out_f as c_int;
        let n_b = batch as c_int;
        let k = in_f as c_int;
        let st = unsafe {
            (self.fns.sgemm)(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                m,
                n_b,
                k,
                &alpha,
                w_dev,
                k,
                x_dev.cast(),
                k,
                &beta,
                y_dev.cast(),
                m,
            )
        };
        if st != CUBLAS_SUCCESS {
            unsafe {
                (self.fns.free)(x_dev);
                (self.fns.free)(y_dev);
            }
            return Err(EngineError::Unsupported(format!(
                "cublasSgemm_v2 status={st}"
            )));
        }
        let mut y = vec![0.0f32; batch * out_f];
        let st = unsafe { (self.fns.memcpy)(y.as_mut_ptr().cast(), y_dev, y_bytes, CUDA_MEMCPY_D2H) };
        unsafe {
            (self.fns.free)(x_dev);
            (self.fns.free)(y_dev);
        }
        if st != CUDA_SUCCESS {
            return Err(cuda_err(self.fns.errstr, st, "cudaMemcpy y"));
        }
        let _ = CUBLAS_OP_N; // keep const used in docs
        Ok(y)
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if let Ok(mut map) = self.weights.lock() {
            for (_, (ptr, _)) in map.drain() {
                unsafe {
                    (self.fns.free)(ptr.cast());
                }
            }
        }
        if let Ok(destroy) = unsafe { self._cublas.get::<CublasDestroy>(b"cublasDestroy_v2") } {
            if !self.handle.is_null() {
                unsafe {
                    destroy(self.handle);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear;

    #[test]
    fn cuda_linear_matches_cpu_if_available() {
        let Ok(ctx) = CudaContext::new() else {
            return;
        };
        let out_f = 4usize;
        let in_f = 3usize;
        let w: Vec<f32> = (0..out_f * in_f).map(|i| i as f32 * 0.1 - 0.2).collect();
        let x: Vec<f32> = (0..in_f).map(|i| i as f32 * 0.25).collect();
        let cpu = linear(&x, &w, out_f, in_f).unwrap();
        let gpu = ctx.linear(&x, &w, out_f, in_f).unwrap();
        for (a, b) in cpu.iter().zip(gpu.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        let xb: Vec<f32> = (0..in_f * 3).map(|i| i as f32 * 0.1).collect();
        let cpu_b = linear(&xb, &w, out_f, in_f).unwrap();
        let gpu_b = ctx.linear(&xb, &w, out_f, in_f).unwrap();
        for (a, b) in cpu_b.iter().zip(gpu_b.iter()) {
            assert!((a - b).abs() < 1e-3, "batch {a} vs {b}");
        }
    }
}
