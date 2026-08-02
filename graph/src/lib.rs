//! Zero-copy compute graph: Layer → Op → TensorView + BufferPool.

use aria_kernel::{linear, matmul_dispatch, EngineError, SimdMode};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    U8,
}

/// Borrowed or owned tensor bytes (mmap / external / pool).
#[derive(Clone)]
pub enum TensorBuf {
    External(Arc<[u8]>),
    Owned(Vec<u8>),
}

impl TensorBuf {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::External(a) => a.as_ref(),
            Self::Owned(v) => v.as_slice(),
        }
    }
}

#[derive(Clone)]
pub struct TensorView {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub buf: TensorBuf,
    pub offset: usize,
    pub len: usize,
}

impl TensorView {
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let nbytes = data.len() * 4;
        let mut bytes = Vec::with_capacity(nbytes);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let strides = row_major_strides(&shape, 4);
        Self {
            dtype: DType::F32,
            shape,
            strides,
            buf: TensorBuf::Owned(bytes),
            offset: 0,
            len: nbytes,
        }
    }

    pub fn from_external(bytes: Arc<[u8]>, dtype: DType, shape: Vec<usize>, offset: usize, len: usize) -> Self {
        let elem = match dtype {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::U8 => 1,
        };
        let strides = row_major_strides(&shape, elem);
        Self {
            dtype,
            shape,
            strides,
            buf: TensorBuf::External(bytes),
            offset,
            len,
        }
    }

    pub fn as_f32_slice(&self) -> Result<&[f32], EngineError> {
        if self.dtype != DType::F32 {
            return Err(EngineError::ShapeMismatch("expected f32 tensor".into()));
        }
        let bytes = &self.buf.as_slice()[self.offset..self.offset + self.len];
        if bytes.len() % 4 != 0 {
            return Err(EngineError::Format("f32 byte length not aligned".into()));
        }
        let ptr = bytes.as_ptr() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, bytes.len() / 4) })
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, EngineError> {
        Ok(self.as_f32_slice()?.to_vec())
    }
}

fn row_major_strides(shape: &[usize], elem: usize) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = elem;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i].max(1);
    }
    strides
}

#[derive(Default)]
pub struct BufferPool {
    buffers: Vec<Vec<u8>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn alloc(&mut self, nbytes: usize) -> &mut [u8] {
        self.buffers.push(vec![0u8; nbytes]);
        self.buffers.last_mut().unwrap().as_mut_slice()
    }

    pub fn reuse_count(&self) -> usize {
        self.buffers.len()
    }
}

#[derive(Debug, Clone)]
pub enum Op {
    /// y = x @ W^T  (W: [out, in])
    Linear { out_f: usize, in_f: usize },
    /// Generic matmul
    MatMul {
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
    },
    /// Fused Hadamard + Dequant + MatMul placeholder: dequantized W already provided as input1.
    HdmLinear { out_f: usize, in_f: usize },
}

pub struct Node {
    pub op: Op,
    pub inputs: Vec<usize>,
    pub output: usize,
}

pub struct Graph {
    pub tensors: Vec<Option<TensorView>>,
    pub nodes: Vec<Node>,
    pub mode: SimdMode,
}

impl Graph {
    pub fn new(mode: SimdMode) -> Self {
        Self {
            tensors: Vec::new(),
            nodes: Vec::new(),
            mode,
        }
    }

    pub fn push_tensor(&mut self, t: TensorView) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Some(t));
        id
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    pub fn execute(&mut self, _pool: &mut BufferPool) -> Result<(), EngineError> {
        for node in &self.nodes {
            match &node.op {
                Op::Linear { out_f, in_f } | Op::HdmLinear { out_f, in_f } => {
                    let x = self.tensors[node.inputs[0]]
                        .as_ref()
                        .ok_or_else(|| EngineError::Format("missing input tensor".into()))?
                        .to_f32_vec()?;
                    let w = self.tensors[node.inputs[1]]
                        .as_ref()
                        .ok_or_else(|| EngineError::Format("missing weight tensor".into()))?
                        .to_f32_vec()?;
                    let y = linear(&x, &w, *out_f, *in_f)?;
                    let shape = if x.len() == *in_f {
                        vec![*out_f]
                    } else {
                        vec![x.len() / *in_f, *out_f]
                    };
                    self.tensors[node.output] = Some(TensorView::from_f32(y, shape));
                }
                Op::MatMul {
                    a_rows,
                    a_cols,
                    b_rows,
                    b_cols,
                } => {
                    let a = self.tensors[node.inputs[0]]
                        .as_ref()
                        .ok_or_else(|| EngineError::Format("missing A".into()))?
                        .to_f32_vec()?;
                    let b = self.tensors[node.inputs[1]]
                        .as_ref()
                        .ok_or_else(|| EngineError::Format("missing B".into()))?
                        .to_f32_vec()?;
                    let c = matmul_dispatch(
                        &a, *a_rows, *a_cols, &b, *b_rows, *b_cols, self.mode,
                    )?;
                    self.tensors[node.output] =
                        Some(TensorView::from_f32(c, vec![*a_rows, *b_cols]));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_dispatch() {
        let mut g = Graph::new(SimdMode::Scalar);
        let x = g.push_tensor(TensorView::from_f32(vec![1.0, 0.0], vec![2]));
        // W 2x2 identity rows
        let w = g.push_tensor(TensorView::from_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]));
        let y = g.push_tensor(TensorView::from_f32(vec![0.0, 0.0], vec![2]));
        g.add_node(Node {
            op: Op::Linear { out_f: 2, in_f: 2 },
            inputs: vec![x, w],
            output: y,
        });
        let mut pool = BufferPool::new();
        g.execute(&mut pool).unwrap();
        let out = g.tensors[y].as_ref().unwrap().to_f32_vec().unwrap();
        assert_eq!(out, vec![1.0, 0.0]);
    }

    #[test]
    fn external_zero_copy() {
        let bytes: Arc<[u8]> = Arc::from([0u8, 0, 0x80, 0x3f].as_slice()); // 1.0f32 LE
        let t = TensorView::from_external(bytes, DType::F32, vec![1], 0, 4);
        assert_eq!(t.to_f32_vec().unwrap(), vec![1.0]);
    }

    #[test]
    fn matmul_shape_err() {
        let mut g = Graph::new(SimdMode::Scalar);
        let a = g.push_tensor(TensorView::from_f32(vec![1.0], vec![1, 1]));
        let b = g.push_tensor(TensorView::from_f32(vec![1.0, 2.0], vec![2, 1]));
        let c = g.push_tensor(TensorView::from_f32(vec![0.0], vec![1, 1]));
        g.add_node(Node {
            op: Op::MatMul {
                a_rows: 1,
                a_cols: 1,
                b_rows: 2,
                b_cols: 1,
            },
            inputs: vec![a, b],
            output: c,
        });
        let err = g.execute(&mut BufferPool::new()).unwrap_err();
        assert!(matches!(err, EngineError::ShapeMismatch(_)));
    }
}
