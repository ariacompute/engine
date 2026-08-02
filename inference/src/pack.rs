use aria_kernel::EngineError;

const INTEGER_BITS: [u8; 5] = [1, 2, 3, 4, 8];

pub fn packed_size(count: usize, bits: u8) -> Result<usize, EngineError> {
    if !INTEGER_BITS.contains(&bits) {
        return Err(EngineError::Quant(format!("bits must be 1-4 or 8, got {bits}")));
    }
    Ok((count * bits as usize).div_ceil(8))
}

pub fn pack_indices(indices: &[u8], bits: u8) -> Result<Vec<u8>, EngineError> {
    if !INTEGER_BITS.contains(&bits) {
        return Err(EngineError::Quant(format!("invalid bits {bits}")));
    }
    if bits == 8 {
        return Ok(indices.to_vec());
    }
    let max_val = (1u8 << bits) - 1;
    if let Some(m) = indices.iter().copied().max() {
        if m > max_val {
            return Err(EngineError::Quant(format!(
                "index {m} exceeds max for {bits}-bit"
            )));
        }
    }
    let mut out = vec![0u8; packed_size(indices.len(), bits)?];
    let mut bit_pos = 0usize;
    for &v0 in indices {
        let v = v0 & max_val;
        for b in 0..bits {
            if v & (1 << b) != 0 {
                let byte_i = bit_pos / 8;
                let bit_i = bit_pos % 8;
                out[byte_i] |= 1 << bit_i;
            }
            bit_pos += 1;
        }
    }
    Ok(out)
}

pub fn unpack_indices(data: &[u8], count: usize, bits: u8) -> Result<Vec<u8>, EngineError> {
    let need = packed_size(count, bits)?;
    if data.len() < need {
        return Err(EngineError::ShapeMismatch(format!(
            "packed data length {} < required {need}",
            data.len()
        )));
    }
    if bits == 8 {
        return Ok(data[..need].to_vec());
    }
    let max_val = (1u8 << bits) - 1;
    let mut out = vec![0u8; count];
    let mut bit_pos = 0usize;
    for slot in &mut out {
        let mut v = 0u8;
        for b in 0..bits {
            let byte_i = bit_pos / 8;
            let bit_i = bit_pos % 8;
            if data[byte_i] & (1 << bit_i) != 0 {
                v |= 1 << b;
            }
            bit_pos += 1;
        }
        *slot = v & max_val;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_4bit() {
        let idx = [0u8, 1, 2, 15, 3, 4];
        let packed = pack_indices(&idx, 4).unwrap();
        let back = unpack_indices(&packed, idx.len(), 4).unwrap();
        assert_eq!(back, idx);
    }

    #[test]
    fn pack_8bit() {
        let idx = [0u8, 128, 255];
        let packed = pack_indices(&idx, 8).unwrap();
        assert_eq!(packed, idx);
    }
}
