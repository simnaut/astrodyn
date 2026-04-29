//! Compact binary serialization for gravity coefficients.
//!
//! Production gravity loads pre-built `.bin` blobs through
//! [`load_binary`] / [`load_binary_from_bytes`]. The blobs are produced
//! by [`save_binary`] from [`SphericalHarmonicsData`] values that test
//! infrastructure assembles from JEOD's `.cc` source files.
//!
//! Parsing JEOD's `.cc` source files lives in
//! `jeod_test_data::jeod_cc` (a dev/test-only crate). `jeod_gravity`
//! itself does not know how to parse JEOD source — only how to consume
//! the binary format derived from it.

use crate::spherical_harmonics_gravity_source::SphericalHarmonicsData;

/// Errors from loading a binary spherical-harmonics coefficient blob.
#[derive(Debug, thiserror::Error)]
pub enum CoeffLoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid binary format: {0}")]
    InvalidFormat(String),
}

/// Save coefficients to a compact binary format.
///
/// Format: degree(u32), order(u32), radius(f64), mu(f64),
///         tide_free(u8), tide_free_delta(f64),
///         then for each n=0..degree: `cnm[n][0..n]` as f64,
///         then for each n=0..degree: `snm[n][0..n]` as f64.
pub fn save_binary(
    data: &SphericalHarmonicsData,
    path: &std::path::Path,
) -> Result<(), std::io::Error> {
    use std::io::Write;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"JEOD"); // 4-byte magic
    buf.extend_from_slice(&1u32.to_le_bytes()); // 4-byte version
    buf.extend_from_slice(&(data.degree as u32).to_le_bytes());
    buf.extend_from_slice(&(data.order as u32).to_le_bytes());
    buf.extend_from_slice(&data.radius.to_le_bytes());
    buf.extend_from_slice(&data.mu.to_le_bytes());
    buf.push(if data.tide_free { 1 } else { 0 });
    buf.extend_from_slice(&data.tide_free_delta.to_le_bytes());
    for n in 0..=data.degree {
        for m in 0..=n {
            buf.extend_from_slice(&data.cnm[n][m].to_le_bytes());
        }
    }
    for n in 0..=data.degree {
        for m in 0..=n {
            buf.extend_from_slice(&data.snm[n][m].to_le_bytes());
        }
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&buf)?;
    Ok(())
}

/// Load coefficients from the compact binary format.
pub fn load_binary(path: &std::path::Path) -> Result<SphericalHarmonicsData, CoeffLoadError> {
    let buf = std::fs::read(path)?;
    load_binary_from_bytes(&buf)
}

/// Load coefficients from binary bytes (for embedded data).
pub fn load_binary_from_bytes(buf: &[u8]) -> Result<SphericalHarmonicsData, CoeffLoadError> {
    let mut pos = 0;

    // Bounds-checked read helpers (prevent panics on truncated/corrupted files)
    let read_u32 = |pos: &mut usize| -> Result<u32, CoeffLoadError> {
        if *pos + 4 > buf.len() {
            return Err(CoeffLoadError::InvalidFormat(format!(
                "truncated binary file at offset {}",
                *pos
            )));
        }
        let val = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(val)
    };
    let read_f64 = |pos: &mut usize| -> Result<f64, CoeffLoadError> {
        if *pos + 8 > buf.len() {
            return Err(CoeffLoadError::InvalidFormat(format!(
                "truncated binary file at offset {}",
                *pos
            )));
        }
        let val = f64::from_le_bytes(buf[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        Ok(val)
    };
    let read_u8 = |pos: &mut usize| -> Result<u8, CoeffLoadError> {
        if *pos >= buf.len() {
            return Err(CoeffLoadError::InvalidFormat(format!(
                "truncated binary file at offset {}",
                *pos
            )));
        }
        let val = buf[*pos];
        *pos += 1;
        Ok(val)
    };

    // Magic number
    if buf.len() < 8 {
        return Err(CoeffLoadError::InvalidFormat(
            "binary coefficient file too short".into(),
        ));
    }
    if &buf[0..4] != b"JEOD" {
        return Err(CoeffLoadError::InvalidFormat(
            "invalid magic in binary coefficient file".into(),
        ));
    }
    pos += 4;
    let version = read_u32(&mut pos)?;
    if version != 1 {
        return Err(CoeffLoadError::InvalidFormat(format!(
            "unsupported binary coefficient version {version}"
        )));
    }

    let degree = read_u32(&mut pos)? as usize;
    let order = read_u32(&mut pos)? as usize;

    // Sanity checks to prevent unbounded memory allocation
    if degree > 10000 {
        return Err(CoeffLoadError::InvalidFormat(format!(
            "degree {degree} exceeds maximum supported (10000)"
        )));
    }
    if order > degree {
        return Err(CoeffLoadError::InvalidFormat(format!(
            "order ({order}) exceeds degree ({degree})"
        )));
    }

    // Verify buffer has enough data for the claimed degree
    // Header: 4(magic) + 4(version) + 4(degree) + 4(order) + 8(radius) + 8(mu)
    //         + 1(tide_free) + 8(tide_free_delta) = 41 bytes
    // Coefficients: 2 * sum(n+1 for n=0..=degree) * 8 = 2 * (degree+1)*(degree+2)/2 * 8
    let num_coeffs = (degree + 1) * (degree + 2) / 2;
    let expected_size = 41 + 2 * num_coeffs * 8;
    if buf.len() < expected_size {
        return Err(CoeffLoadError::InvalidFormat(format!(
            "binary file too short for degree {degree}: need {expected_size} bytes, have {}",
            buf.len()
        )));
    }

    let radius = read_f64(&mut pos)?;
    let mu = read_f64(&mut pos)?;
    let tide_free = read_u8(&mut pos)? != 0;
    let tide_free_delta = read_f64(&mut pos)?;

    let mut cnm = Vec::with_capacity(degree + 1);
    for n in 0..=degree {
        let mut row = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            row.push(read_f64(&mut pos)?);
        }
        cnm.push(row);
    }

    let mut snm = Vec::with_capacity(degree + 1);
    for n in 0..=degree {
        let mut row = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            row.push(read_f64(&mut pos)?);
        }
        snm.push(row);
    }

    Ok(SphericalHarmonicsData::new(
        degree,
        order,
        radius,
        mu,
        cnm,
        snm,
        tide_free,
        tide_free_delta,
    ))
}
