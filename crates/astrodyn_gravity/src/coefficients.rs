//! Compact binary serialization for gravity coefficients.
//!
//! Production gravity loads pre-built `.bin` blobs through
//! [`load_binary`] / [`load_binary_from_bytes`]. The blobs are produced
//! by [`save_binary`] from [`SphericalHarmonicsData`] values that test
//! infrastructure assembles from JEOD's `.cc` source files.
//!
//! Parsing JEOD's `.cc` source files lives in
//! `astrodyn_gravity::jeod_cc` (a dev/test-only crate). `astrodyn_gravity`
//! itself does not know how to parse JEOD source — only how to consume
//! the binary format derived from it.

use crate::spherical_harmonics_gravity_source::{SphericalHarmonicsData, MAX_SH_DEGREE};

/// Errors from loading a binary spherical-harmonics coefficient blob.
#[derive(Debug, thiserror::Error)]
pub enum CoeffLoadError {
    /// Underlying I/O failure when reading the coefficient blob.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Blob bytes did not match the expected layout (truncated header,
    /// triangular-array length mismatch, etc.).
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
    // In-crate access to the flat triangular storage avoids the two
    // bounds asserts and the `tri_idx` recomputation that the public
    // `cnm`/`snm` accessors perform per coefficient. The flat layout is
    // indexed by `tri_idx(n, m) = n*(n+1)/2 + m`, so iterating the slice
    // linearly produces exactly the same `(n=0..=deg, m=0..=n)` order
    // the load path expects -- the on-disk bytes are byte-identical to
    // the per-element accessor path.
    let num_coeffs = (data.degree + 1) * (data.degree + 2) / 2;
    debug_assert_eq!(data.cnm.len(), num_coeffs);
    debug_assert_eq!(data.snm.len(), num_coeffs);
    buf.reserve(2 * num_coeffs * 8);
    for &c in &data.cnm[..num_coeffs] {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    for &s in &data.snm[..num_coeffs] {
        buf.extend_from_slice(&s.to_le_bytes());
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

    // Bound the declared degree against the same ceiling
    // `SphericalHarmonicsData::new` enforces, so a malformed blob
    // surfaces as a typed `CoeffLoadError::InvalidFormat` here
    // instead of panicking at construction. The downstream `new`
    // assertion remains as a sanity net for in-process constructors.
    if degree > MAX_SH_DEGREE {
        return Err(CoeffLoadError::InvalidFormat(format!(
            "degree {degree} exceeds maximum supported ({MAX_SH_DEGREE})"
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic non-trivial `SphericalHarmonicsData` of the
    /// requested degree, with `(n, m)`-dependent coefficients so the
    /// flat-slice fast path cannot accidentally produce a matching byte
    /// stream by swapping rows or zeroing values.
    fn synthesize(degree: usize) -> SphericalHarmonicsData {
        let mut cnm = Vec::with_capacity(degree + 1);
        let mut snm = Vec::with_capacity(degree + 1);
        for n in 0..=degree {
            let mut crow = Vec::with_capacity(n + 1);
            let mut srow = Vec::with_capacity(n + 1);
            for m in 0..=n {
                crow.push(1.0e-6 * (n as f64) + 1.0e-9 * (m as f64) + 0.25);
                srow.push(-2.0e-6 * (n as f64) + 3.0e-9 * (m as f64) - 0.125);
            }
            cnm.push(crow);
            snm.push(srow);
        }
        SphericalHarmonicsData::new(
            degree,
            degree,
            6_378_137.0,
            3.986_004_418e14,
            cnm,
            snm,
            true,
            1.39e-8,
        )
    }

    /// Gate for the flat-slice serialization fast path: writing via the
    /// flat slice must produce the exact same on-disk bytes as the
    /// previous per-element accessor path and must round-trip back to
    /// numerically identical coefficient arrays via
    /// `load_binary_from_bytes`.
    #[test]
    fn save_load_round_trip_byte_identical() {
        for &degree in &[1_usize, 4, 16, 32] {
            let data = synthesize(degree);

            // Write via the production fast path. `NamedTempFile` gives
            // every test invocation a unique path and a drop guard, so
            // concurrent nextest runs cannot collide on the same file
            // and a panicking test still cleans up after itself.
            let tmp = tempfile::NamedTempFile::new().expect("NamedTempFile");
            save_binary(&data, tmp.path()).expect("save_binary");
            let bytes_fast = std::fs::read(tmp.path()).expect("read tmp");

            // Recreate what the per-element accessor path would emit, so
            // any divergence (row order, endianness, missing element)
            // would surface as a byte mismatch on this exact assertion.
            let mut bytes_ref = Vec::new();
            bytes_ref.extend_from_slice(b"JEOD");
            bytes_ref.extend_from_slice(&1u32.to_le_bytes());
            bytes_ref.extend_from_slice(&(data.degree as u32).to_le_bytes());
            bytes_ref.extend_from_slice(&(data.order as u32).to_le_bytes());
            bytes_ref.extend_from_slice(&data.radius.to_le_bytes());
            bytes_ref.extend_from_slice(&data.mu.to_le_bytes());
            bytes_ref.push(if data.tide_free { 1 } else { 0 });
            bytes_ref.extend_from_slice(&data.tide_free_delta.to_le_bytes());
            for n in 0..=data.degree {
                for m in 0..=n {
                    bytes_ref.extend_from_slice(&data.cnm(n, m).to_le_bytes());
                }
            }
            for n in 0..=data.degree {
                for m in 0..=n {
                    bytes_ref.extend_from_slice(&data.snm(n, m).to_le_bytes());
                }
            }
            assert_eq!(
                bytes_fast, bytes_ref,
                "flat-slice fast path must produce bytes identical to per-element accessor path (degree {degree})"
            );

            // And the bytes must round-trip back into the same model.
            let reloaded = load_binary_from_bytes(&bytes_fast).expect("load_binary_from_bytes");
            assert_eq!(reloaded.degree, data.degree);
            assert_eq!(reloaded.order, data.order);
            assert_eq!(reloaded.radius.to_bits(), data.radius.to_bits());
            assert_eq!(reloaded.mu.to_bits(), data.mu.to_bits());
            assert_eq!(reloaded.tide_free, data.tide_free);
            assert_eq!(
                reloaded.tide_free_delta.to_bits(),
                data.tide_free_delta.to_bits()
            );
            for n in 0..=degree {
                for m in 0..=n {
                    assert_eq!(
                        reloaded.cnm(n, m).to_bits(),
                        data.cnm(n, m).to_bits(),
                        "cnm({n},{m}) mismatch after round trip"
                    );
                    assert_eq!(
                        reloaded.snm(n, m).to_bits(),
                        data.snm(n, m).to_bits(),
                        "snm({n},{m}) mismatch after round trip"
                    );
                }
            }
        }
    }
}
