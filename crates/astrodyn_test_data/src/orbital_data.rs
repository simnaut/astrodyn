//! Cartesian state vectors from JEOD's orbital-element verification data.
//!
//! Originally extracted from
//! `models/utils/orbital_elements/verif/SIM_orb_elem/Modified_data/orb_ell_in.py`
//! (~5000 records). The runtime path now reads a committed binary
//! fixture; the Python parser remains for the regen binary.

use glam::DVec3;
use regex::Regex;

/// A Cartesian state vector (position + velocity) from JEOD's orbital elements
/// verification data.
#[derive(Debug, Clone)]
pub struct CartesianStateVector {
    /// Position in meters.
    pub position: DVec3,
    /// Velocity in m/s.
    pub velocity: DVec3,
}

/// Load orbital element test vectors from the committed binary fixture
/// at `test_data/jeod_validation/orbital_vectors.bin`.
///
/// The fixture is pre-extracted from JEOD's `orb_ell_in.py` by the
/// `extract_jeod_validation` binary; callers do not need `JEOD_HOME` set.
///
/// **Format**: little-endian. `u32` count header followed by
/// `count × 6 × f64` (position x,y,z then velocity x,y,z per record).
///
/// # Panics
/// Panics with a fail-loudly diagnostic if the fixture is missing or
/// malformed; the message includes the regen command.
pub fn load_orbital_test_vectors() -> Vec<CartesianStateVector> {
    let path = crate::tier3_csv::test_data_path("jeod_validation/orbital_vectors.bin");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "Cannot read {}: {e}. Regenerate with: cargo run -p astrodyn_test_data \
             --bin extract_jeod_validation",
            path.display(),
        )
    });
    parse_orbital_vectors_bin(&bytes).unwrap_or_else(|msg| {
        panic!(
            "Malformed orbital_vectors fixture at {}: {msg}. Regenerate with: \
             cargo run -p astrodyn_test_data --bin extract_jeod_validation",
            path.display(),
        )
    })
}

/// Decode the binary blob produced by `extract_jeod_validation`.
///
/// Public for the regen binary's roundtrip self-check; runtime callers
/// should use [`load_orbital_test_vectors`].
///
/// The header `count` is validated against the blob length using
/// checked arithmetic so a corrupt/hostile header that overflows
/// `usize::MAX / 48` reports a structured error instead of panicking
/// inside the multiplication.
pub fn parse_orbital_vectors_bin(bytes: &[u8]) -> Result<Vec<CartesianStateVector>, String> {
    if bytes.len() < 4 {
        return Err(format!(
            "blob is {} bytes; need at least 4 for header",
            bytes.len()
        ));
    }
    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    // Bytes per vector = 6 components × 8-byte f64.
    let payload_bytes = count
        .checked_mul(48)
        .ok_or_else(|| format!("header count {count} overflows usize when × 48"))?;
    let expected_total = payload_bytes
        .checked_add(4)
        .ok_or_else(|| format!("header count {count} overflows usize at +4"))?;
    if bytes.len() != expected_total {
        return Err(format!(
            "header says {count} vectors ({payload_bytes} bytes) but body is {}",
            bytes.len() - 4,
        ));
    }
    let mut vectors = Vec::with_capacity(count);
    let mut pos = 4;
    for _ in 0..count {
        let mut comps = [0.0_f64; 6];
        for c in &mut comps {
            *c = f64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
        }
        vectors.push(CartesianStateVector {
            position: DVec3::new(comps[0], comps[1], comps[2]),
            velocity: DVec3::new(comps[3], comps[4], comps[5]),
        });
    }
    Ok(vectors)
}

/// Encode the binary blob consumed by [`load_orbital_test_vectors`].
///
/// Public for the regen binary; runtime callers should not invoke this.
pub fn encode_orbital_vectors_bin(vectors: &[CartesianStateVector]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + vectors.len() * 6 * 8);
    buf.extend_from_slice(&(vectors.len() as u32).to_le_bytes());
    for v in vectors {
        for c in [
            v.position.x,
            v.position.y,
            v.position.z,
            v.velocity.x,
            v.velocity.y,
            v.velocity.z,
        ] {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    buf
}

/// Parse JEOD's `orb_ell_in.py` content into Cartesian state vectors.
///
/// **Regen-only** path: invoked by `extract_jeod_validation` to produce
/// the committed binary fixture. Runtime callers should use
/// [`load_orbital_test_vectors`].
pub fn parse_orbital_test_vectors_py(content: &str) -> Vec<CartesianStateVector> {
    let array_re = Regex::new(
        r"\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]",
    )
    .unwrap();

    let mut vectors = Vec::new();
    for cap in array_re.captures_iter(content) {
        let px: f64 = cap[1].parse().unwrap();
        let py: f64 = cap[2].parse().unwrap();
        let pz: f64 = cap[3].parse().unwrap();
        let vx: f64 = cap[4].parse().unwrap();
        let vy: f64 = cap[5].parse().unwrap();
        let vz: f64 = cap[6].parse().unwrap();
        vectors.push(CartesianStateVector {
            position: DVec3::new(px, py, pz),
            velocity: DVec3::new(vx, vy, vz),
        });
    }
    vectors
}
