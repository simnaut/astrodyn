//! Gravity coefficient loading from JEOD C++ data files.

use crate::spherical_harmonics_gravity_source::SphericalHarmonicsData;

/// Errors from loading gravity coefficient files.
#[derive(Debug, thiserror::Error)]
pub enum CoeffLoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing field '{field}' in {path}")]
    MissingField { field: &'static str, path: String },
    #[error("invalid binary format: {0}")]
    InvalidFormat(String),
}

/// Load spherical harmonics coefficients from a JEOD C++ data file.
///
/// Parses files like `earth_GGM05C.cc` that contain lines of the form:
/// ```text
/// ...->degree = 360;
/// ...->order = 360;
/// ...->mu = 398600.44150E+09;
/// ...->radius = 6378136.30;
/// ...->tide_free = false;
/// ...->tide_free_delta = 4.173E-9;
/// ...->Cnm[2] = JEOD_ALLOC_PRIM_ARRAY(3, double);
/// ...->Cnm[2][0] = -4.8416945732000E-04;
/// ...->Snm[2][0] = 0.0;
/// ```
pub fn load_from_jeod_cc(path: &std::path::Path) -> Result<SphericalHarmonicsData, CoeffLoadError> {
    let content = std::fs::read_to_string(path)?;
    let path_str = path.display().to_string();

    let mut degree: Option<usize> = None;
    let mut order: Option<usize> = None;
    let mut mu: Option<f64> = None;
    let mut radius: Option<f64> = None;
    let mut tide_free: Option<bool> = None;
    let mut tide_free_delta: Option<f64> = None;

    // First pass: extract metadata
    for line in content.lines() {
        let line = line.trim();
        if let Some(val) = extract_assign_usize(line, "degree") {
            degree = Some(val);
        }
        if let Some(val) = extract_assign_usize(line, "order") {
            order = Some(val);
        }
        if let Some(val) = extract_assign_f64(line, "mu") {
            mu = Some(val);
        }
        if let Some(val) = extract_assign_f64(line, "radius") {
            radius = Some(val);
        }
        if line.contains("tide_free") && !line.contains("tide_free_delta") {
            if line.contains("true") {
                tide_free = Some(true);
            } else if line.contains("false") {
                tide_free = Some(false);
            }
        }
        if let Some(val) = extract_assign_f64(line, "tide_free_delta") {
            tide_free_delta = Some(val);
        }
    }

    let degree = degree.ok_or_else(|| CoeffLoadError::MissingField {
        field: "degree",
        path: path_str.clone(),
    })?;
    let order = order.ok_or_else(|| CoeffLoadError::MissingField {
        field: "order",
        path: path_str.clone(),
    })?;
    let mu = mu.ok_or_else(|| CoeffLoadError::MissingField {
        field: "mu",
        path: path_str.clone(),
    })?;
    let radius = radius.ok_or(CoeffLoadError::MissingField {
        field: "radius",
        path: path_str,
    })?;
    let tide_free = tide_free.unwrap_or(true);
    let tide_free_delta = tide_free_delta.unwrap_or(0.0);

    // Allocate coefficient arrays
    let mut cnm: Vec<Vec<f64>> = Vec::with_capacity(degree + 1);
    let mut snm: Vec<Vec<f64>> = Vec::with_capacity(degree + 1);
    for n in 0..=degree {
        cnm.push(vec![0.0; n + 1]);
        snm.push(vec![0.0; n + 1]);
    }

    // Second pass: extract coefficients
    // Patterns: ->Cnm[n][m] = value; and ->Snm[n][m] = value;
    for line in content.lines() {
        let line = line.trim();
        if let Some((n, m, val)) = extract_coeff(line, "Cnm") {
            if n <= degree && m <= n {
                cnm[n][m] = val;
            }
        }
        if let Some((n, m, val)) = extract_coeff(line, "Snm") {
            if n <= degree && m <= n {
                snm[n][m] = val;
            }
        }
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

/// Save coefficients to a compact binary format.
///
/// Format: degree(u32), order(u32), radius(f64), mu(f64),
///         tide_free(u8), tide_free_delta(f64),
///         then for each n=0..degree: cnm[n][0..n] as f64,
///         then for each n=0..degree: snm[n][0..n] as f64.
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
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != 1 {
        return Err(CoeffLoadError::InvalidFormat(format!(
            "unsupported binary coefficient version {version}"
        )));
    }
    pos += 4;

    let read_u32 = |pos: &mut usize| -> u32 {
        let val = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        val
    };
    let read_f64 = |pos: &mut usize| -> f64 {
        let val = f64::from_le_bytes(buf[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        val
    };

    let degree = read_u32(&mut pos) as usize;
    let order = read_u32(&mut pos) as usize;
    let radius = read_f64(&mut pos);
    let mu = read_f64(&mut pos);
    let tide_free = buf[pos] != 0;
    pos += 1;
    let tide_free_delta = read_f64(&mut pos);

    let mut cnm = Vec::with_capacity(degree + 1);
    for n in 0..=degree {
        let mut row = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            row.push(read_f64(&mut pos));
        }
        cnm.push(row);
    }

    let mut snm = Vec::with_capacity(degree + 1);
    for n in 0..=degree {
        let mut row = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            row.push(read_f64(&mut pos));
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

// --- Helper functions ---

fn extract_assign_usize(line: &str, key: &str) -> Option<usize> {
    // Match: ->key = value; or ptr->key = value;
    let pattern = format!("->{} = ", key);
    if let Some(idx) = line.find(&pattern) {
        let rest = &line[idx + pattern.len()..];
        let val_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        val_str.parse().ok()
    } else {
        None
    }
}

fn extract_assign_f64(line: &str, key: &str) -> Option<f64> {
    // Match: ->key = value; with possible scientific notation
    let pattern = format!("->{} = ", key);
    if let Some(idx) = line.find(&pattern) {
        let rest = &line[idx + pattern.len()..];
        let val_str: String = rest
            .chars()
            .take_while(|c| {
                *c == '-' || *c == '+' || *c == '.' || *c == 'E' || *c == 'e' || c.is_ascii_digit()
            })
            .collect();
        val_str.parse().ok()
    } else {
        None
    }
}

fn extract_coeff(line: &str, name: &str) -> Option<(usize, usize, f64)> {
    // Match: ->Cnm[n][m] = value;
    let prefix = format!("{}[", name);
    let start = line.find(&prefix)?;
    let rest = &line[start + prefix.len()..];

    // Parse n
    let n_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n: usize = n_str.parse().ok()?;

    // Find ][m]
    let bracket_pos = rest.find("][")?;
    let after_bracket = &rest[bracket_pos + 2..];
    let m_str: String = after_bracket
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let m: usize = m_str.parse().ok()?;

    // Find = value
    let eq_pos = after_bracket.find('=')?;
    let val_rest = after_bracket[eq_pos + 1..].trim();
    let val_str: String = val_rest
        .chars()
        .take_while(|c| {
            *c == '-' || *c == '+' || *c == '.' || *c == 'E' || *c == 'e' || c.is_ascii_digit()
        })
        .collect();
    let val: f64 = val_str.parse().ok()?;

    Some((n, m, val))
}
