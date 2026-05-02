//! JEOD C++ source file parsers for gravity coefficient files.
//!
//! These parsers exist exclusively to populate test fixtures and Tier 3
//! verification data from a JEOD source checkout (`$JEOD_HOME`).
//! Production gravity code in `jeod_gravity` consumes the compact binary
//! format (`load_binary` / `load_binary_from_bytes`) produced by
//! `save_binary`, which is generated from these parsers as a one-time
//! test-data step.
//!
//! Parsing JEOD's `.cc` files is therefore a *test/data* concern, not a
//! production capability — keeping it here preserves the principle that
//! `jeod_gravity` is a self-contained physics crate with no knowledge of
//! the original C++ source layout.

use std::path::Path;

use jeod_gravity::SphericalHarmonicsData;

/// Errors from parsing a JEOD C++ gravity data file.
#[derive(Debug, thiserror::Error)]
pub enum CoeffLoadError {
    /// Underlying I/O failure when reading the `.cc` source file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Expected field (e.g. `mu`, `radius`, a `Cnm` row) was missing
    /// from the parsed source file.
    #[error("missing field '{field}' in {path}")]
    MissingField {
        /// Name of the expected field that was not found.
        field: &'static str,
        /// Source-file path that was being parsed.
        path: String,
    },
}

/// Load only the gravitational parameter (mu) from a JEOD C++ gravity data file.
///
/// Works with any JEOD gravity file including spherical-only files (like
/// `sun_spherical.cc`, `moon_spherical.cc`) that lack `degree`/`order` fields.
///
/// Returns mu in m³/s².
pub fn load_mu_from_jeod_cc(path: &Path) -> Result<f64, CoeffLoadError> {
    let content = std::fs::read_to_string(path)?;
    let path_str = path.display().to_string();

    for line in content.lines() {
        if let Some(val) = extract_assign_f64(line.trim(), "mu") {
            return Ok(val);
        }
    }

    Err(CoeffLoadError::MissingField {
        field: "mu",
        path: path_str,
    })
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
pub fn load_from_jeod_cc(path: &Path) -> Result<SphericalHarmonicsData, CoeffLoadError> {
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
    // Match: ->key = value; with possible scientific notation or multiplication.
    // Handles: `->mu = 398600.44150E+09;`
    //          `->mu = 1000000000 * (4902.801076);`
    let pattern = format!("->{} = ", key);
    if let Some(idx) = line.find(&pattern) {
        let rest = &line[idx + pattern.len()..];
        // Extract up to semicolon, then evaluate
        let expr: String = rest.chars().take_while(|c| *c != ';').collect();
        eval_simple_expr(&expr)
    } else {
        None
    }
}

/// Evaluate a simple numeric expression: a single f64, or `a * (b)` multiplication.
fn eval_simple_expr(expr: &str) -> Option<f64> {
    let expr = expr.trim();
    // Try direct parse first
    if let Ok(val) = expr.parse::<f64>() {
        return Some(val);
    }
    // Try multiplication: "A * (B)" or "A * B"
    if let Some(star_idx) = expr.find('*') {
        let lhs = expr[..star_idx].trim();
        let rhs = expr[star_idx + 1..]
            .trim()
            .trim_matches(|c| c == '(' || c == ')');
        if let (Ok(a), Ok(b)) = (lhs.parse::<f64>(), rhs.trim().parse::<f64>()) {
            return Some(a * b);
        }
    }
    None
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
