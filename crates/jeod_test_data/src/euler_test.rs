//! Euler-angle / rotation-matrix verification cases from JEOD's
//! `euler_derived_state_ut.cc`.
//!
//! The JEOD source file repeats the same matrix + angle triples across
//! multiple `TEST(EulerDerivedState, ...)` blocks; after dedup there is
//! one unique case (the Roll-Pitch-Yaw sequence test). The runtime path
//! reads it from a committed JSON fixture; the C++/regex parser remains
//! as `parse_euler_test_cases_cc` for the regen binary
//! (`extract_jeod_validation`).

use regex::Regex;

/// A test case extracted from JEOD's `euler_derived_state_ut.cc`.
///
/// Contains a rotation matrix and expected Euler angles (in degrees) for
/// both reference-to-body and body-to-reference decompositions.
#[derive(Debug, Clone, PartialEq)]
pub struct EulerTestCase {
    /// Row-major 3×3 rotation matrix.
    pub matrix: [[f64; 3]; 3],
    /// Expected reference-to-body Roll-Pitch-Yaw angles in degrees.
    pub ref_body_angles_deg: [f64; 3],
    /// Expected body-to-reference Roll-Pitch-Yaw angles in degrees.
    pub body_ref_angles_deg: [f64; 3],
}

/// Load the Euler-angle test cases from the committed JSON fixture at
/// `test_data/jeod_validation/euler_cases.json`.
///
/// The fixture is pre-extracted from JEOD's `euler_derived_state_ut.cc`
/// by the `extract_jeod_validation` binary; callers do not need
/// `JEOD_HOME` set.
///
/// # Panics
/// Panics with a fail-loudly diagnostic if the fixture is missing or
/// malformed; the message includes the regen command.
pub fn load_euler_test_cases() -> Vec<EulerTestCase> {
    let path = crate::tier3_csv::test_data_path("jeod_validation/euler_cases.json");
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Cannot read {}: {e}. Regenerate with: cargo run -p jeod_test_data \
             --bin extract_jeod_validation",
            path.display(),
        )
    });
    parse_euler_cases_json(&content).unwrap_or_else(|msg| {
        panic!(
            "Malformed Euler-cases fixture at {}: {msg}. Regenerate with: \
             cargo run -p jeod_test_data --bin extract_jeod_validation",
            path.display(),
        )
    })
}

/// Decode the JSON blob produced by `extract_jeod_validation`.
///
/// Hand-rolled (no `serde_json` dep) to match the no-`serde_json` style
/// used elsewhere in this crate (see `body_init_fixtures.rs`,
/// `planet_geodetic_verif.rs`).
pub fn parse_euler_cases_json(s: &str) -> Result<Vec<EulerTestCase>, String> {
    let cases_text = extract_array_field(s, "cases")
        .ok_or_else(|| "missing top-level \"cases\" array".to_string())?;
    let entries = split_top_level_objects(cases_text);
    let mut cases = Vec::with_capacity(entries.len());
    for entry in entries {
        let matrix = extract_matrix3x3(entry, "matrix")
            .ok_or_else(|| "case missing \"matrix\" 3x3".to_string())?;
        let rb = extract_array3(entry, "ref_body_angles_deg")
            .ok_or_else(|| "case missing \"ref_body_angles_deg\"".to_string())?;
        let br = extract_array3(entry, "body_ref_angles_deg")
            .ok_or_else(|| "case missing \"body_ref_angles_deg\"".to_string())?;
        cases.push(EulerTestCase {
            matrix,
            ref_body_angles_deg: rb,
            body_ref_angles_deg: br,
        });
    }
    Ok(cases)
}

/// Encode the JSON blob consumed by [`load_euler_test_cases`].
///
/// Public for the regen binary; runtime callers should not invoke this.
pub fn encode_euler_cases_json(cases: &[EulerTestCase]) -> String {
    let mut buf = String::new();
    buf.push_str("{\n  \"source\": \"models/dynamics/derived_state/verif/unit_tests/euler_derived_state_ut.cc\",\n");
    buf.push_str("  \"note\": \"All TEST(EulerDerivedState, ...) blocks share these values; deduplicated to one entry. Regenerate with: cargo run -p jeod_test_data --bin extract_jeod_validation\",\n");
    buf.push_str("  \"cases\": [\n");
    for (i, c) in cases.iter().enumerate() {
        buf.push_str("    {\n");
        buf.push_str("      \"matrix\": [\n");
        for (r, row) in c.matrix.iter().enumerate() {
            buf.push_str(&format!(
                "        [{:.18e}, {:.18e}, {:.18e}]{}\n",
                row[0],
                row[1],
                row[2],
                if r < 2 { "," } else { "" },
            ));
        }
        buf.push_str("      ],\n");
        buf.push_str(&format!(
            "      \"ref_body_angles_deg\": [{:.18e}, {:.18e}, {:.18e}],\n",
            c.ref_body_angles_deg[0], c.ref_body_angles_deg[1], c.ref_body_angles_deg[2],
        ));
        buf.push_str(&format!(
            "      \"body_ref_angles_deg\": [{:.18e}, {:.18e}, {:.18e}]\n",
            c.body_ref_angles_deg[0], c.body_ref_angles_deg[1], c.body_ref_angles_deg[2],
        ));
        buf.push_str(if i + 1 < cases.len() {
            "    },\n"
        } else {
            "    }\n"
        });
    }
    buf.push_str("  ]\n}\n");
    buf
}

/// Parse JEOD's `euler_derived_state_ut.cc` content, returning the
/// deduplicated list of test cases.
///
/// **Regen-only** path: invoked by `extract_jeod_validation` to produce
/// the committed JSON fixture. Runtime callers should use
/// [`load_euler_test_cases`].
pub fn parse_euler_test_cases_cc(content: &str) -> Vec<EulerTestCase> {
    let array3_re =
        Regex::new(r"\{\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\}").unwrap();
    let matrix_re = Regex::new(r"double\s+matrix\[3\]\[3\]\s*=\s*\{([^;]+)\};").unwrap();
    let ref_body_re = Regex::new(r"double\s+exp_ref_body_angles\[3\]\s*=\s*\{([^}]+)\}").unwrap();
    let body_ref_re = Regex::new(r"double\s+exp_body_ref_angles\[3\]\s*=\s*\{([^}]+)\}").unwrap();

    let mut cases = Vec::new();
    let matrix_matches: Vec<_> = matrix_re.captures_iter(content).collect();
    let ref_body_matches: Vec<_> = ref_body_re.captures_iter(content).collect();
    let body_ref_matches: Vec<_> = body_ref_re.captures_iter(content).collect();

    let num_cases = matrix_matches
        .len()
        .min(ref_body_matches.len())
        .min(body_ref_matches.len());

    for i in 0..num_cases {
        let matrix_text = &matrix_matches[i][1];
        let rows: Vec<_> = array3_re.captures_iter(matrix_text).collect();
        if rows.len() < 3 {
            continue;
        }

        let mut matrix = [[0.0_f64; 3]; 3];
        for (r, row_cap) in rows.iter().enumerate().take(3) {
            matrix[r][0] = row_cap[1].parse().unwrap();
            matrix[r][1] = row_cap[2].parse().unwrap();
            matrix[r][2] = row_cap[3].parse().unwrap();
        }

        let rb_text = &ref_body_matches[i][1];
        let rb_vals: Vec<f64> = rb_text
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();

        let br_text = &body_ref_matches[i][1];
        let br_vals: Vec<f64> = br_text
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();

        if rb_vals.len() >= 3 && br_vals.len() >= 3 {
            cases.push(EulerTestCase {
                matrix,
                ref_body_angles_deg: [rb_vals[0], rb_vals[1], rb_vals[2]],
                body_ref_angles_deg: [br_vals[0], br_vals[1], br_vals[2]],
            });
        }
    }

    cases.dedup();
    cases
}

// ── JSON parsing helpers (no serde_json dep) ──

fn extract_array_field<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let key_idx = s.find(&needle)?;
    let after_key = &s[key_idx + needle.len()..];
    let colon_off = after_key.find(':')?;
    let after_colon = &after_key[colon_off + 1..];
    let bracket_off = after_colon.find('[')?;
    let inner_start = bracket_off + 1;
    let mut depth = 1;
    for (i, c) in after_colon[inner_start..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&after_colon[inner_start..inner_start + i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level_objects(s: &str) -> Vec<&str> {
    let mut entries = Vec::new();
    let mut depth = 0;
    let mut start: Option<usize> = None;
    for (i, c) in s.char_indices() {
        match c {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s0) = start.take() {
                        entries.push(&s[s0..i + 1]);
                    }
                }
            }
            _ => {}
        }
    }
    entries
}

fn extract_array3(s: &str, key: &str) -> Option<[f64; 3]> {
    let needle = format!("\"{key}\"");
    let key_idx = s.find(&needle)?;
    let after = &s[key_idx + needle.len()..];
    let bracket = after.find('[')?;
    let close = after[bracket..].find(']')?;
    let inner = &after[bracket + 1..bracket + close];
    let parts: Vec<f64> = inner
        .split(',')
        .map(|p| p.trim().parse::<f64>().ok())
        .collect::<Option<Vec<_>>>()?;
    if parts.len() != 3 {
        return None;
    }
    Some([parts[0], parts[1], parts[2]])
}

fn extract_matrix3x3(s: &str, key: &str) -> Option<[[f64; 3]; 3]> {
    let needle = format!("\"{key}\"");
    let key_idx = s.find(&needle)?;
    let after = &s[key_idx + needle.len()..];
    // Find the outer `[` opening the matrix, then split inner rows.
    let outer = after.find('[')?;
    let mut depth = 0;
    let mut close = None;
    for (i, c) in after[outer..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(outer + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let inner = &after[outer + 1..close];
    let mut rows: Vec<[f64; 3]> = Vec::new();
    let mut row_depth = 0;
    let mut row_start: Option<usize> = None;
    for (i, c) in inner.char_indices() {
        match c {
            '[' => {
                if row_depth == 0 {
                    row_start = Some(i);
                }
                row_depth += 1;
            }
            ']' => {
                row_depth -= 1;
                if row_depth == 0 {
                    if let Some(s0) = row_start.take() {
                        let row_text = &inner[s0 + 1..i];
                        let parts: Vec<f64> = row_text
                            .split(',')
                            .map(|p| p.trim().parse::<f64>().ok())
                            .collect::<Option<Vec<_>>>()?;
                        if parts.len() != 3 {
                            return None;
                        }
                        rows.push([parts[0], parts[1], parts[2]]);
                    }
                }
            }
            _ => {}
        }
    }
    if rows.len() != 3 {
        return None;
    }
    Some([rows[0], rows[1], rows[2]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let original = vec![EulerTestCase {
            matrix: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            ref_body_angles_deg: [30.0, 45.0, 60.0],
            body_ref_angles_deg: [-1.0, 2.0, -3.0],
        }];
        let encoded = encode_euler_cases_json(&original);
        let decoded = parse_euler_cases_json(&encoded).unwrap();
        assert_eq!(decoded, original);
    }
}
