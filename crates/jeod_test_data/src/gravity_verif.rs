//! Parser for JEOD's static gravity-verification reference data.
//!
//! Reads
//! [`models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt)
//! from JEOD v5.4.0 — the 40 case-by-case `(position →
//! acceleration / gradient / potential)` test vectors used by Tier 2
//! `tier2_grav_geospherical`.

use glam::{DMat3, DVec3};

/// A single test case from JEOD's grav_geospherical verification data.
///
/// Parsed from `models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt`.
/// Each line contains 18 space-separated fields:
///   `CaseNum Degree Order PerturbOnly GradActive Pos[3] Potential Accel[3] Gradient[6]`.
///
/// The gradient is stored as the upper triangle of a symmetric 3x3 matrix:
///   `[0,0], [0,1], [0,2], [1,1], [1,2], [2,2]`.
#[derive(Debug, Clone)]
pub struct GravityTestCase {
    /// 1-based case number from the JEOD verif file.
    pub case_num: usize,
    /// Spherical-harmonics degree exercised by this case.
    pub degree: usize,
    /// Spherical-harmonics order exercised by this case.
    pub order: usize,
    /// Whether the case uses perturbing-only (no `n=0,1`) evaluation.
    pub perturb_only: bool,
    /// Whether the case computes the gravity-gradient tensor.
    pub grad_active: bool,
    /// Probe position in metres (planet-fixed frame).
    pub position: DVec3,
    /// Expected gravitational potential (`m²/s²`).
    pub potential: f64,
    /// Expected gravitational acceleration in m/s².
    pub acceleration: DVec3,
    /// Expected gravity-gradient tensor (full symmetric matrix).
    pub gradient: DMat3,
}

/// Load all gravity test cases from the committed
/// `test_data/gravity/grav_geospherical_verif_out.txt` fixture.
///
/// The fixture is a verbatim copy of JEOD's
/// `models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt`
/// (40 lines, plain text, diff-friendly). Refresh it via the
/// `extract_jeod_validation` binary after a JEOD upgrade.
///
/// # Panics
/// Panics with a fail-loudly diagnostic if the fixture is missing or
/// malformed; the message includes the regen command.
pub fn load_gravity_test_cases() -> Vec<GravityTestCase> {
    let path = crate::tier3_csv::test_data_path("gravity/grav_geospherical_verif_out.txt");
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Cannot read {}: {e}. Regenerate with: cargo run -p jeod_test_data \
             --bin extract_jeod_validation",
            path.display(),
        )
    });
    parse_gravity_test_cases_text(&content)
}

/// Parse the line-oriented `verif_out.txt` content into [`GravityTestCase`]s.
///
/// Used by both [`load_gravity_test_cases`] (committed fixture) and
/// the regen binary (`extract_jeod_validation`) when verifying that
/// JEOD upstream still matches the committed fixture.
pub fn parse_gravity_test_cases_text(content: &str) -> Vec<GravityTestCase> {
    let mut cases = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 18 {
            continue;
        }

        let parse_f64 = |s: &str| -> f64 { s.parse().unwrap() };
        let parse_usize = |s: &str| -> usize { s.parse().unwrap() };

        // Fields: case degree order perturbOnly gradActive pos[3] pot accel[3] grad[6]
        let g00 = parse_f64(fields[12]);
        let g01 = parse_f64(fields[13]);
        let g02 = parse_f64(fields[14]);
        let g11 = parse_f64(fields[15]);
        let g12 = parse_f64(fields[16]);
        let g22 = parse_f64(fields[17]);

        cases.push(GravityTestCase {
            case_num: parse_usize(fields[0]),
            degree: parse_usize(fields[1]),
            order: parse_usize(fields[2]),
            perturb_only: fields[3] == "1",
            grad_active: fields[4] == "1",
            position: DVec3::new(
                parse_f64(fields[5]),
                parse_f64(fields[6]),
                parse_f64(fields[7]),
            ),
            potential: parse_f64(fields[8]),
            acceleration: DVec3::new(
                parse_f64(fields[9]),
                parse_f64(fields[10]),
                parse_f64(fields[11]),
            ),
            gradient: DMat3::from_cols(
                DVec3::new(g00, g01, g02), // col 0
                DVec3::new(g01, g11, g12), // col 1 (symmetric)
                DVec3::new(g02, g12, g22), // col 2 (symmetric)
            ),
        });
    }
    cases
}
