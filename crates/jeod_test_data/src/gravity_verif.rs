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
    pub case_num: usize,
    pub degree: usize,
    pub order: usize,
    pub perturb_only: bool,
    pub grad_active: bool,
    pub position: DVec3,
    pub potential: f64,
    pub acceleration: DVec3,
    pub gradient: DMat3, // full symmetric matrix
}

/// Load all gravity test cases from JEOD's verification output file.
///
/// # Panics
/// Panics if the file cannot be read or contains malformed data.
pub fn load_gravity_test_cases(jeod_root: &std::path::Path) -> Vec<GravityTestCase> {
    let path = jeod_root
        .join("models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

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
