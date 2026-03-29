use regex::Regex;

/// A test case extracted from JEOD's `euler_derived_state_ut.cc`.
///
/// Contains a rotation matrix and expected Euler angles (in degrees) for
/// both reference-to-body and body-to-reference decompositions.
#[derive(Debug, Clone)]
pub struct EulerTestCase {
    pub matrix: [[f64; 3]; 3],         // row-major 3x3
    pub ref_body_angles_deg: [f64; 3], // expected ref->body angles (degrees)
    pub body_ref_angles_deg: [f64; 3], // expected body->ref angles (degrees)
}

/// Load Euler angle test cases from JEOD's `euler_derived_state_ut.cc`.
///
/// Parses the C++ unit test file to extract:
/// - The 3x3 rotation matrix
/// - Expected reference-to-body Euler angles (degrees)
/// - Expected body-to-reference Euler angles (degrees)
///
/// The test data has these known values (from the Roll-Pitch-Yaw sequence test):
/// ```text
/// matrix = {{0.3535533905932738, 0.9267766952966369, 0.1268264840443220},
///           {-0.6123724356957946, 0.1268264840443223, 0.7803300858899106},
///           {0.7071067811865475, -0.3535533905932737, 0.6123724356957946}}
/// exp_ref_body_angles = {30, 45.0, 60.0}
/// exp_body_ref_angles = {-51.8765682554021907, 7.2862451871156360, -69.1187903196461093}
/// ```
///
/// # Panics
/// Panics if the file cannot be read or parsed.
pub fn load_euler_test_cases(jeod_root: &std::path::Path) -> Vec<EulerTestCase> {
    let path =
        jeod_root.join("models/dynamics/derived_state/verif/unit_tests/euler_derived_state_ut.cc");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));

    // Parse double arrays like { value1, value2, value3 }
    let array3_re =
        Regex::new(r"\{\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\}").unwrap();

    // Parse matrix rows: look for the double matrix[3][3] = { ... } pattern
    let matrix_re = Regex::new(r"double\s+matrix\[3\]\[3\]\s*=\s*\{([^;]+)\};").unwrap();

    // Parse expected ref_body angles
    let ref_body_re = Regex::new(r"double\s+exp_ref_body_angles\[3\]\s*=\s*\{([^}]+)\}").unwrap();

    // Parse expected body_ref angles
    let body_ref_re = Regex::new(r"double\s+exp_body_ref_angles\[3\]\s*=\s*\{([^}]+)\}").unwrap();

    let mut cases = Vec::new();

    // Find all matrix definitions in the file
    let matrix_matches: Vec<_> = matrix_re.captures_iter(&content).collect();
    let ref_body_matches: Vec<_> = ref_body_re.captures_iter(&content).collect();
    let body_ref_matches: Vec<_> = body_ref_re.captures_iter(&content).collect();

    // Each test block has one matrix, one ref_body_angles, and one body_ref_angles.
    // The file repeats the same values in multiple TEST blocks; we deduplicate.
    let num_cases = matrix_matches
        .len()
        .min(ref_body_matches.len())
        .min(body_ref_matches.len());

    for i in 0..num_cases {
        // Parse matrix rows
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

        // Parse ref_body angles
        let rb_text = &ref_body_matches[i][1];
        let rb_vals: Vec<f64> = rb_text
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();

        // Parse body_ref angles
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

    // Deduplicate identical cases (the same matrix/angles appear in multiple TEST blocks)
    cases.dedup_by(|a, b| {
        a.matrix == b.matrix
            && a.ref_body_angles_deg == b.ref_body_angles_deg
            && a.body_ref_angles_deg == b.body_ref_angles_deg
    });

    cases
}
