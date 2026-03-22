//! Validate gravity computations against JEOD's grav_geospherical test data.
//!
//! Only runs when the JEOD source tree is available (via JEOD_PATH env var
//! or at `../../jeod` relative to the workspace root).
//!
//! NOTE: All 40 cases in verif_out.txt use degree=20, order=20 (spherical
//! harmonics). There are no degree=0 cases, so we cannot directly compare
//! point-mass output against this file. We still validate:
//!   1. The parser works and produces the expected number of cases.
//!   2. Point-mass gravity is a reasonable approximation at the same positions
//!      (correct order-of-magnitude, correct direction).
//!   3. The gradient from verif_out.txt satisfies Laplace's equation (trace ~ 0)
//!      which validates the JEOD data itself and our parser.

use jeod_test_data::{gravity_verif::load_gravity_test_cases, jeod_path};

#[test]
fn load_gravity_test_data() {
    let root = jeod_path();
    if !root.exists() {
        eprintln!("JEOD source tree not found, skipping gravity validation tests");
        return;
    }

    let cases = load_gravity_test_cases(&root);
    assert!(
        !cases.is_empty(),
        "Expected at least one gravity test case"
    );

    // The file should have 40 cases
    assert_eq!(
        cases.len(),
        40,
        "Expected 40 gravity test cases, got {}",
        cases.len()
    );

    // All cases should have degree=20, order=20
    for case in &cases {
        assert_eq!(case.degree, 20, "Case {}: unexpected degree", case.case_num);
        assert_eq!(case.order, 20, "Case {}: unexpected order", case.case_num);
    }
}

#[test]
fn jeod_gravity_data_laplace_equation() {
    let root = jeod_path();
    if !root.exists() {
        eprintln!("JEOD source tree not found, skipping");
        return;
    }

    let cases = load_gravity_test_cases(&root);

    // For all cases with grad_active=true, the gravity gradient tensor
    // should satisfy Laplace's equation: trace(G) ~ 0 outside the body.
    // This validates both the JEOD data and our parser's gradient assembly.
    for case in &cases {
        if !case.grad_active {
            continue;
        }

        let g = case.gradient;
        // trace = G[0][0] + G[1][1] + G[2][2]
        let trace = g.col(0)[0] + g.col(1)[1] + g.col(2)[2];

        // The gradient values are on the order of 1e-9, so a tolerance
        // of 1e-15 is relative precision of ~1e-6.
        assert!(
            trace.abs() < 1e-15,
            "Case {}: Laplace equation violated, trace = {:.6e} (gradient diagonal: [{:.6e}, {:.6e}, {:.6e}])",
            case.case_num,
            trace,
            g.col(0)[0],
            g.col(1)[1],
            g.col(2)[2],
        );
    }
}

#[test]
fn point_mass_reasonable_at_jeod_positions() {
    let root = jeod_path();
    if !root.exists() {
        eprintln!("JEOD source tree not found, skipping");
        return;
    }

    let cases = load_gravity_test_cases(&root);
    let mu_earth = 3.986004418e14;

    for case in &cases {
        // Point-mass should give the correct order-of-magnitude acceleration
        // at the same position, even though the JEOD data includes harmonics.
        let result = jeod_gravity::compute_point_mass_gravity(mu_earth, case.position);

        // The acceleration magnitudes should be within a factor of 2
        // (spherical harmonics perturbations are small relative to point-mass)
        let pm_mag = result.accel.length();
        let jeod_mag = case.acceleration.length();

        if jeod_mag > 0.0 {
            let ratio = pm_mag / jeod_mag;
            assert!(
                (0.5..2.0).contains(&ratio),
                "Case {}: point-mass accel magnitude {:.6e} differs too much from JEOD {:.6e} (ratio {:.4})",
                case.case_num,
                pm_mag,
                jeod_mag,
                ratio,
            );
        }

        // The acceleration direction should be roughly anti-parallel to position
        // (both point-mass and harmonics-dominated gravity point roughly toward center)
        let accel_dir = result.accel.normalize();
        let pos_dir = case.position.normalize();
        let dot = accel_dir.dot(pos_dir);
        assert!(
            dot < -0.9,
            "Case {}: point-mass accel not roughly anti-parallel to position (dot = {:.4})",
            case.case_num,
            dot,
        );
    }
}

#[test]
fn gradient_symmetry_in_jeod_data() {
    let root = jeod_path();
    if !root.exists() {
        eprintln!("JEOD source tree not found, skipping");
        return;
    }

    let cases = load_gravity_test_cases(&root);

    // Our parser constructs the gradient as a symmetric DMat3 from the upper
    // triangle. Verify the symmetry holds in the assembled matrix.
    for case in &cases {
        if !case.grad_active {
            continue;
        }

        let g = case.gradient;
        let tol = 1e-30; // Should be exact since we copy the same value

        assert!(
            (g.col(0)[1] - g.col(1)[0]).abs() < tol,
            "Case {}: G[0][1] != G[1][0]",
            case.case_num
        );
        assert!(
            (g.col(0)[2] - g.col(2)[0]).abs() < tol,
            "Case {}: G[0][2] != G[2][0]",
            case.case_num
        );
        assert!(
            (g.col(1)[2] - g.col(2)[1]).abs() < tol,
            "Case {}: G[1][2] != G[2][1]",
            case.case_num
        );
    }
}
