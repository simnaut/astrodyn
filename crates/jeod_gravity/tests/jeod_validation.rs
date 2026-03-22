//! Validate gravity computations against JEOD's grav_geospherical test data.
//!
//! Requires the JEOD source tree (via JEOD_PATH env var or at `../jeod` as
//! sibling of the workspace root). Gated behind the `jeod-validation` feature
//! (default ON). Disable with `--no-default-features` if JEOD is unavailable.
//!
//! The 40 cases in verif_out.txt use degree=20, order=20 spherical harmonics.
//! 33 cases have perturbOnly=0 (total gravity), 7 have perturbOnly=1
//! (harmonics perturbation only, i.e. total minus point-mass).
#![cfg(feature = "jeod-validation")]

use jeod_test_data::{gravity_verif::load_gravity_test_cases, jeod_path};

#[test]
fn load_gravity_test_data() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let cases = load_gravity_test_cases(&root);
    assert!(!cases.is_empty(), "Expected at least one gravity test case");
    assert_eq!(cases.len(), 40, "Expected 40 gravity test cases, got {}", cases.len());

    // Cases use various degree/order combinations (up to 20x20).
    // Verify all have valid degree >= order >= 0.
    for case in &cases {
        assert!(
            case.degree >= case.order,
            "Case {}: degree {} < order {}",
            case.case_num, case.degree, case.order
        );
    }

    // Verify the perturbOnly distribution
    let perturb_count = cases.iter().filter(|c| c.perturb_only).count();
    let full_count = cases.iter().filter(|c| !c.perturb_only).count();
    assert!(perturb_count > 0, "Expected some perturbOnly cases");
    assert!(full_count > 0, "Expected some full gravity cases");
    assert_eq!(perturb_count + full_count, 40);
}

#[test]
fn jeod_gravity_data_laplace_equation() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let cases = load_gravity_test_cases(&root);

    // For all cases with grad_active=true, the gravity gradient tensor
    // should satisfy Laplace's equation: trace(G) ~ 0 outside the body.
    for case in &cases {
        if !case.grad_active {
            continue;
        }

        let g = case.gradient;
        let trace = g.col(0)[0] + g.col(1)[1] + g.col(2)[2];

        assert!(
            trace.abs() < 1e-15,
            "Case {}: Laplace violated, trace = {:.6e}",
            case.case_num, trace,
        );
    }
}

#[test]
fn point_mass_reasonable_at_jeod_positions() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let cases = load_gravity_test_cases(&root);
    let mu_earth = 3.986004418e14;

    for case in &cases {
        let result = jeod_gravity::compute_point_mass_gravity(mu_earth, case.position);
        let pm_mag = result.accel.length();

        if case.perturb_only {
            // JEOD acceleration is perturbation only (total minus point-mass).
            // It should be small relative to point-mass.
            let pert_mag = case.acceleration.length();
            let ratio = pert_mag / pm_mag;
            assert!(
                ratio < 0.01,
                "Case {}: perturbation {:.6e} > 1% of point-mass {:.6e}",
                case.case_num, pert_mag, pm_mag,
            );
        } else {
            // JEOD acceleration is total gravity. Point-mass should match
            // within ~1% (J2 perturbation is ~0.1% for LEO).
            let jeod_mag = case.acceleration.length();
            let relative_diff = ((pm_mag - jeod_mag) / jeod_mag).abs();
            assert!(
                relative_diff < 0.01,
                "Case {}: point-mass {:.6e} vs JEOD total {:.6e}, diff {:.2e}",
                case.case_num, pm_mag, jeod_mag, relative_diff,
            );
        }

        // Point-mass acceleration should always point toward center.
        let cos_radial = case.position.normalize().dot(result.accel.normalize());
        assert!(
            cos_radial < -0.999,
            "Case {}: not anti-radial, cos = {:.6}",
            case.case_num, cos_radial,
        );
    }
}

#[test]
fn gradient_symmetry_in_jeod_data() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}. Run with --no-default-features to skip.", root.display());

    let cases = load_gravity_test_cases(&root);

    for case in &cases {
        if !case.grad_active {
            continue;
        }

        let g = case.gradient;
        let tol = 1e-30;

        assert!((g.col(0)[1] - g.col(1)[0]).abs() < tol, "Case {}: G[0][1] != G[1][0]", case.case_num);
        assert!((g.col(0)[2] - g.col(2)[0]).abs() < tol, "Case {}: G[0][2] != G[2][0]", case.case_num);
        assert!((g.col(1)[2] - g.col(2)[1]).abs() < tol, "Case {}: G[1][2] != G[2][1]", case.case_num);
    }
}
