//! Tier 2: `grav_geospherical` reference vectors.
//!
//! Validates `jeod_gravity` against the 40 static test cases shipped in
//! `models/environment/gravity/verif/unit_tests/grav_geospherical/data/verif_out.txt`
//! from the JEOD source tree. Each case maps a planet-fixed position
//! (degree=order=20 spherical harmonics evaluation; 33 cases full gravity,
//! 7 cases harmonics-perturbation only) to JEOD's expected acceleration,
//! potential, and gradient.
//!
//! These tests exercise `calc_spherical` / `calc_nonspherical` directly
//! against reference vectors — they do *not* propagate a trajectory.
//! Tier 3 trajectory cross-validation against propagating JEOD sims lives
//! in `crates/jeod_runner/tests/tier3_*`.
//!
//! Requires the JEOD source tree (via `JEOD_HOME` or `JEOD_PATH` env var).

use glam::{DMat3, DVec3};
use jeod_gravity::SphericalHarmonicsData;
use jeod_test_data::{
    gravity_verif::{load_gravity_test_cases, GravityTestCase},
    jeod_cc, jeod_path,
};
use std::path::{Path, PathBuf};

/// Path to the GGM02C coefficient source file in the JEOD tree (the file
/// JEOD's `grav_geospherical` test itself references).
fn ggm02c_path(root: &Path) -> PathBuf {
    root.join("models/environment/gravity/data/src/earth_GGM02C.cc")
}

#[test]
fn tier2_grav_geospherical_loader() {
    let root = jeod_path();
    assert!(
        root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        root.display()
    );

    let cases = load_gravity_test_cases(&root);
    assert!(!cases.is_empty(), "Expected at least one gravity test case");
    assert_eq!(
        cases.len(),
        40,
        "Expected 40 gravity test cases, got {}",
        cases.len()
    );

    // Cases use various degree/order combinations (up to 20x20).
    // Verify all have valid degree >= order >= 0.
    for case in &cases {
        assert!(
            case.degree >= case.order,
            "Case {}: degree {} < order {}",
            case.case_num,
            case.degree,
            case.order
        );
    }

    // Verify the perturbOnly distribution.
    let perturb_count = cases.iter().filter(|c| c.perturb_only).count();
    let full_count = cases.iter().filter(|c| !c.perturb_only).count();
    assert!(perturb_count > 0, "Expected some perturbOnly cases");
    assert!(full_count > 0, "Expected some full gravity cases");
    assert_eq!(perturb_count + full_count, 40);
}

#[test]
fn tier2_grav_geospherical_laplace() {
    let root = jeod_path();
    assert!(
        root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        root.display()
    );

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
            case.case_num,
            trace,
        );
    }
}

#[test]
fn tier2_grav_geospherical_point_mass_sanity() {
    let root = jeod_path();
    assert!(
        root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        root.display()
    );

    let cases = load_gravity_test_cases(&root);

    // Load mu directly from JEOD GGM02C (the file JEOD's grav_geospherical
    // test itself references). This matches `tier2_grav_geospherical_full_validation`
    // below and avoids a literal duplicate of the JEOD-source value.
    let data = jeod_cc::load_from_jeod_cc(&ggm02c_path(&root)).expect("load GGM02C coefficients");
    let mu_earth = data.mu;

    for case in &cases {
        let result = jeod_gravity::calc_spherical(mu_earth, case.position);
        let pm_mag = result.grav_accel.length();

        if case.perturb_only {
            // JEOD acceleration is perturbation only (total minus point-mass).
            // It should be small relative to point-mass.
            let pert_mag = case.acceleration.length();
            let ratio = pert_mag / pm_mag;
            assert!(
                ratio < 0.01,
                "Case {}: perturbation {:.6e} > 1% of point-mass {:.6e}",
                case.case_num,
                pert_mag,
                pm_mag,
            );
        } else {
            // JEOD acceleration is total gravity. Point-mass should match
            // within ~1% (J2 perturbation is ~0.1% for LEO).
            let jeod_mag = case.acceleration.length();
            let relative_diff = ((pm_mag - jeod_mag) / jeod_mag).abs();
            assert!(
                relative_diff < 0.01,
                "Case {}: point-mass {:.6e} vs JEOD total {:.6e}, diff {:.2e}",
                case.case_num,
                pm_mag,
                jeod_mag,
                relative_diff,
            );
        }

        // Point-mass acceleration should always point toward center.
        let cos_radial = case.position.normalize().dot(result.grav_accel.normalize());
        assert!(
            cos_radial < -0.999,
            "Case {}: not anti-radial, cos = {:.6}",
            case.case_num,
            cos_radial,
        );
    }
}

#[test]
fn tier2_grav_geospherical_gradient_symmetry() {
    let root = jeod_path();
    assert!(
        root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        root.display()
    );

    let cases = load_gravity_test_cases(&root);

    for case in &cases {
        if !case.grad_active {
            continue;
        }

        let g = case.gradient;
        let tol = 1e-30;

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

/// Evaluate a single `grav_geospherical` case through the Gottlieb algorithm,
/// returning `(acceleration, potential, gradient)` in the same convention
/// as the JEOD reference vector for the case.
///
/// For `perturb_only=false` cases the result combines the point-mass and
/// non-spherical contributions; for `perturb_only=true` cases only the
/// non-spherical sum is returned. Both `calc_spherical` and `calc_nonspherical`
/// use JEOD's `+mu/r` potential convention, so the sums are direct.
fn evaluate_case(data: &SphericalHarmonicsData, case: &GravityTestCase) -> (DVec3, f64, DMat3) {
    let degree = case.degree;
    let order = case.order;
    let grad_active = case.grad_active;

    // The Gottlieb algorithm computes harmonics from n=2..degree.
    // The sums are initialized to zero (perturbing-only mode).
    let sh_result = jeod_gravity::calc_nonspherical(
        data,
        case.position,
        degree,
        order,
        grad_active,
        if grad_active { degree } else { 0 },
        if grad_active { order } else { 0 },
    );

    if case.perturb_only {
        (
            sh_result.grav_accel,
            sh_result.grav_pot,
            sh_result.grav_grad,
        )
    } else {
        let pm = jeod_gravity::calc_spherical(data.mu, case.position);
        (
            sh_result.grav_accel + pm.grav_accel,
            sh_result.grav_pot + pm.grav_pot,
            if grad_active {
                sh_result.grav_grad + pm.grav_grad
            } else {
                sh_result.grav_grad
            },
        )
    }
}

/// Run all 40 gravity test vectors through the Gottlieb algorithm.
///
/// Loads `earth_GGM02C` coefficients (same as JEOD's `grav_geospherical` test)
/// and compares acceleration, potential, and gradient against the reference
/// data. The JEOD test uses identity planet-fixed rotation, so the test
/// positions are directly in planet-fixed coordinates.
#[test]
fn tier2_grav_geospherical_full_validation() {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found at {}", root.display());

    // Load GGM02C (the test was built against GGM02C, not GGM05C).
    let path = ggm02c_path(&root);
    assert!(path.exists(), "GGM02C not found at {}", path.display());
    let mut data = jeod_cc::load_from_jeod_cc(&path).expect("load GGM02C coefficients");
    // JEOD test overrides tide_free = true (main.cc line 95).
    data.tide_free = true;

    let cases = load_gravity_test_cases(&root);
    assert_eq!(cases.len(), 40);

    // JEOD's own regression tolerances: accel 1e-14, potential 100000, gradient 1e-20.
    // Potential has inherently lower precision in the Gottlieb algorithm.
    let accel_tol = 1e-10; // m/s^2 per component
    let pot_tol = 100_000.0; // m^2/s^2 (matches JEOD's own tolerance)
    let grad_tol = 1e-16; // 1/s^2 per component

    let mut max_accel_err = 0.0_f64;
    let mut max_pot_err = 0.0_f64;
    let mut max_grad_err = 0.0_f64;
    let mut passed = 0;

    for case in &cases {
        let (accel, potential, gradient) = evaluate_case(&data, case);

        // Check acceleration magnitude (full vector error, used for reporting).
        let accel_err = (accel - case.acceleration).length();
        max_accel_err = max_accel_err.max(accel_err);

        // Check potential.
        let pot_err = (potential - case.potential).abs();
        max_pot_err = max_pot_err.max(pot_err);

        // Per-component acceleration tolerance.
        let ax_err = (accel.x - case.acceleration.x).abs();
        let ay_err = (accel.y - case.acceleration.y).abs();
        let az_err = (accel.z - case.acceleration.z).abs();
        let max_comp_err = ax_err.max(ay_err).max(az_err);
        assert!(
            max_comp_err < accel_tol,
            "Case {}: accel component error {:.6e} > tol {:.6e}\n  computed: {:?}\n  expected: {:?}",
            case.case_num, max_comp_err, accel_tol, accel, case.acceleration,
        );

        assert!(
            pot_err < pot_tol,
            "Case {}: potential error {:.6e} > tol {:.6e}",
            case.case_num,
            pot_err,
            pot_tol,
        );

        // Gradient tolerance (only when active).
        if case.grad_active {
            let expected = case.gradient;
            for i in 0..3 {
                for j in 0..3 {
                    let err = (gradient.col(j)[i] - expected.col(j)[i]).abs();
                    max_grad_err = max_grad_err.max(err);
                    assert!(
                        err < grad_tol,
                        "Case {}: gradient[{}][{}] error {:.6e} > tol {:.6e}",
                        case.case_num,
                        i,
                        j,
                        err,
                        grad_tol,
                    );
                }
            }
        }

        passed += 1;
    }

    assert_eq!(passed, 40, "Expected all 40 cases to pass");
    eprintln!("  Max acceleration error: {:.6e} m/s^2", max_accel_err);
    eprintln!("  Max potential error: {:.6e} m^2/s^2", max_pot_err);
    eprintln!("  Max gradient error: {:.6e} 1/s^2", max_grad_err);
}

/// Surface-gravity sanity check at GGM02C's equatorial and polar radii.
///
/// Not part of the 40-vector reference set, but uses the same JEOD-source
/// coefficient file and serves as a coarse end-to-end check that
/// `calc_nonspherical + calc_spherical` reproduces the textbook ~9.78 /
/// ~9.83 m/s^2 surface accelerations.
#[test]
fn tier2_grav_geospherical_surface_gravity_ggm02c() {
    let root = jeod_path();
    assert!(root.exists());
    let path = ggm02c_path(&root);
    assert!(
        path.exists(),
        "GGM02C not found at {}. Requires JEOD source.",
        path.display()
    );
    let data = jeod_cc::load_from_jeod_cc(&path).expect("load GGM02C coefficients");

    // Equatorial surface.
    let pos_eq = DVec3::new(data.radius, 0.0, 0.0);
    let result_eq =
        jeod_gravity::calc_nonspherical(&data, pos_eq, data.degree, data.order, false, 0, 0);
    let pm_eq = jeod_gravity::calc_spherical(data.mu, pos_eq);
    let g_eq = (result_eq.grav_accel + pm_eq.grav_accel).length();
    assert!(
        (g_eq - 9.78).abs() < 0.1,
        "Equatorial surface gravity: {:.4} m/s^2 (expected ~9.78)",
        g_eq
    );

    // Polar surface.
    let r_pol = data.radius * (1.0 - 1.0 / 298.257223563);
    let pos_pol = DVec3::new(0.0, 0.0, r_pol);
    let result_pol =
        jeod_gravity::calc_nonspherical(&data, pos_pol, data.degree, data.order, false, 0, 0);
    let pm_pol = jeod_gravity::calc_spherical(data.mu, pos_pol);
    let g_pol = (result_pol.grav_accel + pm_pol.grav_accel).length();
    assert!(
        (g_pol - 9.83).abs() < 0.1,
        "Polar surface gravity: {:.4} m/s^2 (expected ~9.83)",
        g_pol
    );
}
