//! Validate gravity computations against JEOD's grav_geospherical test data.
//!
//! Requires the JEOD source tree (via `JEOD_HOME` or `JEOD_PATH` env var).
//! Gated behind the `jeod-validation` feature (default ON).
//! Disable with `--no-default-features` if JEOD is unavailable.
//!
//! The 40 cases in verif_out.txt use degree=20, order=20 spherical harmonics.
//! 33 cases have perturbOnly=0 (total gravity), 7 have perturbOnly=1
//! (harmonics perturbation only, i.e. total minus point-mass).
#![cfg(feature = "jeod-validation")]

use glam::DVec3;
use jeod_gravity::coefficients;
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

/// Run all 40 gravity test vectors through the Gottlieb algorithm.
///
/// Loads earth_GGM02C coefficients (same as JEOD's grav_geospherical test)
/// and compares acceleration, potential, and gradient against the reference data.
///
/// The JEOD test uses identity planet-fixed rotation, so the test positions
/// are directly in planet-fixed coordinates.
///
/// For perturb_only=false cases: result = nonspherical(n=2..degree) + point_mass
/// For perturb_only=true cases: result = nonspherical(n=2..degree) only
#[test]
fn spherical_harmonics_40_test_vectors() {
    let root = jeod_path();
    assert!(
        root.exists(),
        "JEOD source not found at {}",
        root.display()
    );

    // Load GGM02C (the test was built against GGM02C, not GGM05C)
    let ggm02c_path = root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    assert!(
        ggm02c_path.exists(),
        "GGM02C not found at {}",
        ggm02c_path.display()
    );
    let mut data = coefficients::load_from_jeod_cc(&ggm02c_path);
    // JEOD test overrides tide_free = true (main.cc line 95)
    data.tide_free = true;

    let cases = load_gravity_test_cases(&root);
    assert_eq!(cases.len(), 40);

    // JEOD's own regression tolerances: accel 1e-14, potential 100000, gradient 1e-20.
    // Potential has inherently lower precision in the Gottlieb algorithm.
    let accel_tol = 1e-10;     // m/s^2 per component
    let pot_tol = 100_000.0;   // m^2/s^2 (matches JEOD's own tolerance)
    let grad_tol = 1e-16;      // 1/s^2 per component

    let mut max_accel_err = 0.0_f64;
    let mut max_pot_err = 0.0_f64;
    let mut max_grad_err = 0.0_f64;
    let mut passed = 0;

    for case in &cases {
        let degree = case.degree;
        let order = case.order;
        let grad_active = case.grad_active;
        let perturb_only = case.perturb_only;

        // The Gottlieb algorithm computes harmonics from n=2..degree.
        // The sums are initialized to zero (perturbing-only mode).
        let sh_result = jeod_gravity::compute_nonspherical_gravity(
            &data,
            case.position,
            degree,
            order,
            grad_active,
            if grad_active { degree } else { 0 },
            if grad_active { order } else { 0 },
        );

        // For full gravity (perturb_only=false), add point-mass contribution.
        // JEOD convention: potential is +mu/r (positive specific gravitational
        // potential energy), while our compute_point_mass_gravity returns -mu/r.
        // Also, JEOD's calc_spherical acceleration uses posn (inertial pos relative
        // to planet center), so we use the same.
        let (accel, potential, gradient) = if perturb_only {
            (sh_result.accel, sh_result.potential, sh_result.gradient)
        } else {
            let pm = jeod_gravity::compute_point_mass_gravity(data.mu, case.position);
            let r_mag = case.position.length();
            (
                sh_result.accel + pm.accel,
                sh_result.potential + data.mu / r_mag, // +mu/r (JEOD convention)
                if grad_active {
                    sh_result.gradient + pm.gradient
                } else {
                    sh_result.gradient
                },
            )
        };

        // Check acceleration
        let accel_err = (accel - case.acceleration).length();
        max_accel_err = max_accel_err.max(accel_err);

        // Check potential
        let pot_err = (potential - case.potential).abs();
        max_pot_err = max_pot_err.max(pot_err);

        // Check per-component acceleration
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
            case.case_num, pot_err, pot_tol,
        );

        // Check gradient if active
        if grad_active {
            let expected = case.gradient;
            for i in 0..3 {
                for j in 0..3 {
                    let err = (gradient.col(j)[i] - expected.col(j)[i]).abs();
                    max_grad_err = max_grad_err.max(err);
                    assert!(
                        err < grad_tol,
                        "Case {}: gradient[{}][{}] error {:.6e} > tol {:.6e}",
                        case.case_num, i, j, err, grad_tol,
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

/// Surface gravity sanity check with GGM02C (if available).
#[test]
fn surface_gravity_ggm02c() {
    let root = jeod_path();
    assert!(root.exists());
    let ggm02c_path = root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    if !ggm02c_path.exists() {
        return;
    }
    let data = coefficients::load_from_jeod_cc(&ggm02c_path);

    // Equatorial surface
    let pos_eq = DVec3::new(data.radius, 0.0, 0.0);
    let result_eq = jeod_gravity::compute_nonspherical_gravity(
        &data, pos_eq, data.degree, data.order, false, 0, 0,
    );
    let pm_eq = jeod_gravity::compute_point_mass_gravity(data.mu, pos_eq);
    let g_eq = (result_eq.accel + pm_eq.accel).length();
    assert!(
        (g_eq - 9.78).abs() < 0.1,
        "Equatorial surface gravity: {:.4} m/s^2 (expected ~9.78)",
        g_eq
    );

    // Polar surface
    let r_pol = data.radius * (1.0 - 1.0 / 298.257223563);
    let pos_pol = DVec3::new(0.0, 0.0, r_pol);
    let result_pol = jeod_gravity::compute_nonspherical_gravity(
        &data, pos_pol, data.degree, data.order, false, 0, 0,
    );
    let pm_pol = jeod_gravity::compute_point_mass_gravity(data.mu, pos_pol);
    let g_pol = (result_pol.accel + pm_pol.accel).length();
    assert!(
        (g_pol - 9.83).abs() < 0.1,
        "Polar surface gravity: {:.4} m/s^2 (expected ~9.83)",
        g_pol
    );
}
