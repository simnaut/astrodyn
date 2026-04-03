//! Tier 3: SIM_torque_compare_simple — high-resolution gravity torque oracle tests
//!
//! Six runs with progressive complexity, each logging at 1-second resolution over
//! 3 hours (10,800 points). Oracle approach: at each JEOD timestep, take JEOD's
//! logged position and attitude, compute our gravity gradient and torque, compare
//! against JEOD's logged torque.
//!
//! Run configurations (from JEOD input.py files):
//!   01: spherical gravity, gradient OFF           → zero torque (control)
//!   02: spherical gravity, point-mass gradient     → point-mass torque
//!   03: spherical gravity, gradient_degree=4       → identical to 02 (spherical overrides)
//!   04: SH 20×20 gravity, gradient OFF             → zero torque (control)
//!   05: SH 20×20 gravity, point-mass gradient      → point-mass torque (SH trajectory)
//!   06: SH 20×20 gravity, SH 4×4 gradient          → SH gradient torque
//!
//! All runs share: ISS mass (400,000 kg, non-diagonal inertia), epoch Nov 20 2007
//! 00:00 UTC, Earth GGM05C + Sun + Moon (spherical, no gradient), RK4, 10,800 s.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{GravityControl, GravityModel, GravitySource, SimulationTime};

// ── ISS mass properties from JEOD Modified_data/mass/iss.py ──

fn iss_inertia() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    )
}

// ── Epoch constants for Nov 20, 2007 00:00:00 UTC ──
// JEOD overrides: leap_sec_override_val = 32, tai_to_ut1_override_val = -32.469

const EPOCH_UTC_TJT: f64 = 14424.0;
const TAI_UTC_S: f64 = 32.0;
const TAI_TO_UT1_S: f64 = -32.469;

/// Load GGM05C spherical harmonics data from JEOD source.
fn load_ggm05c() -> (GravitySource, f64) {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let ggm05c_path = jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc");
    let sh_data = jeod_sim::coefficients::load_from_jeod_cc(&ggm05c_path).expect("load GGM05C");
    let mu = sh_data.mu;
    let source = GravitySource {
        mu,
        model: GravityModel::SphericalHarmonics(Box::new(sh_data)),
    };
    (source, mu)
}

// ── Zero-torque tests (gradient OFF) ──

fn run_zero_torque_test(csv_name: &str, label: &str) {
    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_torque_simple_csv(&csv_path);
    assert!(!records.is_empty());

    println!(
        "=== Tier 3 Oracle: {label} ({} points) ===",
        records.len()
    );

    let mut non_zero = 0;
    for (i, rec) in records.iter().enumerate() {
        if rec.gravity_torque != DVec3::ZERO {
            non_zero += 1;
            if non_zero <= 3 {
                println!(
                    "  Non-zero torque at t={:.0}s (point {}): [{:.2e}, {:.2e}, {:.2e}]",
                    rec.time, i, rec.gravity_torque.x, rec.gravity_torque.y, rec.gravity_torque.z
                );
            }
        }
    }
    assert_eq!(
        non_zero, 0,
        "{label}: expected all-zero torque (gradient OFF) but found {non_zero} non-zero points"
    );
    println!("  PASS: all {} points have zero torque", records.len());
}

#[test]
fn tier3_torque_simple_run01_zero() {
    run_zero_torque_test(
        "torque_simple_run01_torque_simple.csv",
        "RUN_01 (spherical gravity, gradient OFF)",
    );
}

#[test]
fn tier3_torque_simple_run04_zero() {
    run_zero_torque_test(
        "torque_simple_run04_torque_simple.csv",
        "RUN_04 (SH 20x20 gravity, gradient OFF)",
    );
}

// ── Point-mass gradient tests (runs 02, 03, 05) ──

fn run_point_mass_gradient_test(csv_name: &str, label: &str) {
    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_torque_simple_csv(&csv_path);
    assert!(!records.is_empty());

    let inertia = iss_inertia();
    // Use GGM05C source with spherical=true to match JEOD's configuration:
    // JEOD loads GGM05C but overrides to spherical gravity for runs 02/03.
    // Using the actual SH source with spherical=true ensures the gradient
    // goes through the same code path as JEOD's spherical_harmonics_gravity_controls.
    let (source, _mu) = load_ggm05c();
    let ctrl = GravityControl::<usize>::new_spherical(0, true);

    println!(
        "=== Tier 3 Oracle: {label} ({} points) ===",
        records.len()
    );

    let mut max_err = 0.0_f64;
    let mut max_err_time = 0.0_f64;
    let mut max_comp_err = [0.0_f64; 3];

    for (i, rec) in records.iter().enumerate() {
        // Skip t=0 where torque is zero (no gradient yet in JEOD's first output)
        if rec.gravity_torque == DVec3::ZERO && i == 0 {
            continue;
        }

        let result = ctrl.evaluate(&source, rec.position, None);
        // Use the rotation matrix from CSV directly (avoids quaternion→matrix roundtrip)
        let our_torque = jeod_interactions::compute_gravity_torque(
            &result.grav_grad,
            &rec.t_parent_this,
            &inertia,
        );
        let diff = our_torque - rec.gravity_torque;
        let err = diff.length();

        if err > max_err {
            max_err = err;
            max_err_time = rec.time;
        }
        for c in 0..3 {
            max_comp_err[c] = max_comp_err[c].max(diff[c].abs());
        }

        // Log every 1000s
        if i > 0 && (rec.time % 1000.0).abs() < 0.5 {
            println!(
                "  t={:6.0}s: torque err = {:.2e} N·m  [{:.2e}, {:.2e}, {:.2e}]",
                rec.time, err, diff.x, diff.y, diff.z
            );
        }
    }

    println!(
        "  Max torque error: {:.2e} N·m at t={:.0}s",
        max_err, max_err_time
    );
    println!(
        "  Max component errors: [{:.2e}, {:.2e}, {:.2e}] N·m",
        max_comp_err[0], max_comp_err[1], max_comp_err[2]
    );

    // Tolerance: JEOD logs the torque from the last RK4 sub-step (dt=0.03125s
    // before the logged state). The gradient changes as the vehicle moves, so
    // the torque at the logged position differs from the logged torque by
    // approximately dG/dt × dt × I ≈ (v/r × G) × 0.03125 × I ≈ 5e-3 N·m.
    // A 1e-2 N·m threshold provides 2× margin above this timing offset.
    let tolerance = 1e-2;
    assert!(
        max_err < tolerance,
        "{label}: max torque error {:.2e} N·m exceeds {:.0e} N·m threshold (at t={:.0}s)",
        max_err,
        tolerance,
        max_err_time
    );
    println!(
        "  PASS: max error {:.2e} N·m < {:.0e} threshold",
        max_err, tolerance
    );
}

#[test]
fn tier3_torque_simple_run02_point_mass_gradient() {
    run_point_mass_gradient_test(
        "torque_simple_run02_torque_simple.csv",
        "RUN_02 (spherical gravity, point-mass gradient)",
    );
}

#[test]
fn tier3_torque_simple_run03_spherical_gradient_degree4() {
    // Run 03 has gradient_degree=4 but spherical=true, so JEOD computes
    // point-mass gradient only. Run 03 produces identical torques to Run 02.
    run_point_mass_gradient_test(
        "torque_simple_run03_torque_simple.csv",
        "RUN_03 (spherical gravity, gradient_degree=4 — same as point-mass)",
    );
}

#[test]
fn tier3_torque_simple_run05_sh_gravity_point_mass_gradient() {
    run_point_mass_gradient_test(
        "torque_simple_run05_torque_simple.csv",
        "RUN_05 (SH 20x20 gravity, point-mass gradient)",
    );
}

// ── SH 4×4 gradient test (run 06) ──

#[test]
fn tier3_torque_simple_run06_sh_gradient_4x4() {
    let csv_path = test_data_path("torque_simple_run06_torque_simple.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_torque_simple_csv(&csv_path);
    assert!(!records.is_empty());

    let (source, _mu) = load_ggm05c();
    let inertia = iss_inertia();

    // Run 06: SH 20x20 for acceleration, SH 4x4 for gradient
    let mut ctrl = GravityControl::<usize>::new_nonspherical(0, 20, 20, true);
    ctrl.gradient_degree = 4;
    ctrl.gradient_order = 4;

    // Epoch: Nov 20, 2007 00:00:00 UTC → TAI TJT
    let epoch_tai_tjt = EPOCH_UTC_TJT + TAI_UTC_S / 86400.0;
    let mut time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    time.set_ut1_tai_offset(TAI_TO_UT1_S);

    println!(
        "=== Tier 3 Oracle: RUN_06 (SH 20x20 gravity, SH 4x4 gradient) ({} points) ===",
        records.len()
    );

    let mut max_err = 0.0_f64;
    let mut max_err_time = 0.0_f64;
    let mut max_comp_err = [0.0_f64; 3];

    for (i, rec) in records.iter().enumerate() {
        if rec.gravity_torque == DVec3::ZERO && i == 0 {
            continue;
        }

        // Advance time to match this record (records at 1-second intervals)
        if i > 0 {
            time.advance(1.0);
        }

        let t_inertial_pfix =
            jeod_sim::compute_t_parent_this_from_tjt(time.gmst_seconds, time.tt_tjt());

        let result = ctrl.evaluate(&source, rec.position, Some(&t_inertial_pfix));
        let our_torque = jeod_interactions::compute_gravity_torque(
            &result.grav_grad,
            &rec.t_parent_this,
            &inertia,
        );
        let diff = our_torque - rec.gravity_torque;
        let err = diff.length();

        if err > max_err {
            max_err = err;
            max_err_time = rec.time;
        }
        for c in 0..3 {
            max_comp_err[c] = max_comp_err[c].max(diff[c].abs());
        }

        if i > 0 && (rec.time % 1000.0).abs() < 0.5 {
            println!(
                "  t={:6.0}s: torque err = {:.2e} N·m  [{:.2e}, {:.2e}, {:.2e}]",
                rec.time, err, diff.x, diff.y, diff.z
            );
        }
    }

    println!(
        "  Max torque error: {:.2e} N·m at t={:.0}s",
        max_err, max_err_time
    );
    println!(
        "  Max component errors: [{:.2e}, {:.2e}, {:.2e}] N·m",
        max_comp_err[0], max_comp_err[1], max_comp_err[2]
    );

    // Same timing-offset tolerance as point-mass tests (see comment above).
    let tolerance = 1e-2;
    assert!(
        max_err < tolerance,
        "RUN_06: max torque error {:.2e} N·m exceeds {:.0e} N·m threshold (at t={:.0}s)",
        max_err,
        tolerance,
        max_err_time
    );
    println!(
        "  PASS: max error {:.2e} N·m < {:.0e} threshold",
        max_err, tolerance
    );
}
