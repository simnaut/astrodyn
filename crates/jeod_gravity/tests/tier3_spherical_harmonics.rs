//! Tier 3: Cross-validate spherical harmonics gravity trajectories against
//! JEOD SIM_dyncomp RUN_3A (4x4) and RUN_3B (8x8).
//!
//! Propagates from JEOD's initial conditions using our Gottlieb algorithm
//! with our own RNP (precession, nutation, GAST rotation) computation.
//! No JEOD data is used in the computation — only for comparison.
//!
//! RUN_3A: Earth 4x4 geopotential, atmosphere ON (no drag), 28800s, 60s log.
//! RUN_3B: Earth 8x8 geopotential, same configuration.
//! Epoch: 2007-11-20 00:00:00 UTC.
//!
//! Requires (generated via Docker):
//! - `test_data/dyncomp_run3a_state.csv` and `test_data/dyncomp_run3b_state.csv`
//! - JEOD_HOME set (for GGM02C coefficients)

#![cfg(feature = "jeod-validation")]

use glam::{DMat3, DVec3};
use jeod_dynamics::{rk4_translational_step, TranslationalState};
use jeod_frames::rotation_j2000;
use jeod_gravity::coefficients;
use jeod_test_data::jeod_path;
use jeod_time::time_converter_ut1_gmst::ut1_to_gmst_days;
use jeod_time::epoch::{J2000_TAI_TJT, SECONDS_PER_DAY};
use std::path::Path;

// JEOD SIM_dyncomp epoch: 2007-11-20 00:00:00 UTC
// TAI-UTC = 32s (overridden in time.py)
// tai_to_ut1 = -32.469s (JEOD: time.tai_ut1.tai_to_ut1_override_val)
// UTC MJD = 54424.0, TJT = 14424.0
const EPOCH_UTC_TJT: f64 = 14424.0;
const TAI_UTC_S: f64 = 32.0;
const TAI_TO_UT1_S: f64 = -32.469;

/// Compute the inertial-to-planet-fixed rotation matrix at a given sim time.
///
/// Uses our own RNP implementation — no JEOD data.
fn rotation_at_sim_time(sim_time_s: f64) -> DMat3 {
    // TAI TJT at epoch
    let epoch_tai_tjt = EPOCH_UTC_TJT + TAI_UTC_S / SECONDS_PER_DAY;

    // Current TAI TJT
    let tai_tjt = epoch_tai_tjt + sim_time_s / SECONDS_PER_DAY;

    // TT TJT = TAI TJT + 32.184/86400
    let tt_tjt = tai_tjt + 32.184 / SECONDS_PER_DAY;

    // TT centuries since J2000: (tt_tjt - 11544.5) / 36525.0
    let tt_centuries = (tt_tjt - 11544.5) / 36525.0;

    // UT1 TJT = TAI TJT + ut1_tai_offset / 86400
    let ut1_tjt = tai_tjt + TAI_TO_UT1_S / SECONDS_PER_DAY;

    // UT1 days since J2000 (for GMST formula)
    let ut1_days = ut1_tjt - J2000_TAI_TJT;

    // GMST in accumulated sidereal days → seconds (matches JEOD TimeGMST::seconds)
    let gmst_days = ut1_to_gmst_days(ut1_days);
    let gmst_seconds = gmst_days * SECONDS_PER_DAY;

    rotation_j2000::compute_t_parent_this(gmst_seconds, tt_centuries)
}

#[derive(Debug)]
struct JeodStateRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
}

fn load_jeod_trajectory(path: &Path) -> Vec<JeodStateRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("Failed to read {}: {e}", path.display())
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 { continue; }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 17 { continue; }
        let p = |s: &str| -> f64 { s.trim().parse().unwrap() };
        records.push(JeodStateRecord {
            time: p(f[0]),
            position: DVec3::new(p(f[1]), p(f[8]), p(f[15])),
            velocity: DVec3::new(p(f[2]), p(f[9]), p(f[16])),
        });
    }
    records
}

fn run_sh_trajectory_test(csv_name: &str, degree: usize, order: usize, label: &str) {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found");

    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(csv_name);
    assert!(
        csv_path.exists(),
        "JEOD trajectory not found at {}. \
         Generate with: docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let ggm02c_path = root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    let sh_data = coefficients::load_from_jeod_cc(&ggm02c_path);

    let trajectory = load_jeod_trajectory(&csv_path);
    assert!(trajectory.len() > 100);

    // Gravity acceleration using our own RNP with truncated degree/order.
    // Uses calc_nonspherical directly (not gravitation) because
    // the test exercises specific degree/order truncation. gravitation
    // handles full inertial→planet-fixed→inertial transforms per JEOD's
    // calc_nonspherical when using the full model.
    let accel_fn = |s: &TranslationalState, sim_time: f64| -> DVec3 {
        let t_i2pf = rotation_at_sim_time(sim_time);
        let t_pf2i = t_i2pf.transpose();

        let pos_pfix = t_i2pf * s.position;
        let pm = jeod_gravity::calc_spherical(sh_data.mu, s.position);
        let sh_pfix = jeod_gravity::calc_nonspherical(
            &sh_data, pos_pfix, degree, order, false, 0, 0,
        );
        pm.grav_accel + t_pf2i * sh_pfix.grav_accel
    };

    let initial = &trajectory[0];
    let mut state = TranslationalState {
        position: initial.position,
        velocity: initial.velocity,
    };

    eprintln!("Tier 3: JEOD SIM_dyncomp {} cross-validation", label);
    eprintln!("  Gravity: {}x{} + our RNP (precession + nutation + GAST)", degree, order);
    eprintln!("  Trajectory: {} points over {:.0}s", trajectory.len(), trajectory.last().unwrap().time);

    let dt = 10.0;
    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;
    let mut current_time = 0.0_f64;

    for jeod_record in &trajectory[1..] {
        while current_time + dt <= jeod_record.time + 0.001 {
            let t = current_time;
            state = rk4_translational_step(&state, |s| accel_fn(s, t), dt);
            current_time += dt;
        }
        let remainder = jeod_record.time - current_time;
        if remainder > 0.01 {
            let t = current_time;
            state = rk4_translational_step(&state, |s| accel_fn(s, t), remainder);
            current_time += remainder;
        }

        let pos_error = (state.position - jeod_record.position).length();
        let vel_error = (state.velocity - jeod_record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if (jeod_record.time % 3600.0).abs() < 30.1 {
            eprintln!(
                "  t={:6.0}s ({:.1}h): pos_err={:10.2}m  vel_err={:.6}m/s",
                jeod_record.time, jeod_record.time / 3600.0, pos_error, vel_error,
            );
        }
    }

    eprintln!();
    eprintln!("  Max position error: {:.2} m", max_pos_error);
    eprintln!("  Max velocity error: {:.6} m/s", max_vel_error);

    // With our own RNP computation, differences come from:
    // - Floating-point differences in RNP between our code and JEOD
    // - RK4 integration numerical differences
    // - JEOD atmosphere model coupling (active but no drag force)
    assert!(
        max_pos_error < 50.0,
        "{}: Position error {:.2}m exceeds 50m over 8 hours",
        label, max_pos_error,
    );
    assert!(
        max_vel_error < 0.05,
        "{}: Velocity error {:.6}m/s exceeds 0.05 m/s over 8 hours",
        label, max_vel_error,
    );
}

#[test]
fn tier3_dyncomp_run3a_4x4_gravity() {
    run_sh_trajectory_test("dyncomp_run3a_state.csv", 4, 4, "RUN_3A (4x4)");
}

#[test]
fn tier3_dyncomp_run3b_8x8_gravity() {
    run_sh_trajectory_test("dyncomp_run3b_state.csv", 8, 8, "RUN_3B (8x8)");
}
