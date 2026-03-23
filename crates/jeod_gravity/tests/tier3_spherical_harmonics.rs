//! Tier 3: Cross-validate spherical harmonics gravity trajectories against
//! JEOD SIM_dyncomp RUN_3A (4x4) and RUN_3B (8x8).
//!
//! These tests propagate from JEOD's initial conditions using our Gottlieb
//! algorithm and compare the resulting trajectory against JEOD's output.
//!
//! RUN_3A uses: Earth 4x4 geopotential, atmosphere ON (no drag), 28800s, 60s log.
//! RUN_3B uses: Earth 8x8 geopotential, same configuration.
//! Epoch: 2007-11-20 00:00:00 UTC.
//!
//! The gravity field rotates with the Earth, so we must rotate position to
//! planet-fixed coordinates (via GMST) before evaluating harmonics, then
//! rotate the acceleration back to inertial.
//!
//! Requires:
//! - `test_data/dyncomp_run3a_state.csv` and `test_data/dyncomp_run3b_state.csv`
//!   generated via Docker: `docker run --rm -v $(pwd)/test_data:/output jeod-trick`
//! - JEOD_HOME set (for loading GGM02C coefficients)

#![cfg(feature = "jeod-validation")]

use glam::{DMat3, DVec3};
use jeod_dynamics::{rk4_translational_step, TranslationalState};
use jeod_gravity::coefficients;
use jeod_test_data::jeod_path;
use jeod_time::epoch::{J2000_TAI_TJT, SECONDS_PER_DAY};
use std::f64::consts::PI;
use std::path::Path;

/// Earth rotation rate (rad/s) — IAU value.
const OMEGA_EARTH: f64 = 7.292115e-5;

#[derive(Debug)]
struct JeodStateRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
}

fn load_jeod_trajectory(path: &Path) -> Vec<JeodStateRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD trajectory CSV from {}: {e}",
            path.display()
        )
    });

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 17 {
            continue;
        }

        let parse = |s: &str| -> f64 { s.trim().parse().unwrap() };

        records.push(JeodStateRecord {
            time: parse(fields[0]),
            position: DVec3::new(parse(fields[1]), parse(fields[8]), parse(fields[15])),
            velocity: DVec3::new(parse(fields[2]), parse(fields[9]), parse(fields[16])),
        });
    }
    records
}

/// Rotation matrix for angle about z-axis (inertial→planet-fixed).
fn rotation_z(angle: f64) -> DMat3 {
    let (s, c) = angle.sin_cos();
    DMat3::from_cols(
        DVec3::new(c, s, 0.0),
        DVec3::new(-s, c, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    )
}

/// Compute GMST angle in radians at a given simulation time.
///
/// JEOD epoch: 2007-11-20 00:00:00 UTC.
/// TAI-UTC = 32s (overridden in time.py).
/// TAI-UT1 = -32.469s (overridden in time.py).
///
/// We compute UT1 days since J2000.0 and use the IAU GMST formula
/// from jeod_time.
fn gmst_at_sim_time(sim_time_s: f64) -> f64 {
    // Epoch: 2007-11-20 00:00:00 UTC
    // UTC to TAI: TAI = UTC + 32s
    // TAI to UT1: UT1 = TAI + (-32.469)s = UTC - 0.469s
    // MJD of 2007-11-20 00:00:00 UTC = 54424.0
    // TJT = MJD - 40000 = 14424.0
    let epoch_utc_tjt = 14424.0;
    let tai_utc_s = 32.0;
    let tai_ut1_s = -32.469;

    // UT1 at epoch
    let epoch_tai_tjt = epoch_utc_tjt + tai_utc_s / SECONDS_PER_DAY;
    let epoch_ut1_tjt = epoch_tai_tjt + tai_ut1_s / SECONDS_PER_DAY;

    // UT1 at current sim time
    let ut1_tjt = epoch_ut1_tjt + sim_time_s / SECONDS_PER_DAY;
    let ut1_days_since_j2000 = ut1_tjt - J2000_TAI_TJT;

    // GMST formula from jeod_time::conversions
    let dd = ut1_days_since_j2000 - 0.000738762;
    let dd2 = dd * dd;
    let dd3 = dd2 * dd;
    let gmst_frac = 0.7790572733 + 1.002737909350795 * dd + 8.0775e-16 * dd2 - 1.5e-24 * dd3;
    (gmst_frac - gmst_frac.floor()) * 2.0 * PI
}

fn run_sh_trajectory_test(csv_name: &str, degree: usize, order: usize, label: &str) {
    let root = jeod_path();
    assert!(root.exists(), "JEOD source not found");

    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(csv_name);

    assert!(
        csv_path.exists(),
        "JEOD reference trajectory not found at {}. \
         Generate with: docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    // Load GGM02C coefficients (same model as JEOD SIM_dyncomp)
    let ggm02c_path = root.join("models/environment/gravity/data/src/earth_GGM02C.cc");
    let sh_data = coefficients::load_from_jeod_cc(&ggm02c_path);

    let jeod_trajectory = load_jeod_trajectory(&csv_path);
    assert!(
        jeod_trajectory.len() > 100,
        "Expected more than 100 records, got {}",
        jeod_trajectory.len()
    );

    // Gravity acceleration with Earth rotation:
    // 1. Rotate position from inertial to planet-fixed via GMST
    // 2. Evaluate harmonics in planet-fixed coordinates
    // 3. Rotate acceleration back to inertial
    let accel_fn = |s: &TranslationalState, sim_time: f64| -> DVec3 {
        let gmst = gmst_at_sim_time(sim_time);
        let r_inertial_to_pfix = rotation_z(gmst);
        let r_pfix_to_inertial = r_inertial_to_pfix.transpose();

        // Position in planet-fixed
        let pos_pfix = r_inertial_to_pfix * s.position;

        // Point-mass acceleration (frame-independent direction)
        let pm = jeod_gravity::compute_point_mass_gravity(sh_data.mu, s.position);

        // Spherical harmonics perturbation in planet-fixed
        let sh_pfix = jeod_gravity::compute_nonspherical_gravity(
            &sh_data, pos_pfix, degree, order, false, 0, 0,
        );

        // Rotate SH acceleration back to inertial
        let sh_inertial = r_pfix_to_inertial * sh_pfix.accel;

        pm.accel + sh_inertial
    };

    let initial = &jeod_trajectory[0];
    let mut state = TranslationalState {
        position: initial.position,
        velocity: initial.velocity,
    };

    eprintln!("Tier 3: JEOD SIM_dyncomp {} cross-validation", label);
    eprintln!(
        "  Gravity: {}x{} spherical harmonics + point mass (rotating Earth)",
        degree, order
    );
    eprintln!(
        "  JEOD trajectory: {} points over {:.0}s",
        jeod_trajectory.len(),
        jeod_trajectory.last().unwrap().time
    );

    let dt = 10.0;
    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;
    let mut current_time = 0.0_f64;

    for jeod_record in &jeod_trajectory[1..] {
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
                "  t={:6.0}s ({:.1}h): pos_err={:10.1}m  vel_err={:.4}m/s",
                jeod_record.time,
                jeod_record.time / 3600.0,
                pos_error,
                vel_error,
            );
        }
    }

    eprintln!();
    eprintln!("  Max position error: {:.1} m", max_pos_error);
    eprintln!("  Max velocity error: {:.4} m/s", max_vel_error);

    // With spherical harmonics gravity and Earth rotation accounted for,
    // remaining differences come from:
    // - Precession/nutation (JEOD has RNP, we use simple GMST rotation)
    // - Atmosphere model active in JEOD (no drag, but state coupling)
    // - Numerical differences in RK4 implementation
    assert!(
        max_pos_error < 5000.0,
        "{}: Position error {:.1}m exceeds 5 km over 8 hours",
        label,
        max_pos_error,
    );
    assert!(
        max_vel_error < 5.0,
        "{}: Velocity error {:.4}m/s exceeds 5 m/s over 8 hours",
        label,
        max_vel_error,
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
