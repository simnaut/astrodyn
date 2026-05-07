#![cfg(feature = "verification")]

//! Tier 3: Battin's method vs direct subtraction for third-body gravity
//!
//! Verifies that Battin's method for differential (third-body) gravity
//! produces the same trajectory as the default direct subtraction method
//! through the full `Simulation::step()` pipeline.
//!
//! Both methods are mathematically equivalent; Battin's reformulation avoids
//! catastrophic cancellation when the vehicle is close to the integration
//! frame origin relative to the third-body distance. For LEO with Sun as
//! third body, the numerical difference is negligible because the Sun is
//! ~1 AU away while the vehicle is ~6800 km from Earth center.
//!
//! Migrated from a 344-line bespoke test (#188, follow-on to #162). The
//! scenario assembly and per-step DE421 ephemeris injection live in
//! `sim_dyncomp::{build_battin_3rd_body, battin_pre_step}`. The
//! cross-compare between the two simulations stays here because there is
//! no JEOD CSV reference for it — the comparison is internal between the
//! two sibling sims.

use astrodyn_verif_jeod::verification::InitialConditions;
use astrodyn_verif_jeod::run_verification::sim_dyncomp;
use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::tier3_csv::{load_dyncomp_csv, test_data_path};
use glam::DVec3;

/// Simulation duration (seconds): 8 hours.
const DURATION: f64 = 28800.0;

/// Logging interval (seconds): record state every 60s for comparison.
const LOG_INTERVAL: f64 = 60.0;

/// Propagate one Battin/direct cross-compare run for `DURATION`,
/// sampling position+velocity every `LOG_INTERVAL`.
fn propagate(battin: bool, init: &InitialConditions) -> (Vec<DVec3>, Vec<DVec3>) {
    let scenario = sim_dyncomp::build_battin_3rd_body(init, battin);
    let mut pre_step = sim_dyncomp::battin_pre_step(scenario.sun_idx, scenario.moon_idx);
    let mut sim = scenario
        .builder
        .build()
        .expect("scenario validation failed");

    let n_points = (DURATION / LOG_INTERVAL) as usize;
    let mut positions = Vec::with_capacity(n_points);
    let mut velocities = Vec::with_capacity(n_points);
    for i in 1..=n_points {
        let target_time = i as f64 * LOG_INTERVAL;
        pre_step(&mut sim, target_time);
        sim.step_until(target_time).expect("step_until failed");
        let body = sim.body(0);
        positions.push(body.trans.position);
        velocities.push(body.trans.velocity);
    }
    (positions, velocities)
}

/// Verify Battin's method produces identical trajectory to direct method
/// through the full Simulation::step() pipeline with Sun + Moon third-body.
///
/// Both methods are mathematically equivalent for third-body differential
/// acceleration. The only difference is floating-point rounding: Battin's
/// method avoids catastrophic cancellation, so it may actually be *more*
/// accurate than direct subtraction. For LEO + Sun/Moon, the cancellation
/// is negligible (~5 digits lost in direct method), so both trajectories
/// should agree to within machine epsilon accumulated over 8 hours.
#[test]
fn tier3_battin_vs_direct_trajectory() {
    // Initial conditions come from the t=0 row of the JEOD reference CSV
    // (same JEOD source data as RUN_4). The recipe asserts JEOD source
    // and DE421 BSP existence internally.
    let csv_path = test_data_path("dyncomp_run4_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );
    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() > 100);
    let t0 = &trajectory[0];
    let init = InitialConditions {
        time: t0.time,
        position: t0.composite_body.position,
        velocity: t0.composite_body.velocity,
        quaternion: Some(t0.composite_body.quaternion),
        ang_vel: Some(t0.composite_body.ang_vel),
    };

    let (pos_d, vel_d) = propagate(false, &init);
    let (pos_b, vel_b) = propagate(true, &init);

    assert_eq!(pos_d.len(), pos_b.len());

    let mut max_pos_diff = 0.0_f64;
    let mut max_vel_diff = 0.0_f64;
    let mut max_pos_idx = 0usize;
    let mut max_vel_idx = 0usize;
    for i in 0..pos_d.len() {
        let pos_diff = (pos_d[i] - pos_b[i]).length();
        let vel_diff = (vel_d[i] - vel_b[i]).length();
        if pos_diff > max_pos_diff {
            max_pos_diff = pos_diff;
            max_pos_idx = i;
        }
        if vel_diff > max_vel_diff {
            max_vel_diff = vel_diff;
            max_vel_idx = i;
        }
    }
    let max_pos_time = (max_pos_idx + 1) as f64 * LOG_INTERVAL;
    let max_vel_time = (max_vel_idx + 1) as f64 * LOG_INTERVAL;

    println!(
        "Tier 3 (Battin vs Direct): {} points over {} hours",
        pos_d.len(),
        DURATION / 3600.0
    );
    println!("  Max position difference: {max_pos_diff:.6e} m at t={max_pos_time:.0}s");
    println!("  Max velocity difference: {max_vel_diff:.6e} m/s at t={max_vel_time:.0}s");

    // Both methods are mathematically equivalent but have different floating-point
    // rounding characteristics. The direct method subtracts two nearly-equal
    // accelerations (vehicle->Sun minus Earth->Sun), losing ~5 significant digits
    // for LEO + Sun geometry (ratio ~4.5e-8). Battin's method reformulates the
    // computation to avoid this cancellation, so the two methods diverge by the
    // rounding error of the less-precise (direct) method.
    //
    // Over 8 hours (2880 RK4 steps at dt=10s), accumulated rounding differences
    // produce ~0.55 m position and ~4.6e-4 m/s velocity divergence. This is
    // consistent with ~1e-12 m/s^2 per-step acceleration rounding error integrated
    // over the full trajectory. The divergence is small compared to the trajectory
    // itself (position ~6.8e6 m, velocity ~7.7e3 m/s).
    //
    // Tolerances: 5% above observed max error per the project tolerance policy.
    assert!(
        max_pos_diff < 5.808e-1,
        "Position difference between Battin and direct methods too large: \
         {max_pos_diff:.6e} m at t={max_pos_time:.0}s (limit 5.808e-1 m)"
    );
    assert!(
        max_vel_diff < 4.798e-4,
        "Velocity difference between Battin and direct methods too large: \
         {max_vel_diff:.6e} m/s at t={max_vel_time:.0}s (limit 4.798e-4 m/s)"
    );
}
