//! Tier 3: 6-DOF cross-validation of rotational dynamics against JEOD SIM_dyncomp.
//!
//! Validates attitude propagation by comparing our quaternion evolution against
//! JEOD's logged rotational state in the existing CSV files.
//!
//! RUN_2 configuration:
//! - Spherical gravity (point-mass GM/r²)
//! - rotational_dynamics = True (set in Modified_data/integration.py)
//! - ISS mass configuration (non-spherical inertia)
//! - Initial attitude: LVLH-based, Yaw-Pitch-Roll = [0, -11.6 deg, 0]
//! - Initial angular rate: zero in LVLH (body rotates at orbital rate in inertial)
//! - 28800s (8 hours), logged every 60s, 481 data points
//! - No external torques (gravity torque OFF, aero drag OFF)
//!
//! The CSV contains the full composite_body frame state including quaternion
//! Q_parent_this and angular velocity ang_vel_this.

use glam::{DMat3, DVec3};
use jeod_dynamics::{
    rk4_sixdof_step, MassProperties, RotationalState, SixDofState, TranslationalState,
};
use jeod_math::JeodQuat;
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use jeod_test_data::dyncomp_csv::load_dyncomp_csv;
use std::path::Path;

const MU_EARTH: f64 = 3.986_004_415e14;

#[test]
fn tier3_sixdof_attitude_from_run2() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dyncomp_run2_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(
        trajectory.len() >= 100,
        "Expected at least 100 data points, got {}",
        trajectory.len()
    );

    // ISS mass properties from SIM_dyncomp Modified_data/mass.py (set_mass_iss).
    // mass = 400000.0 kg
    // inertia (kg*m^2, with off-diagonal products):
    //   [  1.02e8, -6.96e6, -5.48e6 ]
    //   [ -6.96e6,  0.91e8,  5.90e5 ]
    //   [ -5.48e6,  5.90e5,  1.64e8 ]
    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

    // Initialize from first JEOD record
    let init = &trajectory[0];
    let mut state = SixDofState {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
        },
    };

    let dt = 0.03125; // match JEOD's SIM_dyncomp integration rate (32 Hz)
    let mut current_time = init.time;

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let ref_states: Vec<StateLog> = trajectory
        .iter()
        .skip(1)
        .map(|r| StateLog {
            time: r.time,
            position: Some(r.composite_body.position),
            velocity: Some(r.composite_body.velocity),
            acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
            quaternion: Some(r.composite_body.quaternion),
            ang_vel: Some(r.composite_body.ang_vel),
            ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
        })
        .collect();

    for record in trajectory.iter().skip(1) {
        // Integrate forward to this record's time
        while current_time + dt <= record.time + 0.001 {
            state = rk4_sixdof_step(
                &state,
                |s| {
                    let r_sq = s.trans.position.length_squared();
                    let r_mag = r_sq.sqrt();
                    s.trans.position * (-MU_EARTH / (r_sq * r_mag))
                },
                |_s| DVec3::ZERO, // No external torques
                &mass_props,
                dt,
            );
            current_time += dt;
        }
        let remainder = record.time - current_time;
        if remainder > 0.001 {
            state = rk4_sixdof_step(
                &state,
                |s| {
                    let r_sq = s.trans.position.length_squared();
                    let r_mag = r_sq.sqrt();
                    s.trans.position * (-MU_EARTH / (r_sq * r_mag))
                },
                |_s| DVec3::ZERO,
                &mass_props,
                remainder,
            );
            current_time += remainder;
        }

        our_states.push(StateLog {
            time: record.time,
            position: Some(state.trans.position),
            velocity: Some(state.trans.velocity),
            quaternion: Some(state.rot.quaternion.to_glam()),
            ang_vel: Some(state.rot.ang_vel_body),
            ..Default::default()
        });
    }

    let mut report =
        CrossvalReport::compute("tier3_sixdof_attitude_from_run2", &our_states, &ref_states);
    report.position_tol = Some([0.5; 3]);
    report.velocity_tol = Some([0.001; 3]);
    report.quat_angle_tol = Some(0.01);
    report.ang_vel_tol = Some([1e-5; 3]);
    report.write();

    let max_pos_error = report.max_position_component();
    let max_vel_error = report.max_velocity_component();
    let max_quat_error = report.max_quat_angle();
    let max_angvel_error = report.max_ang_vel_component();

    println!("=== Tier 3 6-DOF Cross-Validation (RUN_2) ===");
    println!(
        "Duration: {} s ({} data points)",
        trajectory.last().unwrap().time,
        trajectory.len()
    );
    println!("Max position error:   {:.6e} m", max_pos_error);
    println!("Max velocity error:   {:.6e} m/s", max_vel_error);
    println!(
        "Max quaternion error: {:.6e} rad ({:.4} deg)",
        max_quat_error,
        max_quat_error.to_degrees()
    );
    println!("Max ang_vel error:    {:.6e} rad/s", max_angvel_error);

    // dt=0.03125s matches JEOD's SIM_dyncomp integration rate (32 Hz).
    // Residual comes from FP differences between our Rust/LLVM implementation
    // and JEOD's C++/GCC implementation accumulated over ~921,600 RK4 steps.
    assert!(
        max_pos_error < 0.5,
        "Position error {:.2} m exceeds 0.5 m threshold",
        max_pos_error
    );
    assert!(
        max_vel_error < 0.001,
        "Velocity error {:.4} m/s exceeds 0.001 m/s threshold",
        max_vel_error
    );
    assert!(
        max_quat_error < 0.01,
        "Quaternion angular error {:.6e} rad exceeds 0.01 rad threshold",
        max_quat_error
    );
    assert!(
        max_angvel_error < 1e-5,
        "Angular velocity error {:.6e} rad/s exceeds 1e-5 rad/s threshold",
        max_angvel_error
    );
}
