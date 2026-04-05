//! Tier 3: Cross-validation of external torque + 6-DOF propagation against
//! JEOD SIM_dyncomp RUN_9A reference data.
//!
//! RUN_9A configuration (from SET_test/RUN_9A/input.py, which defers to RUN_9B
//! then calls set_rot_rate_inrtl()):
//! - Spherical gravity (point-mass GM/r^2, Earth only)
//! - Gravity gradient torque: OFF (grav_torque.active = False in common_input.py)
//! - External torque: [10, 0, 0] N*m applied in structural frame from t=1000s
//!   to t=2000s (via trick.add_read in RUN_9B/input.py)
//! - ISS mass configuration (non-spherical inertia, 400000 kg)
//! - Initial attitude: LVLH-based, Yaw-Pitch-Roll = [0, -11.6 deg, 0]
//! - Initial angular rate: zero in inertial frame (set_rot_rate_inrtl)
//! - Structural-to-body transform: identity (eigen_angle = 0.0 in mass.py)
//! - 28800s (8 hours), logged every 60s, 481 data points
//!
//! This validates:
//! 1. Euler's equation integration (I*alpha = tau - omega x (I*omega))
//! 2. Quaternion propagation under external torque
//! 3. Correct torque magnitude and direction through non-diagonal inertia
//!
//! The structural and composite_body frames are aligned (identity rotation
//! in mass.py), so the [10, 0, 0] N*m structural-frame torque maps directly
//! to body-frame torque without transformation.

use glam::{DMat3, DVec3};
use jeod_dynamics::{
    rk4_sixdof_step, MassProperties, RotationalState, SixDofState, TranslationalState,
};
use jeod_math::JeodQuat;
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

const MU_EARTH: f64 = 3.986_004_415e14;

/// Parsed 6-DOF state record from JEOD CSV.
#[derive(Debug)]
struct JeodSixDofRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    quaternion: JeodQuat,
    ang_vel: DVec3,
    trans_accel: Option<DVec3>,
    rot_accel: Option<DVec3>,
}

/// Parse the JEOD log_state_ASCII CSV for composite_body 6-DOF state.
///
/// CSV column layout (0-indexed, after time at col 0):
/// For each axis i in [0,1,2], stride of 7:
///   position[i], velocity[i], ang_vel_this[i],
///   T_parent_this[i][0..2], Q_parent_this.vector[i]
/// Then: Q_parent_this.scalar
///
/// composite_body columns:
///   i=0: cols 1(pos0), 2(vel0), 3(angvel0), 4-6(T[0][0..2]), 7(Q.vec[0])
///   i=1: cols 8(pos1), 9(vel1), 10(angvel1), 11-13(T[1][0..2]), 14(Q.vec[1])
///   i=2: cols 15(pos2), 16(vel2), 17(angvel2), 18-20(T[2][0..2]), 21(Q.vec[2])
///   col 22: Q.scalar
fn load_sixdof_trajectory(path: &Path) -> Vec<JeodSixDofRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read JEOD trajectory CSV from {}: {e}\n\
             Generate with: docker build -f trick/Dockerfile -t jeod-trick .. && \
             docker run --rm -v $(pwd)/test_data:/output jeod-trick",
            path.display()
        )
    });

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 23,
            "Malformed JEOD CSV at line {}: expected at least 23 fields, found {}",
            i + 1,
            fields.len(),
        );

        let parse = |s: &str, col: usize| -> f64 {
            let line_no = i + 1;
            s.trim().parse::<f64>().unwrap_or_else(|e| {
                panic!("Failed to parse JEOD CSV at line {line_no}, col {col}: {s:?} ({e})")
            })
        };

        // Composite body state columns
        let position = DVec3::new(
            parse(fields[1], 1),
            parse(fields[8], 8),
            parse(fields[15], 15),
        );
        let velocity = DVec3::new(
            parse(fields[2], 2),
            parse(fields[9], 9),
            parse(fields[16], 16),
        );
        let ang_vel = DVec3::new(
            parse(fields[3], 3),
            parse(fields[10], 10),
            parse(fields[17], 17),
        );

        // Quaternion: CSV has vec[0] at col 7, vec[1] at col 14, vec[2] at col 21, scalar at col 22
        let q_scalar = parse(fields[22], 22);
        let q_vec = DVec3::new(
            parse(fields[7], 7),
            parse(fields[14], 14),
            parse(fields[21], 21),
        );
        let quaternion = JeodQuat::new(q_scalar, q_vec.x, q_vec.y, q_vec.z);

        // Parse optional acceleration columns (same layout as sim_test_helpers)
        let (trans_accel, rot_accel) = if fields.len() >= 79 {
            let ta = DVec3::new(
                parse(fields[68], 68),
                parse(fields[72], 72),
                parse(fields[76], 76),
            );
            let ra = DVec3::new(
                parse(fields[69], 69),
                parse(fields[73], 73),
                parse(fields[77], 77),
            );
            (Some(ta), Some(ra))
        } else {
            (None, None)
        };

        records.push(JeodSixDofRecord {
            time: parse(fields[0], 0),
            position,
            velocity,
            quaternion,
            ang_vel,
            trans_accel,
            rot_accel,
        });
    }
    records
}

/// Compute angular error between two quaternions in radians.
fn quaternion_angle_error(q1: &JeodQuat, q2: &JeodQuat) -> f64 {
    let dot = (q1.scalar() * q2.scalar()
        + q1.vector().x * q2.vector().x
        + q1.vector().y * q2.vector().y
        + q1.vector().z * q2.vector().z)
        .abs();
    // Clamp to avoid NaN from numerical noise
    2.0 * dot.min(1.0).acos()
}

#[test]
fn tier3_external_torque_sixdof_run9a() {
    let csv_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/dyncomp_run9a_state.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_sixdof_trajectory(&csv_path);
    assert!(
        trajectory.len() >= 100,
        "Expected at least 100 data points, got {}",
        trajectory.len()
    );

    // ISS mass properties from SIM_dyncomp Modified_data/mass.py (set_mass_iss).
    // mass = 400000.0 kg
    // position (CoM offset from structural origin) = [-3.0, -1.5, 4.0] m
    // inertia (kg*m^2, about body frame axes through CoM):
    //   [  1.02e8, -6.96e6, -5.48e6 ]
    //   [ -6.96e6,  0.91e8,  5.90e5 ]
    //   [ -5.48e6,  5.90e5,  1.64e8 ]
    //
    // Structural-to-body transform is identity (eigen_angle = 0.0 in mass.py),
    // so structural-frame torque equals body-frame torque.
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
            position: init.position,
            velocity: init.velocity,
        },
        rot: RotationalState {
            quaternion: init.quaternion,
            ang_vel_body: init.ang_vel,
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
            position: Some(r.position),
            velocity: Some(r.velocity),
            acceleration: r.trans_accel,
            quaternion: Some(r.quaternion.to_glam()),
            ang_vel: Some(r.ang_vel),
            ang_accel: r.rot_accel,
        })
        .collect();

    // RUN_9A external torque schedule (from RUN_9B/input.py):
    // trick.add_read(1000.0, "vehicle.torque_extern.torque = [10.0, 0.0, 0.0]")
    // trick.add_read(2000.0, "vehicle.torque_extern.torque = [ 0.0, 0.0, 0.0]")
    //
    // The torque is specified in the structural frame. Since the structural-to-body
    // transform is identity for this sim, body-frame torque = structural-frame torque.
    let external_torque = DVec3::new(10.0, 0.0, 0.0);

    // Gravity accel closure (point-mass)
    let gravity_accel = |s: &SixDofState| -> DVec3 {
        let r_sq = s.trans.position.length_squared();
        let r_mag = r_sq.sqrt();
        s.trans.position * (-MU_EARTH / (r_sq * r_mag))
    };

    for record in trajectory.iter().skip(1) {
        // Integrate forward to this record's time
        while current_time + dt <= record.time + 0.001 {
            // Determine torque at current_time:
            // Active from t=1000 to t=2000 (trick.add_read is evaluated at the
            // *start* of the given timestep, so torque is first applied at
            // t=1000 and first removed at t=2000).
            let torque_active = (999.999..1999.999).contains(&current_time);
            let torque = if torque_active {
                external_torque
            } else {
                DVec3::ZERO
            };

            state = rk4_sixdof_step(&state, gravity_accel, |_s| torque, &mass_props, dt);
            current_time += dt;
        }
        let remainder = record.time - current_time;
        if remainder > 0.001 {
            let torque_active = (999.999..1999.999).contains(&current_time);
            let torque = if torque_active {
                external_torque
            } else {
                DVec3::ZERO
            };

            state = rk4_sixdof_step(&state, gravity_accel, |_s| torque, &mass_props, remainder);
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

        // Compare for logging
        let pos_error = (state.trans.position - record.position).length();
        let vel_error = (state.trans.velocity - record.velocity).length();
        let quat_error = quaternion_angle_error(&state.rot.quaternion, &record.quaternion);
        let angvel_error = (state.rot.ang_vel_body - record.ang_vel).length();

        // Log progress at key points: every hour and at torque boundaries
        let log_hourly = (record.time % 3600.0).abs() < 30.1;
        let log_torque = (record.time - 1020.0).abs() < 0.1 || (record.time - 2040.0).abs() < 0.1;
        if log_hourly || log_torque {
            println!(
                "  t={:6.0}s ({:.1}h): pos_err={:10.2}m  vel_err={:.6}m/s  \
                 quat_err={:.6e}rad  angvel_err={:.6e}rad/s",
                record.time,
                record.time / 3600.0,
                pos_error,
                vel_error,
                quat_error,
                angvel_error,
            );
        }
    }

    let mut report = CrossvalReport::compute(
        "tier3_external_torque_sixdof_run9a",
        &our_states,
        &ref_states,
    );
    report.position_tol = Some([0.5; 3]);
    report.velocity_tol = Some([0.001; 3]);
    report.quat_angle_tol = Some(0.01);
    report.ang_vel_tol = Some([1e-5; 3]);
    report.write();

    let max_pos_error = report.max_position_error();
    let max_vel_error = report.max_velocity_error();
    let max_quat_error = report.max_quat_angle_error();
    let max_angvel_error = report.max_ang_vel_error();

    println!();
    println!("=== Tier 3 External Torque 6-DOF Cross-Validation (RUN_9A) ===");
    println!(
        "Duration: {} s ({} data points)",
        trajectory.last().unwrap().time,
        trajectory.len()
    );
    println!("Torque: [10, 0, 0] N*m in structural frame, t=1000-2000s");
    println!("Max position error:   {:.6e} m", max_pos_error);
    println!("Max velocity error:   {:.6e} m/s", max_vel_error);
    println!(
        "Max quaternion error: {:.6e} rad ({:.4} deg)",
        max_quat_error,
        max_quat_error.to_degrees()
    );
    println!("Max ang_vel error:    {:.6e} rad/s", max_angvel_error);

    // Translational thresholds match the existing RUN_2 6-DOF test.
    // dt=0.03125s matches JEOD's SIM_dyncomp integration rate (32 Hz).
    assert!(
        max_pos_error < 0.5,
        "Position error {:.2} m exceeds 0.5 m threshold",
        max_pos_error
    );
    assert!(
        max_vel_error < 0.001,
        "Velocity error {:.6} m/s exceeds 0.001 m/s threshold",
        max_vel_error
    );

    // Rotational thresholds: the external torque creates non-trivial angular
    // motion over 8 hours. Quaternion error < 0.01 rad is the Phase 4 exit
    // criterion. Angular velocity error should track closely given matching
    // integration rates.
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
