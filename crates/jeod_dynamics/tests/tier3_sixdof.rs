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
/// CSV column layout (per frame, repeated for composite_body, core_body, structure):
/// For each axis i in [0,1,2]:
///   position[i], velocity[i], ang_vel_this[i],
///   T_parent_this[i][0], T_parent_this[i][1], T_parent_this[i][2],
///   Q_parent_this.vector[i]
/// Then: Q_parent_this.scalar
///
/// composite_body columns (0-based, after time col 0):
///   Row i=0: cols 1(pos0), 2(vel0), 3(angvel0), 4-6(T[0][0..2]), 7(Q.vec[0])
///   Row i=1: cols 8(pos1), 9(vel1), 10(angvel1), 11-13(T[1][0..2]), 14(Q.vec[1])
///   Row i=2: cols 15(pos2), 16(vel2), 17(angvel2), 18-20(T[2][0..2]), 21(Q.vec[2])
///   Then: col 22 = Q.scalar
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
            continue;
        } // skip header
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

        // Quaternion: JEOD stores [scalar, vec[0], vec[1], vec[2]]
        // CSV has vec[0] at col 7, vec[1] at col 14, vec[2] at col 21, scalar at col 22
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

    let trajectory = load_sixdof_trajectory(&csv_path);
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

    let max_pos_error = report.max_position_error();
    let max_vel_error = report.max_velocity_error();
    let max_quat_error = report.max_quat_angle_error();
    let max_angvel_error = report.max_ang_vel_error();

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
