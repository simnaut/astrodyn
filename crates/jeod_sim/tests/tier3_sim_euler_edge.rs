//! Tier 3: SIM_Euler edge-case cross-validation
//!
//! RUN_ecc: Eccentric orbit (400 km x 8000 km altitude) — varying orbital rate
//!          exercises Euler angle computation at different angular velocities.
//! RUN_equ: Equatorial orbit (i=0) — exercises gimbal-lock-adjacent sequences.
//!
//! Both use point-mass Earth gravity, RK4 at DT=0.03125s, 24h duration.
//! Euler angles are validated against JEOD's logged quaternion data.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    DynamicsConfig, EulerSequence, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, MassProperties, RotationalState, SimBody, Simulation, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::crossval_report;

/// Set up an Euler-style simulation from CSV initial conditions.
/// SIM_Euler uses the same mass/config as SIM_dyncomp RUN_2.
fn run_euler_test(csv_filename: &str, label: &str, test_name: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "SIM_Euler CSV not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let records = load_euler_csv(&csv_path);
    assert!(records.len() > 100);
    let init = &records[0];

    // SIM_Euler uses ISS-like mass properties
    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: init.quaternion,
            ang_vel_body: DVec3::ZERO, // SIM_Euler initializes with zero angular velocity
        }),
        mass: Some(mass_props),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        euler_sequence: Some(EulerSequence::XYZ),
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_Euler {label}, {} points",
        records.len()
    );

    let mut max_angle_err = [0.0_f64; 3];
    let mut max_quat_err = 0.0_f64;

    for record in &records[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);

        let euler = body.euler_angles.unwrap_or_else(|| {
            panic!(
                "Simulation did not compute Euler angles at t={}",
                record.time
            )
        });

        // Compute expected Euler angles from JEOD's quaternion
        let jeod_t = record.quaternion.left_quat_to_transformation();
        let jeod_euler = jeod_math::compute_euler_angles_from_matrix(&jeod_t, EulerSequence::XYZ);

        let quat_err =
            quaternion_angle_error(&body.rot.as_ref().unwrap().quaternion, &record.quaternion);
        max_quat_err = max_quat_err.max(quat_err);

        for k in 0..3 {
            let err = angle_diff(euler[k], jeod_euler[k]);
            max_angle_err[k] = max_angle_err[k].max(err);
        }

        if (record.time % 7200.0).abs() < 6.1 {
            println!(
                "  t={:6.0}s: quat_err={:.6e} rad  euler_err=[{:.6e}, {:.6e}, {:.6e}] rad",
                record.time, quat_err, max_angle_err[0], max_angle_err[1], max_angle_err[2]
            );
        }
    }

    println!("  Max quaternion error: {:.6e} rad", max_quat_err);
    println!(
        "  Max Euler angle errors: [{:.6e}, {:.6e}, {:.6e}] rad",
        max_angle_err[0], max_angle_err[1], max_angle_err[2]
    );

    crossval_report(
        test_name,
        &[
            ("quaternion", max_quat_err, 0.01, "rad"),
            ("euler_roll", max_angle_err[0], f64::INFINITY, "rad"),
            ("euler_pitch", max_angle_err[1], f64::INFINITY, "rad"),
            ("euler_yaw", max_angle_err[2], f64::INFINITY, "rad"),
        ],
    );

    assert!(
        max_quat_err < 0.01,
        "{label}: quaternion error {max_quat_err:.2e} rad exceeds 0.01 rad"
    );
    for (k, &err) in max_angle_err.iter().enumerate() {
        assert!(
            err < 1e-6,
            "{label}: Euler angle[{k}] error {err:.2e} rad exceeds 1e-6 rad",
        );
    }
}

#[test]
fn tier3_simulation_euler_ecc() {
    run_euler_test(
        "euler_ecc_euler.csv",
        "RUN_ecc (eccentric)",
        "tier3_simulation_euler_ecc",
    );
}

#[test]
fn tier3_simulation_euler_equ() {
    run_euler_test(
        "euler_equ_euler.csv",
        "RUN_equ (equatorial)",
        "tier3_simulation_euler_equ",
    );
}
