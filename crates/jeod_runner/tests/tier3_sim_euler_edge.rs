//! Tier 3: SIM_Euler edge-case cross-validation
//!
//! RUN_ecc: Eccentric orbit (400 km x 8000 km altitude) — varying orbital rate
//!          exercises Euler angle computation at different angular velocities.
//! RUN_equ: Equatorial orbit (i=0) — exercises gimbal-lock-adjacent sequences.
//!
//! Both use point-mass Earth gravity, RK4 at the SIM_Euler S_define step size, 24h duration.
//! Euler angles are validated against JEOD's logged quaternion data.

// `compute_euler_angles_from_matrix` is deprecated in favor of its typed
// sibling; this Tier 3 test keeps the bare variant while the derived-state
// consumers migrate.
#![allow(deprecated)]

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_runner::{
    DerivedStateConfig, GravitySourceEntry, RotationModel, Simulation, VehicleConfig,
};
use jeod_sim::{
    EulerSequence, GravityControl, GravityControls, GravityModel, GravitySource, MassProperties,
    RotationalState, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

fn load_mu_earth() -> f64 {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu from GGM05C")
}

/// Set up an Euler-style simulation from CSV initial conditions.
/// SIM_Euler uses the same mass/config as SIM_dyncomp RUN_2.
fn run_euler_test(csv_filename: &str, label: &str, test_name: &str, quat_tol: f64, euler_tol: f64) {
    let mu_earth = load_mu_earth();
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

    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );
    let dt = jeod_test_data::s_define::load_dynamics_dt(
        &jeod_root.join("models/dynamics/derived_state/verif/SIM_Euler/S_define"),
    );

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        rot: Some(RotationalState {
            quaternion: init.quaternion,
            ang_vel_body: DVec3::ZERO, // SIM_Euler initializes with zero angular velocity
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        derived: DerivedStateConfig {
            euler_sequence: Some(EulerSequence::XYZ),
            ..Default::default()
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): SIM_Euler {label}, {} points",
        records.len()
    );

    let mut our_states = Vec::with_capacity(records.len() - 1);
    let mut ref_states = Vec::with_capacity(records.len() - 1);
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

        our_states.push(StateLog {
            time: record.time,
            quaternion: Some(body.rot.as_ref().unwrap().quaternion.to_glam()),
            ..Default::default()
        });
        ref_states.push(StateLog {
            time: record.time,
            quaternion: Some(record.quaternion.to_glam()),
            ..Default::default()
        });

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

    let mut report = CrossvalReport::compute(test_name, &our_states, &ref_states);
    report.add_extra("euler_roll", max_angle_err[0], "rad");
    assert!(max_angle_err[0] < euler_tol, "euler_roll");
    report.add_extra("euler_pitch", max_angle_err[1], "rad");
    assert!(max_angle_err[1] < euler_tol, "euler_pitch");
    report.add_extra("euler_yaw", max_angle_err[2], "rad");
    assert!(max_angle_err[2] < euler_tol, "euler_yaw");
    report.write();

    report.assert_quat_angle(quat_tol);

    for (k, &err) in max_angle_err.iter().enumerate() {
        assert!(
            err < euler_tol,
            "{label}: Euler angle[{k}] error {err:.2e} rad exceeds {euler_tol:.2e} rad",
        );
    }
}

#[test]
fn tier3_simulation_euler_ecc() {
    run_euler_test(
        "euler_ecc_euler.csv",
        "RUN_ecc (eccentric)",
        "tier3_simulation_euler_ecc",
        1e-10,
        1e-10,
    );
}

#[test]
fn tier3_simulation_euler_equ() {
    run_euler_test(
        "euler_equ_euler.csv",
        "RUN_equ (equatorial)",
        "tier3_simulation_euler_equ",
        1e-10,
        1e-10,
    );
}
