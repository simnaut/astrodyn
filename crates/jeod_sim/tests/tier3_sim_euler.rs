//! Tier 3: SIM_Euler cross-validation (derived_state/verif/SIM_Euler)
//!
//! Uses the RUN_2 point-mass 6-DOF trajectory (which has quaternion data)
//! to validate Euler angle computation through the Simulation pipeline.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    DynamicsConfig, EulerSequence, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, JeodQuat, MassProperties, RotationModel, RotationalState, SimBody,
    Simulation, SimulationTime, TranslationalState,
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

#[test]
fn tier3_simulation_euler() {
    let mu_earth = load_mu_earth();
    let csv_path = test_data_path("dyncomp_run2_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    // ISS mass properties (from Modified_data/mass.py)
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
    let dt =
        jeod_test_data::s_define::load_dynamics_dt(&jeod_root.join("verif/SIM_dyncomp/S_define"));

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(GravitySourceEntry {
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
    });

    sim.add_body(SimBody {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init.composite_body.ang_vel,
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
        "Tier 3 (Simulation): Euler angles via RUN_2 6-DOF, {} points",
        trajectory.len()
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let mut ref_states = Vec::with_capacity(trajectory.len() - 1);
    let mut max_angle_err = [0.0_f64; 3];
    let mut max_quat_err = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);

        // Verify Euler angles are populated
        let euler = body.euler_angles.unwrap_or_else(|| {
            panic!(
                "Simulation did not compute Euler angles at t={}",
                record.time
            )
        });

        // Compute expected Euler angles from JEOD's quaternion for comparison
        let jeod_t =
            JeodQuat::from_glam(record.composite_body.quaternion).left_quat_to_transformation();
        let jeod_euler = jeod_math::compute_euler_angles_from_matrix(&jeod_t, EulerSequence::XYZ);

        // Also check quaternion error to understand the attitude tracking
        let quat_err = dquat_angle_error(
            body.rot.as_ref().unwrap().quaternion.to_glam(),
            record.composite_body.quaternion,
        );
        max_quat_err = max_quat_err.max(quat_err);

        for k in 0..3 {
            let err = angle_diff(euler[k], jeod_euler[k]);
            max_angle_err[k] = max_angle_err[k].max(err);
        }

        our_states.push(StateLog {
            time: record.time,
            acceleration: Some(body.frame_derivs.trans_accel),
            quaternion: Some(body.rot.as_ref().unwrap().quaternion.to_glam()),
            ang_vel: Some(body.rot.as_ref().unwrap().ang_vel_body),
            ang_accel: Some(body.frame_derivs.rot_accel),
            ..Default::default()
        });
        ref_states.push(StateLog {
            time: record.time,
            acceleration: record.derivs.as_ref().map(|d| d.trans_accel),
            quaternion: Some(record.composite_body.quaternion),
            ang_vel: Some(record.composite_body.ang_vel),
            ang_accel: record.derivs.as_ref().map(|d| d.rot_accel),
            ..Default::default()
        });

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: quat_err={:.6e} rad  euler_err=[{:.6e}, {:.6e}, {:.6e}] rad",
                record.time, quat_err, max_angle_err[0], max_angle_err[1], max_angle_err[2]
            );
        }
    }

    println!("  Max quaternion error: {:.6e} rad", max_quat_err);

    let mut report = CrossvalReport::compute("tier3_simulation_euler", &our_states, &ref_states);
    report.add_extra("euler_roll", max_angle_err[0], "rad");
    assert!(max_angle_err[0] < 1.846e-13, "euler_roll");
    report.add_extra("euler_pitch", max_angle_err[1], "rad");
    assert!(max_angle_err[1] < 8.674e-14, "euler_pitch");
    report.add_extra("euler_yaw", max_angle_err[2], "rad");
    assert!(max_angle_err[2] < 1.103e-13, "euler_yaw");
    report.write();

    println!(
        "  Max Euler angle errors: [{:.6e}, {:.6e}, {:.6e}] rad",
        max_angle_err[0], max_angle_err[1], max_angle_err[2]
    );

    report.assert_quat_angle(4.426e-8);
}
