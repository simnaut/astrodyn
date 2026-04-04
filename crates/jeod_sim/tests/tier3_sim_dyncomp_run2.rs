//! Tier 3: SIM_dyncomp RUN_2 — Point-mass gravity (3-DOF and 6-DOF)

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DQuat, DVec3};
use jeod_sim::{
    DynamicsConfig, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, MassProperties, RotationalState, SimBody, Simulation, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateSnapshot};

// ── Scenario 1: Point-mass 3-DOF (RUN_2) ──

#[test]
fn tier3_simulation_run2_3dof() {
    let csv_path = test_data_path("dyncomp_run2_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let trajectory = load_trans_trajectory(&csv_path);
    assert!(trajectory.len() > 100);

    let init = &trajectory[0];

    // Set up Simulation — point-mass gravity, no atmosphere, no interactions
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
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_2 point-mass 3-DOF, {} points",
        trajectory.len()
    );

    let mut report = CrossvalReport::new("tier3_simulation_run2_3dof");

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        report.accumulate(
            &StateSnapshot {
                position: Some(body.trans.position),
                velocity: Some(body.trans.velocity),
                ..Default::default()
            },
            &StateSnapshot {
                position: Some(record.position),
                velocity: Some(record.velocity),
                ..Default::default()
            },
        );
    }

    report.position_tol = Some([0.5; 3]);
    report.velocity_tol = Some([0.001; 3]);
    report.write();

    let max_pos_error = report
        .position
        .unwrap()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let max_vel_error = report
        .velocity
        .unwrap()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);

    println!("  Max position error: {:.6e} m", max_pos_error);
    println!("  Max velocity error: {:.6e} m/s", max_vel_error);

    assert!(
        max_pos_error < 0.5,
        "Position error {max_pos_error:.2} m exceeds 0.5 m over 8 hours"
    );
    assert!(
        max_vel_error < 0.001,
        "Velocity error {max_vel_error:.6} m/s exceeds 0.001 m/s over 8 hours"
    );
}

// ── Scenario 2: Point-mass 6-DOF with ISS mass (RUN_2) ──

#[test]
fn tier3_simulation_run2_6dof() {
    let csv_path = test_data_path("dyncomp_run2_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}",
        csv_path.display()
    );

    let trajectory = load_sixdof_trajectory(&csv_path);
    assert!(trajectory.len() > 100);

    let init = &trajectory[0];

    // ISS mass properties from Modified_data/mass.py
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
            ang_vel_body: init.ang_vel,
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
        ..Default::default()
    });

    sim.validate().unwrap();

    println!(
        "Tier 3 (Simulation): RUN_2 point-mass 6-DOF, {} points",
        trajectory.len()
    );

    let mut report = CrossvalReport::new("tier3_simulation_run2_6dof");

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let rot = body.rot.as_ref().unwrap();
        report.accumulate(
            &StateSnapshot {
                position: Some(body.trans.position),
                velocity: Some(body.trans.velocity),
                quaternion: Some(rot.quaternion.to_glam()),
                ang_vel: Some(rot.ang_vel_body),
                ..Default::default()
            },
            &StateSnapshot {
                position: Some(record.position),
                velocity: Some(record.velocity),
                quaternion: Some(record.quaternion.to_glam()),
                ang_vel: Some(record.ang_vel),
                ..Default::default()
            },
        );
    }

    report.position_tol = Some([0.5; 3]);
    report.velocity_tol = Some([0.001; 3]);
    report.quat_angle_tol = Some(0.01);
    report.ang_vel_tol = Some([1e-5; 3]);
    report.write();

    let max_pos_error = report
        .position
        .unwrap()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let max_vel_error = report
        .velocity
        .unwrap()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let max_quat_error = report.quat_angle.unwrap();
    let max_omega_error = report
        .ang_vel
        .unwrap()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);

    println!("  Max position error:  {:.6e} m", max_pos_error);
    println!("  Max velocity error:  {:.6e} m/s", max_vel_error);
    println!("  Max quaternion error: {:.6e} rad", max_quat_error);
    println!("  Max omega error:     {:.6e} rad/s", max_omega_error);

    assert!(
        max_pos_error < 0.5,
        "Position error {max_pos_error:.2} m exceeds 0.5 m"
    );
    assert!(
        max_vel_error < 0.001,
        "Velocity error {max_vel_error:.6} m/s exceeds 0.001 m/s"
    );
    assert!(
        max_quat_error < 0.01,
        "Quaternion error {max_quat_error:.2e} rad exceeds 0.01 rad"
    );
    assert!(
        max_omega_error < 1e-5,
        "Omega error {max_omega_error:.2e} rad/s exceeds 1e-5 rad/s"
    );
}
