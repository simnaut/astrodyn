//! Tier 3: SIM_dyncomp RUN_5B/5C — Elliptical orbit, 6-DOF (ISS mass)
//!
//! JEOD labels these "atmosphere comparison" runs, but drag is disabled
//! so the atmosphere model has no effect on the trajectory. These are
//! effectively point-mass 6-DOF tests with elliptical orbit ICs and ISS
//! mass/inertia — validating gravity + rotational dynamics on a different
//! orbit geometry than the typical (near-circular) RUN_2.
//!
//! RUN_5B: JEOD config uses F10.7 = 128.8 (solar mean)
//! RUN_5C: JEOD config uses F10.7 = 250.0 (solar max)

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    DynamicsConfig, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, MassProperties, RotationalState, SimBody, Simulation, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::crossval_report;

// ── RUN_5B: MET solar mean, elliptical orbit, no drag ──

#[test]
fn tier3_simulation_run5b_atmosphere_mean() {
    run_atmosphere_test("dyncomp_run5b_state.csv", "RUN_5B (solar mean)");
}

// ── RUN_5C: MET solar max, elliptical orbit, no drag ──

#[test]
fn tier3_simulation_run5c_atmosphere_max() {
    run_atmosphere_test("dyncomp_run5c_state.csv", "RUN_5C (solar max)");
}

/// Shared test body for RUN_5B/5C.
///
/// Both runs have identical physics (point-mass gravity, 6-DOF, drag off,
/// gravity torque off) with elliptical orbit ICs. We propagate with the
/// Simulation runner and compare against JEOD CSV.
fn run_atmosphere_test(csv_filename: &str, label: &str) {
    let csv_path = test_data_path(csv_filename);
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let trajectory = load_sixdof_trajectory(&csv_path);
    assert!(trajectory.len() >= 100);
    let init = &trajectory[0];

    // ISS mass properties (from Modified_data/mass.py set_mass_iss)
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

    // Drag OFF, gravity torque OFF (common_input defaults).
    // Gravity gradient is computed (gradient=true) but not used as torque.
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
            controls: vec![GravityControl::new_spherical(earth, true)], // gradient=true
        },
        // No drag, no gravity torque, no atmosphere interaction
        ..Default::default()
    });

    sim.validate().unwrap();

    println!("Tier 3 (Simulation): {label}, {} points", trajectory.len());

    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;
    let mut max_quat_error = 0.0_f64;

    for record in &trajectory[1..] {
        sim.step_until(record.time);

        let body = sim.body(0);
        let pos_error = (body.trans.position - record.position).length();
        let vel_error = (body.trans.velocity - record.velocity).length();
        max_pos_error = max_pos_error.max(pos_error);
        max_vel_error = max_vel_error.max(vel_error);

        if let Some(ref rot) = body.rot {
            let quat_error = quaternion_angle_error(&rot.quaternion, &record.quaternion);
            max_quat_error = max_quat_error.max(quat_error);
        }

        if (record.time % 7200.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:.3e} m  vel_err={:.3e} m/s  quat_err={:.6e} rad",
                record.time, pos_error, vel_error, max_quat_error
            );
        }
    }

    println!("  Max position error:  {:.6e} m", max_pos_error);
    println!("  Max velocity error:  {:.6e} m/s", max_vel_error);
    println!("  Max quaternion error: {:.6e} rad", max_quat_error);

    crossval_report(
        "tier3_simulation_run5c_atmosphere_max",
        &[
            ("position", max_pos_error, 0.5, "m"),
            ("velocity", max_vel_error, 0.001, "m/s"),
            ("quaternion", max_quat_error, 0.01, "rad"),
        ],
    );

    assert!(
        max_pos_error < 0.5,
        "{label}: position error {max_pos_error:.4} m exceeds 0.5 m"
    );
    assert!(
        max_vel_error < 0.001,
        "{label}: velocity error {max_vel_error:.6} m/s exceeds 0.001 m/s"
    );
    assert!(
        max_quat_error < 0.01,
        "{label}: quaternion error {max_quat_error:.2e} rad exceeds 0.01 rad"
    );
}
