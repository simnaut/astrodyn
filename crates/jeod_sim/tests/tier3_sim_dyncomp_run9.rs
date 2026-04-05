//! Tier 3: SIM_dyncomp RUN_9A/9C/9D — External force/torque

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_sim::{
    DynamicsConfig, GravityControl, GravityControls, GravityModel, GravitySource, JeodQuat,
    MassProperties, RotationalState, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};

// ── RUN_9A: External torque, 6-DOF ──
//
// RUN_9A applies [10, 0, 0] N·m structural-frame torque from t=1000s to t=2000s.
// The Simulation runner doesn't natively support time-scheduled external forces,
// so we step manually and inject the torque by modifying total_force between
// force_collection and integration. Since step() is monolithic, we instead
// step one dt at a time and set the body's total_force.torque after each step's
// force collection would normally produce zero torque. We compensate by adding
// the external torque to what collect_and_resolve_forces produces.
//
// This exercises the same jeod_sim::integrate_body code path as the Simulation
// runner, just with manual torque injection.

#[test]
fn tier3_simulation_run9a_torque() {
    let csv_path = test_data_path("dyncomp_run9a_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() >= 100);
    let init = &trajectory[0];

    // ISS mass properties (from Modified_data/mass.py)
    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

    // Use per-body functions directly for torque injection.
    // This still validates jeod_sim's integrate_body and accumulate_gravity.
    let mut trans = TranslationalState {
        position: init.composite_body.position,
        velocity: init.composite_body.velocity,
    };
    let mut rot = RotationalState {
        quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
        ang_vel_body: init.composite_body.ang_vel,
    };

    let config = DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    };

    let gravity_controls: GravityControls<usize> = GravityControls {
        controls: vec![GravityControl::new_spherical(0_usize, false)],
    };

    let earth_source = GravitySource {
        mu: MU_EARTH,
        model: GravityModel::PointMass,
    };

    println!(
        "Tier 3 (jeod_sim per-body): RUN_9A torque 6-DOF, {} points",
        trajectory.len()
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let mut current_time = init.time;

    for record in &trajectory[1..] {
        while current_time + DT <= record.time + 0.001 {
            // Gravity (per-body function)
            let grav = jeod_sim::accumulate_gravity(
                trans.position,
                &gravity_controls,
                DVec3::ZERO,
                |_| {
                    Some(jeod_sim::ResolvedSource {
                        source: &earth_source,
                        rotation: None,
                        position: DVec3::ZERO,
                    })
                },
            );

            // External torque: [10, 0, 0] N·m in body frame during [1000, 2000)s
            let external_torque = if (999.999..1999.999).contains(&current_time) {
                DVec3::new(10.0, 0.0, 0.0)
            } else {
                DVec3::ZERO
            };

            // Force collection (no interactions, just gravity)
            let (total, _derivs) = jeod_sim::collect_and_resolve_forces(
                None,
                None,
                None,
                Some(&rot),
                DMat3::IDENTITY,
                Some(&mass_props),
                grav.grav_accel,
            );

            // Integration with external torque added.
            // Gravity recomputed at each RK4 intermediate state via closure.
            let gravity_fn = |pos: DVec3| {
                let r = pos.length();
                pos * (-MU_EARTH / (r * r * r))
            };
            jeod_sim::integrate_body(
                &config,
                &mut trans,
                Some(&mut rot),
                Some(&mass_props),
                gravity_fn,
                total.force,
                total.torque + external_torque,
                DT,
            );
            current_time += DT;
        }

        // Handle fractional remainder
        let remainder = record.time - current_time;
        if remainder > 0.001 {
            let grav = jeod_sim::accumulate_gravity(
                trans.position,
                &gravity_controls,
                DVec3::ZERO,
                |_| {
                    Some(jeod_sim::ResolvedSource {
                        source: &earth_source,
                        rotation: None,
                        position: DVec3::ZERO,
                    })
                },
            );
            let external_torque = if (999.999..1999.999).contains(&current_time) {
                DVec3::new(10.0, 0.0, 0.0)
            } else {
                DVec3::ZERO
            };
            let (total, _) = jeod_sim::collect_and_resolve_forces(
                None,
                None,
                None,
                Some(&rot),
                DMat3::IDENTITY,
                Some(&mass_props),
                grav.grav_accel,
            );
            let gravity_fn = |pos: DVec3| {
                let r = pos.length();
                pos * (-MU_EARTH / (r * r * r))
            };
            jeod_sim::integrate_body(
                &config,
                &mut trans,
                Some(&mut rot),
                Some(&mass_props),
                gravity_fn,
                total.force,
                total.torque + external_torque,
                remainder,
            );
            current_time += remainder;
        }

        let pos_error = (trans.position - record.composite_body.position).length();
        let quat_error =
            dquat_angle_error(rot.quaternion.to_glam(), record.composite_body.quaternion);
        let omega_error = (rot.ang_vel_body - record.composite_body.ang_vel).length();

        if (record.time % 3600.0).abs() < 30.1 {
            println!(
                "  t={:6.0}s: pos_err={:10.4} m  quat_err={:.6e} rad  omega_err={:.6e}",
                record.time, pos_error, quat_error, omega_error
            );
        }

        our_states.push(StateLog {
            time: record.time,
            position: Some(trans.position),
            velocity: Some(trans.velocity),
            quaternion: Some(rot.quaternion.to_glam()),
            ang_vel: Some(rot.ang_vel_body),
            ..Default::default()
        });
    }

    // Reference states from JEOD CSV
    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
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

    // Post-process: compute errors
    let report = CrossvalReport::compute("tier3_simulation_run9a_torque", &our_states, &ref_states);
    report.write();

    println!(
        "  Max position error:  {:.6e} m",
        report.max_position_component()
    );
    println!(
        "  Max velocity error:  {:.6e} m/s",
        report.max_velocity_component()
    );
    println!(
        "  Max quaternion error: {:.6e} rad",
        report.max_quat_angle()
    );
    println!(
        "  Max omega error:     {:.6e} rad/s",
        report.max_ang_vel_component()
    );

    report.assert_position([9.965e-7, 1.284e-6, 1.103e-6]);
    report.assert_velocity([9.763e-10, 1.623e-9, 1.232e-9]);
    report.assert_quat_angle(4.426e-8);
    report.assert_ang_vel([3.558e-20, 4.447e-21, 7.116e-21]);
}

// ── RUN_9C: External force + torque, zero inertial rate ──
//
// ISS mass, force [10,0,0] N + torque [10,0,0] N·m during t=1000-2000s.

#[test]
fn tier3_simulation_run9c_force_torque() {
    let csv_path = test_data_path("dyncomp_run9c_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() >= 100);
    let init = &trajectory[0];

    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

    let mut trans = TranslationalState {
        position: init.composite_body.position,
        velocity: init.composite_body.velocity,
    };
    let mut rot = RotationalState {
        quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
        ang_vel_body: init.composite_body.ang_vel,
    };

    let config = DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    };

    let gravity_controls: GravityControls<usize> = GravityControls {
        controls: vec![GravityControl::new_spherical(0_usize, false)],
    };

    let earth_source = GravitySource {
        mu: MU_EARTH,
        model: GravityModel::PointMass,
    };

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let mut current_time = init.time;

    for record in &trajectory[1..] {
        while current_time + DT <= record.time + 0.001 {
            let grav = jeod_sim::accumulate_gravity(
                trans.position,
                &gravity_controls,
                DVec3::ZERO,
                |_| {
                    Some(jeod_sim::ResolvedSource {
                        source: &earth_source,
                        rotation: None,
                        position: DVec3::ZERO,
                    })
                },
            );

            // External force [10,0,0] N and torque [10,0,0] N·m in
            // structural frame during [1000, 2000)s. Force must be rotated
            // to inertial frame; torque stays in body frame.
            let (ext_force_struct, external_torque) = if (999.999..1999.999).contains(&current_time)
            {
                (DVec3::new(10.0, 0.0, 0.0), DVec3::new(10.0, 0.0, 0.0))
            } else {
                (DVec3::ZERO, DVec3::ZERO)
            };

            let t_inertial_body = rot.quaternion.left_quat_to_transformation();
            let external_force_inertial = t_inertial_body.transpose() * ext_force_struct;

            let (total, _) = jeod_sim::collect_and_resolve_forces(
                None,
                None,
                None,
                Some(&rot),
                DMat3::IDENTITY,
                Some(&mass_props),
                grav.grav_accel,
            );

            let gravity_fn = |pos: DVec3| {
                let r = pos.length();
                pos * (-MU_EARTH / (r * r * r))
            };
            jeod_sim::integrate_body(
                &config,
                &mut trans,
                Some(&mut rot),
                Some(&mass_props),
                gravity_fn,
                total.force + external_force_inertial,
                total.torque + external_torque,
                DT,
            );
            current_time += DT;
        }

        our_states.push(StateLog {
            time: record.time,
            position: Some(trans.position),
            velocity: Some(trans.velocity),
            quaternion: Some(rot.quaternion.to_glam()),
            ang_vel: Some(rot.ang_vel_body),
            ..Default::default()
        });
    }

    // Reference states from JEOD CSV
    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
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

    // Post-process: compute errors
    let report = CrossvalReport::compute(
        "tier3_simulation_run9c_force_torque",
        &our_states,
        &ref_states,
    );
    report.write();

    println!(
        "RUN_9C: max pos={:.4} m  vel={:.6} m/s  quat={:.6e} rad  omega={:.6e} rad/s",
        report.max_position_component(),
        report.max_velocity_component(),
        report.max_quat_angle(),
        report.max_ang_vel_component(),
    );

    report.assert_position([7.679e-5, 1.2e-4, 8.628e-5]);
    report.assert_velocity([8.526e-8, 1.269e-7, 1.062e-7]);
    report.assert_quat_angle(4.426e-8);
    report.assert_ang_vel([3.558e-20, 4.447e-21, 7.116e-21]);
}

// ── RUN_9D: External force + torque, with orbit rate ──

#[test]
fn tier3_simulation_run9d_force_torque_rate() {
    let csv_path = test_data_path("dyncomp_run9d_state.csv");
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() >= 100);
    let init = &trajectory[0];

    let inertia = DMat3::from_cols(
        DVec3::new(1.02e8, -6.96e6, -5.48e6),
        DVec3::new(-6.96e6, 0.91e8, 5.90e5),
        DVec3::new(-5.48e6, 5.90e5, 1.64e8),
    );
    let mass_props = MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));

    let mut trans = TranslationalState {
        position: init.composite_body.position,
        velocity: init.composite_body.velocity,
    };
    let mut rot = RotationalState {
        quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
        ang_vel_body: init.composite_body.ang_vel,
    };

    let config = DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    };

    let gravity_controls: GravityControls<usize> = GravityControls {
        controls: vec![GravityControl::new_spherical(0_usize, false)],
    };

    let earth_source = GravitySource {
        mu: MU_EARTH,
        model: GravityModel::PointMass,
    };

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let mut current_time = init.time;

    for record in &trajectory[1..] {
        while current_time + DT <= record.time + 0.001 {
            let grav = jeod_sim::accumulate_gravity(
                trans.position,
                &gravity_controls,
                DVec3::ZERO,
                |_| {
                    Some(jeod_sim::ResolvedSource {
                        source: &earth_source,
                        rotation: None,
                        position: DVec3::ZERO,
                    })
                },
            );

            let (ext_force_struct, external_torque) = if (999.999..1999.999).contains(&current_time)
            {
                (DVec3::new(10.0, 0.0, 0.0), DVec3::new(10.0, 0.0, 0.0))
            } else {
                (DVec3::ZERO, DVec3::ZERO)
            };

            let t_inertial_body = rot.quaternion.left_quat_to_transformation();
            let external_force_inertial = t_inertial_body.transpose() * ext_force_struct;

            let (total, _) = jeod_sim::collect_and_resolve_forces(
                None,
                None,
                None,
                Some(&rot),
                DMat3::IDENTITY,
                Some(&mass_props),
                grav.grav_accel,
            );

            let gravity_fn = |pos: DVec3| {
                let r = pos.length();
                pos * (-MU_EARTH / (r * r * r))
            };
            jeod_sim::integrate_body(
                &config,
                &mut trans,
                Some(&mut rot),
                Some(&mass_props),
                gravity_fn,
                total.force + external_force_inertial,
                total.torque + external_torque,
                DT,
            );
            current_time += DT;
        }

        our_states.push(StateLog {
            time: record.time,
            position: Some(trans.position),
            velocity: Some(trans.velocity),
            quaternion: Some(rot.quaternion.to_glam()),
            ang_vel: Some(rot.ang_vel_body),
            ..Default::default()
        });
    }

    // Reference states from JEOD CSV
    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
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

    // Post-process: compute errors
    let report = CrossvalReport::compute(
        "tier3_simulation_run9d_force_torque_rate",
        &our_states,
        &ref_states,
    );
    report.write();

    println!(
        "RUN_9D: max pos={:.4} m  vel={:.6} m/s  quat={:.6e} rad  omega={:.6e} rad/s",
        report.max_position_component(),
        report.max_velocity_component(),
        report.max_quat_angle(),
        report.max_ang_vel_component(),
    );

    report.assert_position([5.278e-3, 8.255e-3, 6.635e-3]);
    report.assert_velocity([5.911e-6, 9.056e-6, 7.276e-6]);
    report.assert_quat_angle(4.426e-8);
    report.assert_ang_vel([1.651e-18, 1.367e-18, 6.262e-19]);
}
