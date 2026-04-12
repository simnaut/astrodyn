//! Bevy-vs-Simulation parity tests: gravity gradient torque and external torque.

mod parity_helpers;

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, ExternalForceC, ExternalTorqueC, GravityControlsC, GravityTorqueC,
    MassPropertiesC, RotationalStateC, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, VehicleConfig};
use jeod_sim::{
    DynamicsConfig, GravityControl, GravityControls, GravityModel, GravitySource, JeodQuat,
    MassProperties, RotationalState, SixDofState, TranslationalState,
};

use parity_helpers::*;

// ── Scenario D: Gravity gradient torque, 6-DOF ──

#[test]
fn tier3_bevy_gravity_torque_sixdof() {
    println!("Scenario D: Gravity gradient torque, 6-DOF");

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(bevy_jeod::JeodPlugin);

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            bevy_jeod::GravitySourceC(earth_source()),
            bevy_jeod::SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()),
            RotationalStateC(tumble_rot()),
            MassPropertiesC(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, true)],
            }),
            GravityTorqueC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(GravitySourceEntry::new(earth_source(), DVec3::ZERO, None));

    let mut body = new_sim_body_sixdof(earth_idx, true);
    body.compute_gravity_gradient = true;
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim (grav torque)", &bevy_state, &sim_state);
}

// ── Scenario G: External torque via per-body functions ──

#[test]
fn tier3_bevy_external_torque_per_body() {
    println!("Scenario G: External torque via per-body functions");

    let mass_props = MassProperties::with_inertia(
        400_000.0,
        DMat3::from_cols(
            DVec3::new(1.02e8, -6.96e6, -5.48e6),
            DVec3::new(-6.96e6, 0.91e8, 5.90e5),
            DVec3::new(-5.48e6, 5.90e5, 1.64e8),
        ),
        DVec3::new(-3.0, -1.5, 4.0),
    );

    let config = DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    };

    let earth_src = GravitySource {
        mu: MU_EARTH,
        model: GravityModel::PointMass,
    };
    let controls: GravityControls<usize> = GravityControls {
        controls: vec![GravityControl::new_spherical(0_usize, false)],
    };

    let external_torque = DVec3::new(10.0, 0.0, 0.0);
    let step_dt = 10.0;
    let num_steps = 100;

    // Path A
    let mut trans_a = iss_trans();
    let mut rot_a = tumble_rot();
    for step in 0..num_steps {
        let torque = if (10..20).contains(&step) {
            external_torque
        } else {
            DVec3::ZERO
        };
        let grav = jeod_sim::accumulate_gravity(trans_a.position, &controls, DVec3::ZERO, |_| {
            Some(jeod_sim::ResolvedSource {
                source: &earth_src,
                rotation: None,
                position: DVec3::ZERO,
                delta_c20: 0.0,
                has_delta_coeffs: false,
            })
        });
        let (total, _) = jeod_sim::collect_and_resolve_forces(
            None,
            None,
            None,
            Some(&rot_a),
            DMat3::IDENTITY,
            Some(&mass_props),
            grav.grav_accel,
        );
        jeod_sim::integrate_body(
            &config,
            &mut trans_a,
            Some(&mut rot_a),
            Some(&mass_props),
            |pos, _vel| {
                jeod_sim::accumulate_gravity(pos, &controls, DVec3::ZERO, |_| {
                    Some(jeod_sim::ResolvedSource {
                        source: &earth_src,
                        rotation: None,
                        position: DVec3::ZERO,
                        delta_c20: 0.0,
                        has_delta_coeffs: false,
                    })
                })
                .grav_accel
            },
            total.force,
            total.torque + torque,
            step_dt,
            1.0,
            jeod_sim::IntegratorType::Rk4,
            None,
        );
    }

    // Path B: Simulation::step() pipeline with set_body_external_torque
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = jeod_runner::Simulation::new(time, step_dt);
    let earth_idx = sim.add_source(GravitySourceEntry::new(earth_src, DVec3::ZERO, None));
    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();

    for step in 0..num_steps {
        let torque = if (10..20).contains(&step) {
            external_torque
        } else {
            DVec3::ZERO
        };
        sim.set_body_external_torque(0, torque);
        sim.step();
    }

    let state_a = SixDofState {
        trans: trans_a,
        rot: rot_a,
    };
    let sim_body = sim.body(0);
    let state_b = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq(
        "Per-body functions vs Simulation::step() (ext torque)",
        &state_a,
        &state_b,
    );
}

// ── Gravity torque parity (elliptical + with rate) ──

fn run_gravity_torque_parity(label: &str, trans: TranslationalState, rot: RotationalState) {
    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(trans),
            RotationalStateC(rot),
            MassPropertiesC(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, true)],
            }),
            GravityTorqueC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans,
        rot: Some(rot),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, true)],
        },
        compute_gravity_gradient: true,
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);
}

#[test]
fn tier3_bevy_run10c_gravity_torque_elliptical() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    let rot = RotationalState {
        quaternion: JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5),
        ang_vel_body: DVec3::ZERO,
    };
    run_gravity_torque_parity("run10c_grav_torque_ecc", ecc_trans, rot);
}

#[test]
fn tier3_bevy_run10d_gravity_torque_elliptical_rate() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_gravity_torque_parity("run10d_grav_torque_ecc_rate", ecc_trans, tumble_rot());
}

// ── External force/torque parity tests ──

const TORQUE_START: f64 = 1000.0;
const TORQUE_END: f64 = 2000.0;

fn in_torque_window(t: f64, dt: f64) -> bool {
    t + dt * 0.5 >= TORQUE_START && t + dt * 0.5 < TORQUE_END
}

fn run_external_parity(
    label: &str,
    init_ang_vel: DVec3,
    force_torque_fn: fn(f64, f64, &JeodQuat) -> (DVec3, DVec3),
) {
    let n_steps = 300;
    let dt = DT;

    // ── Bevy ──
    let mut app = new_bevy_app(dt);
    let planet = spawn_earth_source(&mut app);

    let rot = RotationalState {
        quaternion: JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5),
        ang_vel_body: init_ang_vel,
    };

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(iss_trans()),
            RotationalStateC(rot),
            MassPropertiesC(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            ExternalForceC::default(),
            ExternalTorqueC::default(),
        ))
        .id();

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(dt);
    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        rot: Some(rot),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();

    for step in 0..n_steps {
        let t = step as f64 * dt;

        let quat = sim.body(0).rot.as_ref().unwrap().quaternion;
        let (force, torque) = force_torque_fn(t, dt, &quat);

        let mut ext_f = app.world_mut().get_mut::<ExternalForceC>(vehicle).unwrap();
        ext_f.0 = force;
        let mut ext_t = app.world_mut().get_mut::<ExternalTorqueC>(vehicle).unwrap();
        ext_t.0 = torque;

        sim.set_body_external_force(0, force);
        sim.set_body_external_torque(0, torque);

        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(std::time::Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
        sim.step();
    }

    let bevy_state = read_sixdof(app.world(), vehicle);
    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);
}

#[test]
fn tier3_bevy_run9a_torque() {
    run_external_parity("run9a_torque", DVec3::ZERO, |t, dt, _quat| {
        let torque = if in_torque_window(t, dt) {
            DVec3::new(10.0, 0.0, 0.0)
        } else {
            DVec3::ZERO
        };
        (DVec3::ZERO, torque)
    });
}

#[test]
fn tier3_bevy_run9c_force_torque() {
    run_external_parity("run9c_force_torque", DVec3::ZERO, |t, dt, quat| {
        if in_torque_window(t, dt) {
            let t_inertial_body = quat.left_quat_to_transformation();
            let force_inertial = t_inertial_body.transpose() * DVec3::new(10.0, 0.0, 0.0);
            let torque = DVec3::new(10.0, 0.0, 0.0);
            (force_inertial, torque)
        } else {
            (DVec3::ZERO, DVec3::ZERO)
        }
    });
}

#[test]
fn tier3_bevy_run9d_force_torque_rate() {
    let init_ang_vel = DVec3::new(0.001, -0.0005, 0.001);
    run_external_parity("run9d_force_torque_rate", init_ang_vel, |t, dt, quat| {
        if in_torque_window(t, dt) {
            let t_inertial_body = quat.left_quat_to_transformation();
            let force_inertial = t_inertial_body.transpose() * DVec3::new(10.0, 0.0, 0.0);
            let torque = DVec3::new(10.0, 0.0, 0.0);
            (force_inertial, torque)
        } else {
            (DVec3::ZERO, DVec3::ZERO)
        }
    });
}
