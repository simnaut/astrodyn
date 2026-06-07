// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy-vs-Simulation parity tests: gravity gradient torque and external torque.

mod common;

use astrodyn::{
    DynamicsConfig, GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
    JeodQuat, SixDofState, TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_bevy::{
    DynamicsConfigC, ExternalForceC, ExternalTorqueC, GravityControlsC, GravityTorqueC,
    IntegrationDtR, MassPropertiesC, RotationalStateC, TranslationalStateC,
};
use bevy::prelude::*;
use glam::{DMat3, DVec3};

use common::*;

// ── Scenario D: Gravity gradient torque, 6-DOF ──

#[test]
fn bevy_parity_gravity_torque_sixdof() {
    println!("Scenario D: Gravity gradient torque, 6-DOF");

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(astrodyn_bevy::AstrodynPlugin);

    let planet = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::FrameUid::of::<
                astrodyn::PlanetInertial<astrodyn::Earth>,
            >()),
            Name::new("Earth"),
            astrodyn_bevy::GravitySourceC(earth_source()),
            astrodyn_bevy::SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(&format!(
                "bevy-parity-gravity-torque-b1-{}",
                NEXT_BODY_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ))),
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
            RotationalStateC::from(tumble_rot()),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(
                    planet,
                    GravityGradient::Compute,
                )],
            }),
            GravityTorqueC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        earth_source(),
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);

    let mut body = new_sim_body_sixdof(earth_idx, true);
    body.compute_gravity_gradient = true;
    sim.add_body(body);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: astrodyn::typed_bridge::trans_typed_to_raw(&body.trans),
        rot: astrodyn::typed_bridge::rot_typed_to_raw(&body.rot.unwrap()),
    };

    assert_sixdof_eq("Bevy vs Sim (grav torque)", &bevy_state, &sim_state);
}

// ── Scenario G: External torque via per-body functions ──

#[test]
fn bevy_parity_gravity_torque_external_torque_per_body() {
    println!("Scenario G: External torque via per-body functions");

    let mass_props = astrodyn::MassPropertiesTyped::<astrodyn::SelfRef>::with_inertia(
        uom::si::f64::Mass::new::<uom::si::mass::kilogram>(400_000.0),
        astrodyn::InertiaTensor::<astrodyn::BodyFrame<astrodyn::SelfRef>>::from_dmat3_unchecked(
            DMat3::from_cols(
                DVec3::new(1.02e8, -6.96e6, -5.48e6),
                DVec3::new(-6.96e6, 0.91e8, 5.90e5),
                DVec3::new(-5.48e6, 5.90e5, 1.64e8),
            ),
        ),
        astrodyn::Position::<astrodyn::StructuralFrame<astrodyn::SelfRef>>::from_raw_si(
            DVec3::new(-3.0, -1.5, 4.0),
        ),
    );
    // Cached untyped form for kernel calls below (`accumulate_gravity`,
    // `collect_and_resolve_forces`, `integrate_body` all consume raw
    // `MassProperties`).
    let mass_props_raw = mass_props.to_untyped();

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
        controls: vec![GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        )],
    };

    let external_torque = DVec3::new(10.0, 0.0, 0.0);
    let step_dt = 10.0;
    let num_steps = 100;

    // Path A
    let mut trans_a = iss_trans().to_untyped();
    let mut rot_a = tumble_rot().to_untyped();
    for step in 0..num_steps {
        let torque = if (10..20).contains(&step) {
            external_torque
        } else {
            DVec3::ZERO
        };
        let grav = astrodyn::accumulate_gravity(trans_a.position, &controls, DVec3::ZERO, |_| {
            Some(astrodyn::ResolvedSource {
                source: &earth_src,
                rotation: None,
                position: DVec3::ZERO,
                delta_c20: 0.0,
                has_delta_coeffs: false,
            })
        });
        let (total, _) = astrodyn::collect_and_resolve_forces(
            None,
            None,
            None,
            Some(&rot_a),
            DMat3::IDENTITY,
            Some(&mass_props_raw),
            grav.grav_accel,
        );
        astrodyn::integrate_body(
            &config,
            &mut trans_a,
            Some(&mut rot_a),
            Some(&mass_props_raw),
            |pos, _vel, _time_frac| {
                astrodyn::accumulate_gravity(pos, &controls, DVec3::ZERO, |_| {
                    Some(astrodyn::ResolvedSource {
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
            astrodyn::StructuralWrench::NONE,
            step_dt,
            1.0,
            astrodyn::IntegratorType::Rk4,
            None,
            None,
            None,
        );
    }

    // Path B: Simulation::step() pipeline with set_body_external_torque
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, step_dt);
    let mut earth_entry = GravitySourceEntry::new(
        earth_src,
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                earth_idx,
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("bevy-parity-gravity-torque-2")
    });
    sim.validate().unwrap();

    for step in 0..num_steps {
        let torque = if (10..20).contains(&step) {
            external_torque
        } else {
            DVec3::ZERO
        };
        sim.set_body_external_torque(0, torque);
        sim.step().expect("step failed");
    }

    let state_a = SixDofState {
        trans: trans_a,
        rot: rot_a,
    };
    let sim_body = sim.body(0);
    let state_b = SixDofState {
        trans: astrodyn::typed_bridge::trans_typed_to_raw(&sim_body.trans),
        rot: astrodyn::typed_bridge::rot_typed_to_raw(&sim_body.rot.unwrap()),
    };
    assert_sixdof_eq(
        "Per-body functions vs Simulation::step() (ext torque)",
        &state_a,
        &state_b,
    );
}

// ── Gravity torque parity (elliptical + with rate) ──

fn run_gravity_torque_parity(
    label: &str,
    trans: TranslationalState,
    rot: astrodyn::RotationalStateTyped<astrodyn::SelfRef>,
) {
    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(&format!(
                "bevy-parity-gravity-torque-b2-{}",
                NEXT_BODY_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ))),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(trans),
            RotationalStateC::from(rot),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(
                    planet,
                    GravityGradient::Compute,
                )],
            }),
            GravityTorqueC::default(),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&trans),
        rot: Some(rot),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                earth_idx,
                GravityGradient::Compute,
            )],
        },
        compute_gravity_gradient: true,
        ..VehicleConfig::named("bevy-parity-gravity-torque-1")
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: astrodyn::typed_bridge::trans_typed_to_raw(&sim_body.trans),
        rot: astrodyn::typed_bridge::rot_typed_to_raw(&sim_body.rot.unwrap()),
    };
    assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);
}

#[test]
fn bevy_parity_gravity_torque_run10c_gravity_torque_elliptical() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    let rot = {
        // Normalize the deliberately-non-trivial quaternion at the
        // test boundary: typed `RotationalStateC` requires unit-norm.
        let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
        q.normalize();
        astrodyn::RotationalStateTyped::<astrodyn::SelfRef>::new(
            astrodyn::BodyAttitude::<astrodyn::SelfRef>::from_jeod_quat(q),
            astrodyn::AngularVelocity::<astrodyn::BodyFrame<astrodyn::SelfRef>>::from_raw_si(
                DVec3::ZERO,
            ),
        )
    };
    run_gravity_torque_parity("run10c_grav_torque_ecc", ecc_trans, rot);
}

#[test]
fn bevy_parity_gravity_torque_run10d_gravity_torque_elliptical_rate() {
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

    let rot = {
        // Normalize at boundary; see comment above on tumble quat.
        let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
        q.normalize();
        astrodyn::RotationalStateTyped::<astrodyn::SelfRef>::new(
            astrodyn::BodyAttitude::<astrodyn::SelfRef>::from_jeod_quat(q),
            astrodyn::AngularVelocity::<astrodyn::BodyFrame<astrodyn::SelfRef>>::from_raw_si(
                init_ang_vel,
            ),
        )
    };

    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(&format!(
                "bevy-parity-gravity-torque-b3-{}",
                NEXT_BODY_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ))),
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
            RotationalStateC::from(rot),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, GravityGradient::Skip)],
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
            controls: vec![GravityControl::new_spherical(
                earth_idx,
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("bevy-parity-gravity-torque-0")
    });
    sim.validate().unwrap();

    for step in 0..n_steps {
        let t = step as f64 * dt;

        let quat = sim
            .body(0)
            .rot
            .as_ref()
            .unwrap()
            .q_inertial_body
            .to_jeod_quat();
        let (force, torque) = force_torque_fn(t, dt, &quat);

        let mut ext_f = app.world_mut().get_mut::<ExternalForceC>(vehicle).unwrap();
        ext_f.0 = astrodyn::Force::<astrodyn::RootInertial>::from_raw_si(force);
        let mut ext_t = app.world_mut().get_mut::<ExternalTorqueC>(vehicle).unwrap();
        ext_t.0 = astrodyn::Torque::<astrodyn::BodyFrame<astrodyn::SelfRef>>::from_raw_si(torque);

        sim.set_body_external_force(0, force);
        sim.set_body_external_torque(0, torque);

        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(std::time::Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
        sim.step().expect("step failed");
    }

    let bevy_state = read_sixdof(app.world(), vehicle);
    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: astrodyn::typed_bridge::trans_typed_to_raw(&sim_body.trans),
        rot: astrodyn::typed_bridge::rot_typed_to_raw(&sim_body.rot.unwrap()),
    };
    assert_sixdof_eq(&format!("Bevy vs Sim ({label})"), &bevy_state, &sim_state);
}

#[test]
fn bevy_parity_gravity_torque_run9a_torque() {
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
fn bevy_parity_gravity_torque_run9c_force_torque() {
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
fn bevy_parity_gravity_torque_run9d_force_torque_rate() {
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

/// Per-call unique suffix for swept test-body identities (#664): helpers
/// spawning multiple bodies per App must mint distinct identities.
static NEXT_BODY_UID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
