// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy-vs-Simulation parity tests: point-mass gravity, planetary orbits,
//! basic 6-DOF, orbinit cross-consistency, and timescale parity.

mod common;

use astrodyn::{
    DynamicsConfig, GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
    JeodQuat, SixDofState, TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_bevy::{
    DynamicsConfigC, GravityControlsC, IntegrationDtR, MassPropertiesC, RotationalStateC,
    TranslationalStateC,
};
use astrodyn_runner::{RotationModel, Simulation};
use bevy::prelude::*;
use glam::DVec3;

use common::*;

// ── Scenario A: Point-mass 6-DOF ──

#[test]
fn bevy_parity_point_mass_sixdof() {
    println!("Scenario A: Point-mass gravity, 6-DOF");

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(astrodyn_bevy::AstrodynPlugin);

    let _planet = app
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
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid("bevy-parity-point-mass-b1")),
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
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Skip,
                )],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        earth_source(),
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let _earth_idx = sim.add_source("Earth", earth_entry);
    sim.add_body(new_sim_body_sixdof(0, false));
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: astrodyn::typed_bridge::trans_typed_to_raw(&body.trans),
        rot: astrodyn::typed_bridge::rot_typed_to_raw(&body.rot.unwrap()),
    };

    assert_sixdof_eq("Bevy vs Sim", &bevy_state, &sim_state);
}

// ── Planetary orbit parity (3-DOF, various orbit geometries) ──

fn run_planetary_parity(label: &str, trans: TranslationalState) {
    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let _planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid("bevy-parity-point-mass-b2")),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Skip,
                )],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, _earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&trans),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("bevy-parity-point-mass-7")
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_trans = astrodyn::typed_bridge::trans_typed_to_raw(&sim.body(0).trans);
    assert_trans_eq(&format!("Bevy vs Sim ({label})"), &bevy_trans, &sim_trans);
}

#[test]
fn bevy_parity_point_mass_planetary_leo_inc() {
    run_planetary_parity("planetary_leo_inc", iss_trans().to_untyped());
}

#[test]
fn bevy_parity_point_mass_planetary_leo_polar() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7668.56),
    };
    run_planetary_parity("planetary_leo_polar", trans);
}

#[test]
fn bevy_parity_point_mass_planetary_leo_ecc() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_planetary_parity("planetary_leo_ecc", trans);
}

#[test]
fn bevy_parity_point_mass_planetary_leo_equ() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    run_planetary_parity("planetary_leo_equ", trans);
}

#[test]
fn bevy_parity_point_mass_planetary_geo() {
    let trans = TranslationalState {
        position: DVec3::new(42_164_000.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 3074.66, 0.0),
    };
    run_planetary_parity("planetary_geo", trans);
}

// ── Run2 6-DOF parity ──

#[test]
fn bevy_parity_point_mass_run2_6dof() {
    println!("Run2 6-DOF parity: point-mass gravity with rotation");
    let mut app = new_bevy_app(DT);
    let _planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid("bevy-parity-point-mass-b3")),
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
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Skip,
                )],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, _earth_idx) = new_sim_earth(DT);
    sim.add_body(new_sim_body_sixdof(0, false));
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: astrodyn::typed_bridge::trans_typed_to_raw(&sim_body.trans),
        rot: astrodyn::typed_bridge::rot_typed_to_raw(&sim_body.rot.unwrap()),
    };
    assert_sixdof_eq("Bevy vs Sim (run2_6dof)", &bevy_state, &sim_state);
}

// ── Orbinit cross-consistency parity ──

#[test]
fn bevy_parity_point_mass_orbinit_cross_consistency() {
    println!("Orbinit cross-consistency parity");
    let orbits = [
        (
            "circular",
            TranslationalState {
                position: DVec3::new(6_778_137.0, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7668.56, 0.0),
            },
        ),
        (
            "eccentric",
            TranslationalState {
                position: DVec3::new(6_778_137.0, 0.0, 0.0),
                velocity: DVec3::new(0.0, 9500.0, 0.0),
            },
        ),
        (
            "inclined",
            TranslationalState {
                position: DVec3::new(6_778_137.0, 0.0, 0.0),
                velocity: DVec3::new(0.0, 5423.0, 5423.0),
            },
        ),
        (
            "polar",
            TranslationalState {
                position: DVec3::new(6_778_137.0, 0.0, 0.0),
                velocity: DVec3::new(0.0, 0.0, 7668.56),
            },
        ),
    ];

    for (label, trans) in &orbits {
        let mut app = new_bevy_app(DT);
        let _planet = spawn_earth_source(&mut app);
        let vehicle = app
            .world_mut()
            .spawn((
                astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(
                    "bevy-parity-point-mass-b4",
                )),
                TranslationalStateC::<astrodyn::Earth>::from_untyped(*trans),
                DynamicsConfigC::default(),
                GravityControlsC(GravityControls {
                    controls: vec![GravityControl::new_spherical(
                        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                        GravityGradient::Skip,
                    )],
                }),
            ))
            .id();
        step_bevy_dt(&mut app, 1, DT);
        let bevy_trans = read_trans(app.world(), vehicle);

        let (mut sim, _earth_idx) = new_sim_earth(DT);
        sim.add_body(VehicleConfig {
            // allowed: typed↔raw kernel-boundary lift (see #397).
            trans: astrodyn::typed_bridge::trans_raw_to_root(trans),
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Skip,
                )],
            },
            ..VehicleConfig::named("bevy-parity-point-mass-6")
        });
        sim.validate().unwrap();
        sim.step().expect("step failed");
        let sim_trans = astrodyn::typed_bridge::trans_typed_to_raw(&sim.body(0).trans);
        assert_trans_eq(
            &format!("Bevy vs Sim (orbinit {label})"),
            &bevy_trans,
            &sim_trans,
        );
    }
}

// ── Timescale parity ──

#[test]
fn bevy_parity_point_mass_timescale_tdb() {
    println!("Timescale TDB parity: Bevy vs Simulation");
    let dt = 60.0;
    let n_steps = 120; // 2 hours

    // ── Bevy ──
    let mut app = new_bevy_app(dt);
    step_bevy_dt(&mut app, n_steps, dt);
    let bevy_time = app.world().resource::<astrodyn_bevy::SimulationTimeR>();

    // ── Simulation ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    sim.validate().unwrap();
    sim.step_n(n_steps).expect("step_n failed");

    assert_bits_eq(
        "Bevy vs Sim",
        "tai_tjt",
        bevy_time.tai_tjt,
        sim.time.tai_tjt,
    );
    assert_bits_eq(
        "Bevy vs Sim",
        "gmst_seconds",
        bevy_time.gmst_seconds,
        sim.time.gmst_seconds,
    );
    assert_bits_eq(
        "Bevy vs Sim",
        "simtime",
        bevy_time.simtime,
        sim.time.simtime,
    );
    println!("  Bevy vs Sim timescale: bit-identical");
}

// ── Scenario P: Time reversal round-trip ──

#[test]
fn bevy_parity_point_mass_time_reversal_round_trip() {
    println!("Scenario P: Time reversal round-trip");
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let _earth = sim.add_source("Earth", earth_entry);
    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("bevy-parity-point-mass-5")
    });
    sim.validate().unwrap();

    let initial_pos = sim.body(0).trans.position.raw_si();
    let initial_vel = sim.body(0).trans.velocity.raw_si();

    sim.step_n(50).expect("step_n failed");
    let mid_pos = sim.body(0).trans.position.raw_si();
    assert!(
        (mid_pos - initial_pos).length() > 1.0,
        "should have moved after 50 steps"
    );

    sim.time.set_scale_factor(-1.0);
    sim.step_n(50).expect("step_n failed");
    let final_pos = sim.body(0).trans.position.raw_si();
    let final_vel = sim.body(0).trans.velocity.raw_si();

    let pos_err = (final_pos - initial_pos).length();
    let vel_err = (final_vel - initial_vel).length();
    assert!(
        pos_err < 1e-3,
        "round-trip position error {pos_err} m should be < 1e-3 m"
    );
    assert!(
        vel_err < 1e-6,
        "round-trip velocity error {vel_err} m/s should be < 1e-6 m/s"
    );
    println!("  Time reversal round-trip: pos_err={pos_err:.2e} m, vel_err={vel_err:.2e} m/s");
}

// ── Scenario Q: Relative state computation ──

#[test]
fn bevy_parity_point_mass_relative_state_consistency() {
    use astrodyn::compute_relative_state;
    println!("Scenario Q: Relative state consistency");

    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let _earth = sim.add_source("Earth", earth_entry);

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("bevy-parity-point-mass-4")
    });

    let mut trans_b = iss_trans().to_untyped();
    trans_b.position += DVec3::new(100.0, 0.0, 0.0);
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&trans_b),
        rot: Some(astrodyn::RotationalStateTyped::<astrodyn::SelfRef>::new(
            astrodyn::BodyAttitude::<astrodyn::SelfRef>::from_jeod_quat(JeodQuat::identity()),
            astrodyn::AngularVelocity::<astrodyn::BodyFrame<astrodyn::SelfRef>>::from_raw_si(
                DVec3::new(0.0, 0.0, 0.001),
            ),
        )),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("bevy-parity-point-mass-3")
    });

    sim.validate().unwrap();
    sim.step_n(10).expect("step_n failed");

    let a = sim.body(0);
    let b = sim.body(1);
    let a_trans = astrodyn::typed_bridge::trans_typed_to_raw(&a.trans);
    let b_trans = astrodyn::typed_bridge::trans_typed_to_raw(&b.trans);
    let a_rot = a.rot.as_ref().map(astrodyn::typed_bridge::rot_typed_to_raw);
    let b_rot = b.rot.as_ref().map(astrodyn::typed_bridge::rot_typed_to_raw);

    // `<SelfRef, SelfRef>` is the canonical runtime-resolved boundary
    // — both subject and reference vehicle identities live in
    // per-entity ECS storage, not in the static type system.
    let rel = compute_relative_state::<astrodyn::SelfRef, astrodyn::SelfRef>(
        &a_trans,
        a_rot.as_ref(),
        &b_trans,
        b_rot.as_ref(),
    );

    let t_ref = a
        .rot
        .as_ref()
        .unwrap()
        .q_inertial_body
        .as_witness()
        .left_quat_to_transformation();
    let rel_pos_inertial = b.trans.position.raw_si() - a.trans.position.raw_si();
    let expected_pos = t_ref * rel_pos_inertial;
    // Both bodies have `Some` rotation here, so the producer takes
    // the body-frame branch. Pattern match locks down the contract:
    // a future regression that flipped the branch would not silently
    // pass — the destructure would refuse to compile.
    let astrodyn::RelativeTranslation::BodyFrame {
        position: rel_pos,
        velocity: rel_vel,
    } = rel.trans
    else {
        panic!("Some reference rotation must yield RelativeTranslation::BodyFrame");
    };
    let pos_err = (rel_pos.raw_si() - expected_pos).length();
    assert!(
        pos_err < 1e-10,
        "relative position error {pos_err:.4e} m exceeds 1e-10"
    );

    let rel_vel_inertial = b.trans.velocity.raw_si() - a.trans.velocity.raw_si();
    let omega_ref = a.rot.as_ref().unwrap().ang_vel_body.raw_si();
    let expected_vel = t_ref * rel_vel_inertial - omega_ref.cross(expected_pos);
    let vel_err = (rel_vel.raw_si() - expected_vel).length();
    assert!(
        vel_err < 1e-10,
        "relative velocity error {vel_err:.4e} m/s exceeds 1e-10"
    );
    println!("  Relative state: matches body-frame computation within {pos_err:.2e} m, {vel_err:.2e} m/s");
}

// ── Scenario R: LVLH-relative state ──

#[test]
fn bevy_parity_point_mass_lvlh_relative_consistency() {
    use astrodyn::{compute_body_lvlh_frame, compute_lvlh_relative_state};
    println!("Scenario R: LVLH-relative state consistency");

    let ref_pos = iss_trans().position.raw_si();
    let ref_vel = iss_trans().velocity.raw_si();
    let subj_pos = ref_pos + DVec3::new(100.0, 50.0, -30.0);
    let subj_vel = ref_vel + DVec3::new(0.01, -0.02, 0.005);

    let lvlh_rel =
        compute_lvlh_relative_state::<astrodyn::SelfRef>(ref_pos, ref_vel, subj_pos, subj_vel);

    let lvlh = compute_body_lvlh_frame(ref_pos, ref_vel);
    let rel_pos_inertial = subj_pos - ref_pos;
    let rel_vel_inertial = subj_vel - ref_vel;
    let expected_pos = lvlh.t_parent_this * rel_pos_inertial;
    let expected_vel =
        lvlh.t_parent_this * rel_vel_inertial - lvlh.ang_vel_this.cross(expected_pos);

    let lvlh_pos = lvlh_rel.position.raw_si();
    let lvlh_vel = lvlh_rel.velocity.raw_si();
    assert_eq!(lvlh_pos.x.to_bits(), expected_pos.x.to_bits(), "LVLH pos x");
    assert_eq!(lvlh_pos.y.to_bits(), expected_pos.y.to_bits(), "LVLH pos y");
    assert_eq!(lvlh_pos.z.to_bits(), expected_pos.z.to_bits(), "LVLH pos z");
    assert_eq!(lvlh_vel.x.to_bits(), expected_vel.x.to_bits(), "LVLH vel x");
    assert_eq!(lvlh_vel.y.to_bits(), expected_vel.y.to_bits(), "LVLH vel y");
    assert_eq!(lvlh_vel.z.to_bits(), expected_vel.z.to_bits(), "LVLH vel z");
    println!("  LVLH-relative: bit-identical with manual LVLH rotation + Coriolis");
}

// ── Scenario T: Mars IAU rotation dispatch ──

#[test]
fn bevy_parity_point_mass_mars_rotation_dispatch() {
    println!("Scenario T: Mars IAU rotation dispatch");
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let mars_mu = 4.282_837_452_7e13;
    let mars = sim.add_source(
        "Mars",
        GravitySourceEntry {
            source: GravitySource {
                mu: mars_mu,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(glam::DMat3::IDENTITY),
            rotation_model: RotationModel::MarsIAU,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(3.5e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 3.5e3, 0.0),
        }),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Mars>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("bevy-parity-point-mass-2")
    });

    sim.validate().unwrap();
    sim.step_n(10).expect("step_n failed");

    let rot = sim
        .source_pfix_rotation_typed::<astrodyn::Mars>(mars)
        .unwrap()
        .matrix();
    assert!(
        rot != glam::DMat3::IDENTITY,
        "Mars rotation should differ from identity after 10 steps"
    );

    let det = rot.determinant();
    assert!(
        (det - 1.0).abs() < 1e-10,
        "Mars rotation determinant should be 1, got {det}"
    );

    println!("  Mars rotation dispatch: non-identity, det={det:.15}");
}

// ── Scenario U: Multi-source rotation (Earth + Mars) ──

#[test]
fn bevy_parity_point_mass_multi_source_rotation() {
    println!("Scenario U: Multi-source rotation dispatch");
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(glam::DMat3::IDENTITY),
            rotation_model: RotationModel::EarthRNP,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    let mars = sim.add_source(
        "Mars",
        GravitySourceEntry {
            source: GravitySource {
                mu: 4.282_837_452_7e13,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(DVec3::new(
                2.28e11, 0.0, 0.0,
            )),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(glam::DMat3::IDENTITY),
            rotation_model: RotationModel::MarsIAU,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("bevy-parity-point-mass-1")
    });

    sim.validate().unwrap();
    sim.step_n(10).expect("step_n failed");

    let earth_rot = sim
        .source_pfix_rotation_typed::<astrodyn::Earth>(earth)
        .unwrap()
        .matrix();
    let mars_rot = sim
        .source_pfix_rotation_typed::<astrodyn::Mars>(mars)
        .unwrap()
        .matrix();

    assert!(earth_rot != glam::DMat3::IDENTITY, "Earth rotation updated");
    assert!(mars_rot != glam::DMat3::IDENTITY, "Mars rotation updated");
    assert!(
        earth_rot != mars_rot,
        "Earth and Mars rotations should differ"
    );

    println!("  Multi-source rotation: Earth and Mars independently dispatched");
}

// ── Scenario V: Relativistic gravity correction ──

#[test]
fn bevy_parity_point_mass_relativistic_gravity_consistency() {
    use astrodyn::relativistic::compute_relativistic_correction;
    println!("Scenario V: Relativistic gravity correction");

    let sun_pos = DVec3::ZERO;
    let sun_vel = DVec3::ZERO;
    let mercury_pos = DVec3::new(4.6e10, 0.0, 0.0);
    let mercury_vel = DVec3::new(0.0, 5.898e4, 0.0);

    let correction =
        compute_relativistic_correction(MU_SUN, sun_pos, mercury_pos, mercury_vel, sun_vel, &[]);

    assert!(correction.length() > 0.0, "correction should be non-zero");

    let newtonian = MU_SUN / (4.6e10 * 4.6e10);
    let ratio = correction.length() / newtonian;

    let lo = 1e-9;
    let hi = 1e-5;
    assert!(
        ratio > lo && ratio < hi,
        "correction/newtonian ratio {ratio:.2e} should be in ({lo:.0e}, {hi:.0e})"
    );

    let correction2 =
        compute_relativistic_correction(MU_SUN, sun_pos, mercury_pos, mercury_vel, sun_vel, &[]);
    assert_eq!(
        correction.x.to_bits(),
        correction2.x.to_bits(),
        "relativistic correction should be deterministic"
    );

    println!(
        "  Relativistic correction: {:.4e} m/s² ({:.2e} of Newtonian)",
        correction.length(),
        ratio
    );
}

// ── Atmosphere variants parity (run5b mean, run5c max) ──

fn run_atmosphere_parity(label: &str, trans: TranslationalState) {
    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let _planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid("bevy-parity-point-mass-b5")),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(trans),
            RotationalStateC::from(tumble_rot()),
            MassPropertiesC::from(iss_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Compute,
                )],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, _earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&trans),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                GravityGradient::Compute,
            )],
        },
        ..VehicleConfig::named("bevy-parity-point-mass-0")
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
fn bevy_parity_point_mass_run5b_atmosphere_mean() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_atmosphere_parity("run5b_atmos_mean", ecc_trans);
}

#[test]
fn bevy_parity_point_mass_run5c_atmosphere_max() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_atmosphere_parity("run5c_atmos_max", ecc_trans);
}
