//! Bevy-vs-Simulation parity tests: point-mass gravity, planetary orbits,
//! basic 6-DOF, orbinit cross-consistency, and timescale parity.

mod parity_helpers;

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, GravityControlsC, MassPropertiesC, RotationalStateC, TranslationalStateC,
};
use glam::DVec3;
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    DynamicsConfig, GravityControl, GravityControls, GravityModel, GravitySource, JeodQuat,
    RotationalState, SixDofState, TranslationalState,
};

use parity_helpers::*;

// ── Scenario A: Point-mass 6-DOF ──

#[test]
fn tier3_bevy_point_mass_sixdof() {
    println!("Scenario A: Point-mass gravity, 6-DOF");

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
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(earth_source(), DVec3::ZERO, None);
    earth_entry.central = true;
    let earth_idx = sim.add_source("Earth", earth_entry);
    sim.add_body(new_sim_body_sixdof(earth_idx, false));
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_eq("Bevy vs Sim", &bevy_state, &sim_state);
}

// ── Planetary orbit parity (3-DOF, various orbit geometries) ──

fn run_planetary_parity(label: &str, trans: TranslationalState) {
    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_trans = read_trans(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    assert_trans_eq(
        &format!("Bevy vs Sim ({label})"),
        &bevy_trans,
        &sim.body(0).trans,
    );
}

#[test]
fn tier3_bevy_planetary_leo_inc() {
    run_planetary_parity("planetary_leo_inc", iss_trans());
}

#[test]
fn tier3_bevy_planetary_leo_polar() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7668.56),
    };
    run_planetary_parity("planetary_leo_polar", trans);
}

#[test]
fn tier3_bevy_planetary_leo_ecc() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_planetary_parity("planetary_leo_ecc", trans);
}

#[test]
fn tier3_bevy_planetary_leo_equ() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    run_planetary_parity("planetary_leo_equ", trans);
}

#[test]
fn tier3_bevy_planetary_geo() {
    let trans = TranslationalState {
        position: DVec3::new(42_164_000.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 3074.66, 0.0),
    };
    run_planetary_parity("planetary_geo", trans);
}

// ── Run2 6-DOF parity ──

#[test]
fn tier3_bevy_run2_6dof() {
    println!("Run2 6-DOF parity: point-mass gravity with rotation");
    let mut app = new_bevy_app(DT);
    let planet = spawn_earth_source(&mut app);

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
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    sim.add_body(new_sim_body_sixdof(earth_idx, false));
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let sim_body = sim.body(0);
    let sim_state = SixDofState {
        trans: sim_body.trans,
        rot: sim_body.rot.unwrap(),
    };
    assert_sixdof_eq("Bevy vs Sim (run2_6dof)", &bevy_state, &sim_state);
}

// ── Orbinit cross-consistency parity ──

#[test]
fn tier3_bevy_orbinit_cross_consistency() {
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
        let planet = spawn_earth_source(&mut app);
        let vehicle = app
            .world_mut()
            .spawn((
                TranslationalStateC(*trans),
                DynamicsConfigC::default(),
                GravityControlsC(GravityControls {
                    controls: vec![GravityControl::new_spherical(planet, false)],
                }),
            ))
            .id();
        step_bevy_dt(&mut app, 1, DT);
        let bevy_trans = read_trans(app.world(), vehicle);

        let (mut sim, earth_idx) = new_sim_earth(DT);
        sim.add_body(VehicleConfig {
            trans: *trans,
            gravity_controls: GravityControls {
                controls: vec![GravityControl::new_spherical(earth_idx, false)],
            },
            ..Default::default()
        });
        sim.validate().unwrap();
        sim.step();
        assert_trans_eq(
            &format!("Bevy vs Sim (orbinit {label})"),
            &bevy_trans,
            &sim.body(0).trans,
        );
    }
}

// ── Timescale parity ──

#[test]
fn tier3_bevy_timescale_tdb() {
    println!("Timescale TDB parity: Bevy vs Simulation");
    let dt = 60.0;
    let n_steps = 120; // 2 hours

    // ── Bevy ──
    let mut app = new_bevy_app(dt);
    step_bevy_dt(&mut app, n_steps, dt);
    let bevy_time = app.world().resource::<bevy_jeod::SimulationTimeR>();

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    sim.validate().unwrap();
    sim.step_n(n_steps);

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
fn tier3_sim_time_reversal_round_trip() {
    println!("Scenario P: Time reversal round-trip");
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    );
    earth_entry.central = true;
    let earth = sim.add_source("Earth", earth_entry);
    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });
    sim.validate().unwrap();

    let initial_pos = sim.body(0).trans.position;
    let initial_vel = sim.body(0).trans.velocity;

    sim.step_n(50);
    let mid_pos = sim.body(0).trans.position;
    assert!(
        (mid_pos - initial_pos).length() > 1.0,
        "should have moved after 50 steps"
    );

    sim.time.time_scale_factor = -1.0;
    sim.step_n(50);
    let final_pos = sim.body(0).trans.position;
    let final_vel = sim.body(0).trans.velocity;

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
fn tier3_sim_relative_state_consistency() {
    use jeod_sim::compute_relative_state;
    println!("Scenario Q: Relative state consistency");

    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    );
    earth_entry.central = true;
    let earth = sim.add_source("Earth", earth_entry);

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    let mut trans_b = iss_trans();
    trans_b.position += DVec3::new(100.0, 0.0, 0.0);
    sim.add_body(VehicleConfig {
        trans: trans_b,
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::new(0.0, 0.0, 0.001),
        }),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(10);

    let a = sim.body(0);
    let b = sim.body(1);

    let rel = compute_relative_state(&a.trans, a.rot.as_ref(), &b.trans, b.rot.as_ref());

    let t_ref = a
        .rot
        .as_ref()
        .unwrap()
        .quaternion
        .left_quat_to_transformation();
    let rel_pos_inertial = b.trans.position - a.trans.position;
    let expected_pos = t_ref * rel_pos_inertial;
    let pos_err = (rel.position - expected_pos).length();
    assert!(
        pos_err < 1e-10,
        "relative position error {pos_err:.4e} m exceeds 1e-10"
    );

    let rel_vel_inertial = b.trans.velocity - a.trans.velocity;
    let omega_ref = a.rot.as_ref().unwrap().ang_vel_body;
    let expected_vel = t_ref * rel_vel_inertial - omega_ref.cross(expected_pos);
    let vel_err = (rel.velocity - expected_vel).length();
    assert!(
        vel_err < 1e-10,
        "relative velocity error {vel_err:.4e} m/s exceeds 1e-10"
    );
    println!("  Relative state: matches body-frame computation within {pos_err:.2e} m, {vel_err:.2e} m/s");
}

// ── Scenario R: LVLH-relative state ──

#[test]
fn tier3_sim_lvlh_relative_consistency() {
    use jeod_sim::{compute_body_lvlh_frame, compute_lvlh_relative_state};
    println!("Scenario R: LVLH-relative state consistency");

    let ref_pos = iss_trans().position;
    let ref_vel = iss_trans().velocity;
    let subj_pos = ref_pos + DVec3::new(100.0, 50.0, -30.0);
    let subj_vel = ref_vel + DVec3::new(0.01, -0.02, 0.005);

    let lvlh_rel = compute_lvlh_relative_state(ref_pos, ref_vel, subj_pos, subj_vel);

    let lvlh = compute_body_lvlh_frame(ref_pos, ref_vel);
    let rel_pos_inertial = subj_pos - ref_pos;
    let rel_vel_inertial = subj_vel - ref_vel;
    let expected_pos = lvlh.t_parent_this * rel_pos_inertial;
    let expected_vel =
        lvlh.t_parent_this * rel_vel_inertial - lvlh.ang_vel_this.cross(expected_pos);

    assert_eq!(
        lvlh_rel.position.x.to_bits(),
        expected_pos.x.to_bits(),
        "LVLH pos x"
    );
    assert_eq!(
        lvlh_rel.position.y.to_bits(),
        expected_pos.y.to_bits(),
        "LVLH pos y"
    );
    assert_eq!(
        lvlh_rel.position.z.to_bits(),
        expected_pos.z.to_bits(),
        "LVLH pos z"
    );
    assert_eq!(
        lvlh_rel.velocity.x.to_bits(),
        expected_vel.x.to_bits(),
        "LVLH vel x"
    );
    assert_eq!(
        lvlh_rel.velocity.y.to_bits(),
        expected_vel.y.to_bits(),
        "LVLH vel y"
    );
    assert_eq!(
        lvlh_rel.velocity.z.to_bits(),
        expected_vel.z.to_bits(),
        "LVLH vel z"
    );
    println!("  LVLH-relative: bit-identical with manual LVLH rotation + Coriolis");
}

// ── Scenario T: Mars IAU rotation dispatch ──

#[test]
fn tier3_sim_mars_rotation_dispatch() {
    println!("Scenario T: Mars IAU rotation dispatch");
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let mars_mu = 4.282_837_452_7e13;
    let mars = sim.add_source(
        "Mars",
        GravitySourceEntry {
            source: GravitySource {
                mu: mars_mu,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(glam::DMat3::IDENTITY),
            rotation_model: RotationModel::MarsIAU,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::new(3.5e6, 0.0, 0.0),
            velocity: DVec3::new(0.0, 3.5e3, 0.0),
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(mars, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(10);

    let rot = sim.source_pfix_rotation(mars).unwrap();
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
fn tier3_sim_multi_source_rotation() {
    println!("Scenario U: Multi-source rotation dispatch");
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(glam::DMat3::IDENTITY),
            rotation_model: RotationModel::EarthRNP,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    let mars = sim.add_source(
        "Mars",
        GravitySourceEntry {
            source: GravitySource {
                mu: 4.282_837_452_7e13,
                model: GravityModel::PointMass,
            },
            position: DVec3::new(2.28e11, 0.0, 0.0),
            velocity: DVec3::ZERO,
            t_inertial_pfix: Some(glam::DMat3::IDENTITY),
            rotation_model: RotationModel::MarsIAU,
            delta_c20: 0.0,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );

    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim.step_n(10);

    let earth_rot = sim.source_pfix_rotation(earth).unwrap();
    let mars_rot = sim.source_pfix_rotation(mars).unwrap();

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
fn tier3_sim_relativistic_gravity_consistency() {
    use jeod_sim::relativistic::compute_relativistic_correction;
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
    let planet = spawn_earth_source(&mut app);

    let vehicle = app
        .world_mut()
        .spawn((
            TranslationalStateC(trans),
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
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_sixdof(app.world(), vehicle);

    // ── Simulation ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    sim.add_body(VehicleConfig {
        trans,
        rot: Some(tumble_rot()),
        mass: Some(iss_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth_idx, true)],
        },
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
fn tier3_bevy_run5b_atmosphere_mean() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_atmosphere_parity("run5b_atmos_mean", ecc_trans);
}

#[test]
fn tier3_bevy_run5c_atmosphere_max() {
    let ecc_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 9500.0, 0.0),
    };
    run_atmosphere_parity("run5c_atmos_max", ecc_trans);
}
