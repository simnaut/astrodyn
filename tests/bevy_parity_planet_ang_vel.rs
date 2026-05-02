//! Tier 3: Bevy App vs jeod_runner::Simulation parity for the
//! planet-fixed frame's angular velocity (issue #71 item 1).
//!
//! Issue #71 catalogued that `jeod_runner::Simulation` writes
//! `ang_vel_this = [0, 0, planet_omega]` onto every pfix node each step
//! (matching JEOD's `planet_rnp.cc`), while the Bevy adapter only wrote
//! the rotation matrix. This test exercises the new
//! `PlanetAngularVelocityC` component populated by
//! `planet_fixed_rotation_system` and asserts:
//!
//! 1. The Bevy planet entity's `PlanetAngularVelocityC` equals
//!    `[0, 0, omega_planet]` for each rotation model.
//! 2. The Bevy value is bit-identical to the corresponding pfix node's
//!    `state.rot.ang_vel_this` in `jeod_runner::Simulation::frame_tree()`.
//!
//! Phase B step B11 of the issue #71 plan; closes the parity-test gap
//! flagged in the plan (no existing parity test exercises pfix angular
//! velocity).

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    GravitySourceC, JeodPlugin, PlanetAngularVelocityC, PlanetBundle, PlanetFixedRotationC,
    PlanetOmegaC, RotationModelC, SourceInertialPositionC,
};
use glam::DVec3;
use jeod_runner::{RotationModel, Simulation};
use jeod_sim::{GravityModel, GravitySource, GravitySourceEntry, PlanetConfig, EARTH, MARS, MOON};

const DT: f64 = 60.0;

fn step_bevy_once(app: &mut App) {
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

/// Build a Bevy app with a single planet-source entity (no integrated
/// vehicle — this test only exercises the ephemeris stage).
fn build_planet_app(name: &str, config: &PlanetConfig) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
    let planet = app
        .world_mut()
        .spawn(PlanetBundle::point_mass(name, config))
        .id();
    (app, planet)
}

/// Build a `jeod_runner::Simulation` with the same planet as `central_body`
/// so we can read its pfix node back from `frame_tree()`.
fn build_sim(name: &str, config: &PlanetConfig) -> Simulation {
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let entry = GravitySourceEntry::central_body(config);
    sim.add_source(name, entry);
    sim.validate().unwrap();
    sim.step().expect("step failed");
    sim
}

fn assert_bits_eq(label: &str, component: &str, a: f64, b: f64) {
    assert!(
        a.to_bits() == b.to_bits(),
        "{label} {component} not bit-identical:\n  \
         A: {a} (bits={:#018x})\n  \
         B: {b} (bits={:#018x})",
        a.to_bits(),
        b.to_bits(),
    );
}

fn assert_dvec3_bits_eq(label: &str, a: DVec3, b: DVec3) {
    for i in 0..3 {
        assert_bits_eq(label, &format!("ang_vel[{i}]"), a[i], b[i]);
    }
}

/// Read the pfix node's `ang_vel_this` from `jeod_runner` for the
/// central source by name (the only source we registered in each test).
fn sim_pfix_ang_vel(sim: &Simulation, name: &str) -> DVec3 {
    let frame_tree = sim.frame_tree();
    let pfix_id = frame_tree
        .find_by_name(&format!("{name}.pfix"))
        .unwrap_or_else(|| panic!("no '{name}.pfix' frame node in jeod_runner frame tree"));
    frame_tree.get(pfix_id).state.rot.ang_vel_this
}

#[test]
fn tier3_bevy_planet_ang_vel_earth_rnp() {
    let (mut app, planet) = build_planet_app("Earth", &EARTH);
    step_bevy_once(&mut app);

    let bevy_ang_vel = app
        .world()
        .get::<PlanetAngularVelocityC>(planet)
        .unwrap()
        .0
        .raw_si();

    // Expected: [0, 0, omega_earth] from PlanetConfig::omega.
    let expected = DVec3::new(0.0, 0.0, EARTH.omega);
    assert_dvec3_bits_eq("Bevy Earth ang_vel vs PlanetConfig", bevy_ang_vel, expected);

    let sim = build_sim("Earth", &EARTH);
    let sim_ang_vel = sim_pfix_ang_vel(&sim, "Earth");
    assert_dvec3_bits_eq("Bevy vs Sim Earth pfix ang_vel", bevy_ang_vel, sim_ang_vel);
}

#[test]
fn tier3_bevy_planet_ang_vel_mars_iau() {
    let (mut app, planet) = build_planet_app("Mars", &MARS);
    step_bevy_once(&mut app);

    let bevy_ang_vel = app
        .world()
        .get::<PlanetAngularVelocityC>(planet)
        .unwrap()
        .0
        .raw_si();

    let expected = DVec3::new(0.0, 0.0, MARS.omega);
    assert_dvec3_bits_eq("Bevy Mars ang_vel vs PlanetConfig", bevy_ang_vel, expected);

    let sim = build_sim("Mars", &MARS);
    let sim_ang_vel = sim_pfix_ang_vel(&sim, "Mars");
    assert_dvec3_bits_eq("Bevy vs Sim Mars pfix ang_vel", bevy_ang_vel, sim_ang_vel);
}

#[test]
fn tier3_bevy_planet_ang_vel_moon_iau() {
    let (mut app, planet) = build_planet_app("Moon", &MOON);
    step_bevy_once(&mut app);

    let bevy_ang_vel = app
        .world()
        .get::<PlanetAngularVelocityC>(planet)
        .unwrap()
        .0
        .raw_si();

    let expected = DVec3::new(0.0, 0.0, MOON.omega);
    assert_dvec3_bits_eq("Bevy Moon ang_vel vs PlanetConfig", bevy_ang_vel, expected);

    let sim = build_sim("Moon", &MOON);
    let sim_ang_vel = sim_pfix_ang_vel(&sim, "Moon");
    assert_dvec3_bits_eq("Bevy vs Sim Moon pfix ang_vel", bevy_ang_vel, sim_ang_vel);
}

#[test]
fn tier3_bevy_planet_ang_vel_rotation_none_leaves_default() {
    // RotationModel::None: ang_vel must remain at default zero.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    // Spawn a manually-configured planet with RotationModel::None.
    // PlanetBundle::point_mass would copy EARTH's RotationModel; we want
    // RotationModel::None explicitly to confirm the system skips writing
    // ang_vel when the rotation model is None.
    let planet = app
        .world_mut()
        .spawn((
            Name::new("Inert"),
            GravitySourceC(GravitySource {
                mu: EARTH.shape.mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            PlanetFixedRotationC(jeod_sim::FrameTransform::from_matrix(glam::DMat3::IDENTITY)),
            RotationModelC(RotationModel::None),
            PlanetOmegaC(EARTH.omega),
            PlanetAngularVelocityC::default(),
        ))
        .id();
    step_bevy_once(&mut app);

    let bevy_ang_vel = app
        .world()
        .get::<PlanetAngularVelocityC>(planet)
        .unwrap()
        .0
        .raw_si();
    assert_dvec3_bits_eq(
        "RotationModel::None leaves ang_vel default",
        bevy_ang_vel,
        DVec3::ZERO,
    );
}
