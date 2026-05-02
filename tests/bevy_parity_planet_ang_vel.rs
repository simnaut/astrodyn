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
    EphemerisR, FrameTreeR, GravitySourceC, JeodPlugin, PlanetAngularVelocityC, PlanetBundle,
    PlanetFixedRotationC, PlanetOmegaC, RotationModelC, SourceInertialPositionC,
    SourcePfixFrameIdC,
};
use glam::DVec3;
use jeod_runner::{RotationModel, Simulation};
use jeod_sim::{
    Ephemeris, GravityModel, GravitySource, GravitySourceEntry, PlanetConfig, EARTH, MARS, MOON,
};
use jeod_test_data::tier3_csv::test_data_path;

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

/// Read the FrameTreeR pfix node's `ang_vel_this` for a Bevy planet
/// entity. Asserts the pfix-node-of-truth is in sync with the ECS
/// `PlanetAngularVelocityC` component (PR #260 review fixup).
fn bevy_pfix_node_ang_vel(app: &App, planet: Entity) -> DVec3 {
    let pfix_fid = app
        .world()
        .get::<SourcePfixFrameIdC>(planet)
        .expect("planet entity is missing SourcePfixFrameIdC")
        .0;
    let frame_tree = app.world().resource::<FrameTreeR>();
    frame_tree.0.get(pfix_fid).state.rot.ang_vel_this
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

    // FrameTreeR pfix node must match too — frame-tree consumers rely
    // on this (compute_relative_state through pfix). PR #260 review.
    let bevy_node_ang_vel = bevy_pfix_node_ang_vel(&app, planet);
    assert_dvec3_bits_eq(
        "Bevy FrameTreeR Earth pfix-node ang_vel vs Sim",
        bevy_node_ang_vel,
        sim_ang_vel,
    );
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

    let bevy_node_ang_vel = bevy_pfix_node_ang_vel(&app, planet);
    assert_dvec3_bits_eq(
        "Bevy FrameTreeR Mars pfix-node ang_vel vs Sim",
        bevy_node_ang_vel,
        sim_ang_vel,
    );
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

    let bevy_node_ang_vel = bevy_pfix_node_ang_vel(&app, planet);
    assert_dvec3_bits_eq(
        "Bevy FrameTreeR Moon pfix-node ang_vel vs Sim",
        bevy_node_ang_vel,
        sim_ang_vel,
    );
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

/// Toggling a source's `RotationModelC` from a rotating model to
/// `None` at runtime must remove the `SourcePfixFrameIdC` component,
/// not just clear the pfix node to identity — otherwise consumers
/// that branch on the *presence* of the component would still treat
/// the source as rotating-capable. Mirrors the registration symmetry:
/// `register_pfix_frames_system` inserts the component when a source
/// gains a non-`None` rotation model after registration; this test
/// verifies the inverse.
///
/// The orphan pfix node must also be *reused* on the next toggle back
/// to a rotating model. Without reuse, every cycle would (a) leak a
/// `<name>.pfix` node into the frame tree (which has no removal API)
/// and (b) let `FrameTree::find_by_name` shadow the live frame with a
/// stale orphan. Run several toggle cycles and assert that the frame
/// tree's node count is stable and that `find_by_name` always
/// resolves to the live `SourcePfixFrameIdC`.
#[test]
fn tier3_bevy_rotation_none_toggle_removes_pfix_component() {
    let (mut app, planet) = build_planet_app("Earth", &EARTH);
    // Step once with the default `EarthRNP` model so registration runs
    // and `SourcePfixFrameIdC` is inserted.
    step_bevy_once(&mut app);
    assert!(
        app.world().get::<SourcePfixFrameIdC>(planet).is_some(),
        "EarthRNP source must carry SourcePfixFrameIdC after registration"
    );

    // Capture the post-registration tree size and the original pfix
    // FrameId — both should remain stable across the toggle cycles.
    let initial_tree_len = app.world().resource::<FrameTreeR>().0.len();
    let original_pfix_id = app.world().get::<SourcePfixFrameIdC>(planet).unwrap().0;

    // Toggle to `RotationModel::None` and step again. The clear branch
    // in `planet_fixed_rotation_system` should remove the component so
    // the source matches the "no pfix node" case from registration.
    app.world_mut()
        .entity_mut(planet)
        .insert(RotationModelC(RotationModel::None));
    step_bevy_once(&mut app);
    assert!(
        app.world().get::<SourcePfixFrameIdC>(planet).is_none(),
        "toggling RotationModel to None must remove SourcePfixFrameIdC; \
         leaving it in place reintroduces the Some(identity) vs None \
         ambiguity that round-9 registration fixed"
    );
    // The orphan node must be renamed off `Earth.pfix` so a
    // `find_by_name` lookup of the canonical name returns nothing
    // (no live frame exists at this point) and won't shadow a future
    // live frame after toggling back.
    {
        let frame_tree = &app.world().resource::<FrameTreeR>().0;
        assert!(
            frame_tree.find_by_name("Earth.pfix").is_none(),
            "after toggle to None, no frame should answer to the canonical \
             `Earth.pfix` name — the orphan must be renamed to a sentinel \
             so future `find_by_name` lookups don't shadow a future live frame"
        );
        assert_eq!(
            frame_tree.len(),
            initial_tree_len,
            "the orphan node must be retained (renamed in place), not \
             allocated nor leaked alongside a freshly-allocated one"
        );
    }

    // Toggling back to `EarthRNP` must reinstate the component on the
    // next registration pass AND reuse the same FrameId — proving the
    // reuse path works (no fresh allocation).
    app.world_mut()
        .entity_mut(planet)
        .insert(RotationModelC(RotationModel::EarthRNP));
    step_bevy_once(&mut app);
    let reinstated = app
        .world()
        .get::<SourcePfixFrameIdC>(planet)
        .expect(
            "toggling back to EarthRNP must reinstate SourcePfixFrameIdC \
             via register_pfix_frames_system's reuse path",
        )
        .0;
    assert_eq!(
        reinstated, original_pfix_id,
        "reuse path: the reinstated FrameId must equal the original \
         orphan's FrameId — no fresh allocation"
    );
    assert_eq!(
        app.world().resource::<FrameTreeR>().0.len(),
        initial_tree_len,
        "tree size must stay constant across a None→rotating toggle — \
         the reuse path replaces, not appends"
    );
    // Canonical name now resolves to the live frame again.
    assert_eq!(
        app.world()
            .resource::<FrameTreeR>()
            .0
            .find_by_name("Earth.pfix"),
        Some(original_pfix_id),
        "after toggle back to rotating, find_by_name must resolve \
         `Earth.pfix` to the reused live frame"
    );

    // Run a few more toggle cycles to confirm the tree size stays
    // bounded and the FrameId stays stable. Without the reuse path,
    // each cycle would push the tree size up by 1.
    for _ in 0..5 {
        app.world_mut()
            .entity_mut(planet)
            .insert(RotationModelC(RotationModel::None));
        step_bevy_once(&mut app);
        app.world_mut()
            .entity_mut(planet)
            .insert(RotationModelC(RotationModel::EarthRNP));
        step_bevy_once(&mut app);
    }
    assert_eq!(
        app.world().resource::<FrameTreeR>().0.len(),
        initial_tree_len,
        "frame-tree size must not grow with toggle cycle count"
    );
    assert_eq!(
        app.world().get::<SourcePfixFrameIdC>(planet).unwrap().0,
        original_pfix_id,
        "the same pfix FrameId must be reused across every toggle cycle"
    );
}

#[test]
fn tier3_bevy_planet_ang_vel_moon_de421() {
    // MoonDE421 is the only RotationModel branch whose Bevy ↔ runner
    // end-to-end path was previously uncovered (issue #265). Mirror the
    // MoonIAU test, additionally inserting `EphemerisR` with the BPC
    // kernel that `planet_fixed_rotation_system`'s MoonDE421 branch
    // requires. The kernel `moon_pa_de421_1900-2050.bpc` is already
    // committed to `test_data/` (used by `tier3_simulation_earth_moon_clem`),
    // so no new fixture commit is needed.
    let bsp = test_data_path("de421.bsp");
    let bpc = test_data_path("moon_pa_de421_1900-2050.bpc");

    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
    let mut eph = Ephemeris::from_bsp(&bsp).expect("DE421 BSP load");
    eph.load_bpc(&bpc).expect("Moon DE421 BPC load");
    app.insert_resource(EphemerisR(eph));
    let planet = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Moon", &MOON))
        .id();
    // Override the bundle's default `RotationModelC(MoonIAU)` (copied
    // from `MOON.rotation_model`) with `MoonDE421`. Inserting after
    // spawn replaces the bundle's component cleanly.
    app.world_mut()
        .entity_mut(planet)
        .insert(RotationModelC(RotationModel::MoonDE421));
    step_bevy_once(&mut app);

    let bevy_ang_vel = app
        .world()
        .get::<PlanetAngularVelocityC>(planet)
        .unwrap()
        .0
        .raw_si();
    let expected = DVec3::new(0.0, 0.0, MOON.omega);
    assert_dvec3_bits_eq(
        "Bevy Moon DE421 ang_vel vs PlanetConfig",
        bevy_ang_vel,
        expected,
    );

    // ── jeod_runner ──
    // `central_body(&MOON)` copies `MOON.rotation_model` (= MoonIAU);
    // override after construction to switch the branch to MoonDE421.
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let mut entry = GravitySourceEntry::central_body(&MOON);
    entry.rotation_model = RotationModel::MoonDE421;
    sim.add_source("Moon", entry);
    let mut sim_eph = Ephemeris::from_bsp(&bsp).expect("DE421 BSP load (sim)");
    sim_eph.load_bpc(&bpc).expect("Moon DE421 BPC load (sim)");
    sim.ephemeris = Some(sim_eph);
    sim.validate().unwrap();
    sim.step().expect("step failed");

    let sim_ang_vel = sim_pfix_ang_vel(&sim, "Moon");
    assert_dvec3_bits_eq(
        "Bevy vs Sim Moon DE421 pfix ang_vel",
        bevy_ang_vel,
        sim_ang_vel,
    );

    let bevy_node_ang_vel = bevy_pfix_node_ang_vel(&app, planet);
    assert_dvec3_bits_eq(
        "Bevy FrameTreeR Moon DE421 pfix-node ang_vel vs Sim",
        bevy_node_ang_vel,
        sim_ang_vel,
    );
}
