//! Tier 3: Bevy `SourceMutator` vs `astrodyn_runner::Simulation::set_source_*`
//! parity.
//!
//! `astrodyn_runner::Simulation` exposes `set_source_position`,
//! `set_source_state`, and `set_source_ephemeris` for runtime
//! gravity-source retargeting. The Bevy adapter mirrors the
//! frame-state-touching mutators via [`astrodyn_bevy::SourceMutator`].
//! This test asserts:
//!
//! 1. After mutation, the Bevy planet entity's `SourceInertialPositionC`,
//!    `SourceInertialVelocityC`, and `TranslationalStateC` carry the
//!    requested values.
//! 2. After mutation, the Bevy source's frame entity (`FrameTransC`)
//!    carries the same `(position, velocity)` as
//!    `astrodyn_runner::Simulation::frame_tree()`'s source-inertial node
//!    after the equivalent `Simulation::set_source_state` call.
//! 3. Mutating a [`CentralSourceMarker`]-tagged source panics in
//!    both adapters (mirrors astrodyn_runner's `assert_ne!(fid,
//!    root_frame_id, …)` rejection of central-body mutation).

use astrodyn::{GravitySourceEntry, EARTH, MOON};
use astrodyn_bevy::{
    AstrodynPlugin, CentralSourceMarker, FrameEntityC, FrameTransC, PlanetBundle,
    SourceInertialPositionC, SourceInertialVelocityC, SourceMutator, TranslationalStateC,
};
use astrodyn_runner::Simulation;
use bevy::prelude::*;
use glam::DVec3;

const DT: f64 = 60.0;

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    app
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
        assert_bits_eq(label, &format!("[{i}]"), a[i], b[i]);
    }
}

#[test]
fn tier3_bevy_source_mutator_set_state_matches_runner() {
    // Two Earth-orbiting reference frames with the Moon as a third-body
    // source. Mutate the Moon's state via SourceMutator (Bevy) and
    // Simulation::set_source_state (astrodyn_runner), then confirm the
    // post-mutation source position/velocity in the frame tree match
    // bit-for-bit.
    //
    // Bevy uses divergent frame-tree topology (Earth and Moon both as
    // children of a generic root), but the mutation contract is the same:
    // write the requested (pos, vel) into the source's inertial frame node.

    // ── Bevy ──
    let mut app = build_app();
    app.world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH));
    // Spawn the Moon WITHOUT `SourceInertialVelocityC` so the test
    // exercises `SourceMutator::set_source_state`'s auto-insert path:
    // `PlanetBundle::point_mass` doesn't include the velocity
    // component, and the auto-insert is the contract that prevents
    // the silent-no-op footgun. Asserting the post-mutation velocity
    // below confirms the component was inserted.
    let moon_entity = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Moon", &MOON))
        .id();
    // Force the Startup schedule to run once so register_source_frames_system fires.
    app.world_mut().run_schedule(Startup);

    let new_pos = DVec3::new(3.844e8, 0.0, 0.0);
    let new_vel = DVec3::new(0.0, 1024.0, 0.0);

    // Run a one-shot system that uses SourceMutator.
    let id = app
        .world_mut()
        .register_system(move |mut mutator: SourceMutator<astrodyn::Earth>| {
            mutator.set_source_state(moon_entity, new_pos, new_vel);
        });
    app.world_mut().run_system(id).unwrap();

    let bevy_pos_c = app
        .world()
        .get::<SourceInertialPositionC>(moon_entity)
        .unwrap()
        .0
        .raw_si();
    let bevy_vel_c = app
        .world()
        .get::<SourceInertialVelocityC>(moon_entity)
        .unwrap()
        .0
        .raw_si();
    let bevy_trans = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(moon_entity)
        .unwrap()
        .0
        .to_untyped();

    // The source's frame entity (FrameTransC) should reflect the
    // same values.
    let moon_frame_entity = app.world().get::<FrameEntityC>(moon_entity).unwrap().0;
    let frame_trans = app
        .world()
        .get::<FrameTransC>(moon_frame_entity)
        .expect("source's frame entity must carry FrameTransC");
    let bevy_node_pos = frame_trans.position;
    let bevy_node_vel = frame_trans.velocity;

    assert_dvec3_bits_eq("Bevy SourceInertialPositionC", bevy_pos_c, new_pos);
    assert_dvec3_bits_eq("Bevy SourceInertialVelocityC", bevy_vel_c, new_vel);
    assert_dvec3_bits_eq(
        "Bevy TranslationalStateC.position",
        bevy_trans.position,
        new_pos,
    );
    assert_dvec3_bits_eq(
        "Bevy TranslationalStateC.velocity",
        bevy_trans.velocity,
        new_vel,
    );
    assert_dvec3_bits_eq(
        "Bevy moon frame entity FrameTransC.position",
        bevy_node_pos,
        new_pos,
    );
    assert_dvec3_bits_eq(
        "Bevy moon frame entity FrameTransC.velocity",
        bevy_node_vel,
        new_vel,
    );

    // ── astrodyn_runner ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::third_body(&MOON, astrodyn::Position::<astrodyn::RootInertial>::zero()),
    );
    sim.set_source_state(moon_idx, new_pos, new_vel);

    let sim_pos = sim.source_position(moon_idx);
    let sim_node_vel = sim
        .frame_tree()
        .get(sim.source_frame(moon_idx))
        .state
        .trans
        .velocity;

    assert_dvec3_bits_eq("Bevy vs Sim source_position", bevy_node_pos, sim_pos);
    assert_dvec3_bits_eq("Bevy vs Sim source_velocity", bevy_node_vel, sim_node_vel);
}

#[test]
#[should_panic(expected = "carries CentralSourceMarker")]
fn tier3_bevy_source_mutator_central_marker_panics_on_set_position() {
    // Mission code attaches `CentralSourceMarker` to the gravity-source
    // entity it treats as the pinned origin. `SourceMutator::set_source_position`
    // must panic on that entity, mirroring `astrodyn_runner::Simulation`'s
    // `assert_ne!(fid, root_frame_id, …)` rejection of root-source
    // mutation.
    let mut app = build_app();
    let earth = app
        .world_mut()
        .spawn((
            PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH),
            CentralSourceMarker,
        ))
        .id();
    // Run Startup so register_source_frames_system attaches `FrameEntityC`.
    // The marker guard fires *before* the frame-entity lookup, so this isn't
    // strictly required for the panic, but it keeps the test exercising
    // the same shape as a real mission setup.
    app.world_mut().run_schedule(Startup);

    let id = app
        .world_mut()
        .register_system(move |mut mutator: SourceMutator<astrodyn::Earth>| {
            mutator.set_source_position(earth, DVec3::new(1.0, 2.0, 3.0));
        });
    let _ = app.world_mut().run_system(id);
}

#[test]
#[should_panic(expected = "carries CentralSourceMarker")]
fn tier3_bevy_source_mutator_central_marker_panics_on_set_state() {
    // Same as the position case above, but for `set_source_state`.
    // Both setters must reject central-body mutation.
    let mut app = build_app();
    let earth = app
        .world_mut()
        .spawn((
            PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH),
            CentralSourceMarker,
        ))
        .id();
    app.world_mut().run_schedule(Startup);

    let id = app
        .world_mut()
        .register_system(move |mut mutator: SourceMutator<astrodyn::Earth>| {
            mutator.set_source_state(earth, DVec3::new(1.0, 2.0, 3.0), DVec3::ZERO);
        });
    let _ = app.world_mut().run_system(id);
}
