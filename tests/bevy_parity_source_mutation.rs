//! Tier 3: Bevy `SourceMutator` vs `jeod_runner::Simulation::set_source_*`
//! parity (issue #71 item 5).
//!
//! Issue #71 catalogued that `jeod_runner::Simulation` exposed
//! `set_source_position`, `set_source_state`, and `set_source_ephemeris`
//! for runtime gravity-source retargeting; the Bevy adapter had no
//! equivalent. This test exercises the new
//! [`bevy_jeod::SourceMutator`] system parameter and asserts that:
//!
//! 1. After mutation, the Bevy planet entity's `SourceInertialPositionC`,
//!    `SourceInertialVelocityC`, and `TranslationalStateC` carry the
//!    requested values.
//! 2. After mutation, the Bevy `FrameTreeR` resource's source-inertial
//!    node carries the same `(position, velocity)` as
//!    `jeod_runner::Simulation::frame_tree()`'s source-inertial node
//!    after the equivalent `Simulation::set_source_state` call.
//! 3. Mutating a root-mapped source panics in both adapters (jeod_runner
//!    asserts central-body mutations are forbidden; Bevy currently
//!    doesn't map any source to root, so this codepath only fires in
//!    jeod_runner).
//!
//! Phase B step B11 of the issue #71 plan; closes the parity-test gap
//! flagged in the plan (no existing parity test exercised source
//! mutation).

use bevy::prelude::*;
use bevy_jeod::{
    FrameTreeR, JeodPlugin, PlanetBundle, RootFrameIdR, SourceFrameIdC, SourceInertialPositionC,
    SourceInertialVelocityC, SourceMutator, TranslationalStateC,
};
use glam::DVec3;
use jeod_runner::Simulation;
use jeod_sim::{GravitySourceEntry, EARTH, MOON};

const DT: f64 = 60.0;

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
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
    // Simulation::set_source_state (jeod_runner), then confirm the
    // post-mutation source position/velocity in the frame tree match
    // bit-for-bit.
    //
    // Bevy uses divergent frame-tree topology (Earth and Moon both as
    // children of a generic root), but the mutation contract is the same:
    // write the requested (pos, vel) into the source's inertial frame node.

    // ── Bevy ──
    let mut app = build_app();
    app.world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH));
    // Spawn the Moon WITHOUT `SourceInertialVelocityC` so the test
    // exercises `SourceMutator::set_source_state`'s auto-insert path
    // (PR #260 round-3 fixup): `PlanetBundle::point_mass` doesn't
    // include the velocity component, and the auto-insert is the
    // contract that prevents the silent-no-op footgun. Asserting the
    // post-mutation velocity below confirms the component was inserted.
    let moon_entity = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Moon", &MOON))
        .id();
    // Force the Startup schedule to run once so register_source_frames_system fires.
    app.world_mut().run_schedule(Startup);

    let new_pos = DVec3::new(3.844e8, 0.0, 0.0);
    let new_vel = DVec3::new(0.0, 1024.0, 0.0);

    // Run a one-shot system that uses SourceMutator.
    let id = app
        .world_mut()
        .register_system(move |mut mutator: SourceMutator| {
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
        .get::<TranslationalStateC>(moon_entity)
        .unwrap()
        .0
        .to_untyped();

    // Frame-tree node should reflect the same values.
    let moon_fid = app.world().get::<SourceFrameIdC>(moon_entity).unwrap().0;
    let frame_tree = app.world().resource::<FrameTreeR>();
    let node = frame_tree.0.get(moon_fid);
    let bevy_node_pos = node.state.trans.position;
    let bevy_node_vel = node.state.trans.velocity;

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
    assert_dvec3_bits_eq("Bevy FrameTreeR.moon.position", bevy_node_pos, new_pos);
    assert_dvec3_bits_eq("Bevy FrameTreeR.moon.velocity", bevy_node_vel, new_vel);

    // ── jeod_runner ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::third_body(&MOON, jeod_sim::Position::<jeod_sim::RootInertial>::zero()),
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
#[should_panic(expected = "set_source_position: cannot set position of the root")]
fn tier3_bevy_source_mutator_root_mutation_panics() {
    // The Bevy adapter never maps a source to the root frame (every
    // gravity source becomes a child of root in `register_source_frames_system`).
    // Construct the panic by directly inserting `SourceFrameIdC` pointing
    // at the root and then calling `set_source_position`. This verifies
    // the lifted helper's `assert_ne!(fid, root_frame_id, …)` guard
    // surfaces through the Bevy mutator.
    use bevy_jeod::components::SourceFrameIdC;

    let mut app = build_app();
    let root_id = app.world().resource::<RootFrameIdR>().0;
    let source = app
        .world_mut()
        .spawn((
            Name::new("PinnedToRoot"),
            bevy_jeod::components::GravitySourceC(jeod_sim::GravitySource {
                mu: EARTH.shape.mu,
                model: jeod_sim::GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            SourceFrameIdC(root_id),
        ))
        .id();

    let id = app
        .world_mut()
        .register_system(move |mut mutator: SourceMutator| {
            mutator.set_source_position(source, DVec3::new(1.0, 2.0, 3.0));
        });
    let _ = app.world_mut().run_system(id);
}
