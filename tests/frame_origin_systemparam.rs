//! Issue #278 (Frame-Tree-ECS-Native § 13 PR 2) bit-identity validation
//! for the new [`FrameOrigin`] `SystemParam`. Mirrors the load-bearing
//! correctness check that PR 1 added for [`RelativeFrameState`]
//! (`tests/frame_storage_relative_frame_state.rs`): the new
//! mission-facing surface produces numerics bit-identical to the
//! arena helpers (`jeod_sim::frame_origin` /
//! `jeod_sim::frame_origin_typed`) the dual-write keeps in sync.
//!
//! ## Before / After diff (the design-doc evidence)
//!
//! **Before** (arena read through `FrameTreeR` + `FrameId`
//! components, held in a `Res<FrameTreeR>`):
//! ```ignore
//! fn read_via_arena(
//!     frame_tree: Res<FrameTreeR>,
//!     root: Res<RootFrameIdR>,
//!     bodies: Query<&BodyFrameIdC, With<MyBody>>,
//! ) -> DVec3 {
//!     let body_fid = bodies.single().unwrap().0;
//!     jeod_sim::frame_origin(&frame_tree.0, root.0, body_fid).0
//! }
//! ```
//!
//! **After** (`FrameOrigin` SystemParam, ECS-native, typed return):
//! ```ignore
//! fn read_via_systemparam(
//!     origin: FrameOrigin,
//!     root: Res<RootFrameEntityR>,
//!     bodies: Query<&FrameEntityC, With<MyBody>>,
//! ) -> Position<RootInertial> {
//!     let body_e = bodies.single().unwrap().0;
//!     let (pos, _vel) = origin.origin_in_root(root.0, body_e);
//!     pos
//! }
//! ```
//!
//! The "After" surface never names a `FrameId`, never holds a
//! `Res<FrameTreeR>`, and reads like any other Bevy `SystemParam`.
//! It also returns `Position<RootInertial>` directly, lifting the
//! arena's raw `DVec3` into the typed sibling without a per-call
//! `from_raw_si` boundary at the consumer's site.

// `FrameTreeR` is `#[deprecated]` for mission-code use; this test
// reads the arena to assert the new SystemParam returns
// bit-identical numerics. This is dual-write infrastructure
// validation, not mission-shaped reads.
#![allow(deprecated)]

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::frame_param::{FrameOrigin, RelativeFrameState};
use bevy_jeod::{
    BodyFrameIdC, DynamicsConfigC, FrameEntityC, FrameTreeR, GravityControlsC, JeodPlugin,
    MassPropertiesC, PfixFrameEntityC, PlanetBundle, RootFrameEntityR, RootFrameIdR,
    RotationalStateC, SourcePfixFrameIdC, TranslationalStateC,
};
use glam::DVec3;
use jeod_sim::{
    DynamicsConfig, GravityControls, JeodQuat, MassProperties, PlanetConfig, Position,
    RootInertial, RotationalState, TranslationalState, Velocity, EARTH,
};

const DT: f64 = 60.0;

fn step_once(app: &mut App) {
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

fn build_app(planet_name: &str, planet: &PlanetConfig) -> (App, Entity, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let planet_e = app
        .world_mut()
        .spawn(PlanetBundle::point_mass(planet_name, planet))
        .id();

    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    let rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    };
    let body_e = app
        .world_mut()
        .spawn((
            Name::new("body"),
            TranslationalStateC::from(trans),
            RotationalStateC::from(rot),
            MassPropertiesC::from(MassProperties::new(1.0)),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls::<Entity> { controls: vec![] }),
        ))
        .id();

    (app, planet_e, body_e)
}

fn assert_bits_eq(label: &str, a: f64, b: f64) {
    assert!(
        a.to_bits() == b.to_bits(),
        "{label} not bit-identical:\n  arena: {a} (bits={:#018x})\n  ecs:   {b} (bits={:#018x})",
        a.to_bits(),
        b.to_bits(),
    );
}

fn assert_dvec3_bits_eq(label: &str, arena: DVec3, ecs: DVec3) {
    for i in 0..3 {
        assert_bits_eq(&format!("{label}[{i}]"), arena[i], ecs[i]);
    }
}

/// `FrameOrigin::origin_in_root(root, body)` returns the same
/// `(position, velocity)` as
/// `jeod_sim::frame_origin(&frame_tree.0, root_fid, body_fid)`.
/// This is the load-bearing correctness check for PR 2 — proving
/// that the new mission-facing SystemParam reads bit-identical
/// numerics to the arena helper consumers used to call.
#[test]
fn frame_origin_in_root_matches_arena() {
    let (mut app, _planet_e, body_e) = build_app("Earth", &EARTH);
    step_once(&mut app);

    // Arena answer (same path internal physics still uses).
    let (arena_pos, arena_vel) = {
        let world = app.world();
        let body_fid = world.get::<BodyFrameIdC>(body_e).unwrap().0;
        let root_fid = world.resource::<RootFrameIdR>().0;
        let frame_tree = world.resource::<FrameTreeR>();
        jeod_sim::frame_origin(&frame_tree.0, root_fid, body_fid)
    };

    // ECS answer via the typed FrameOrigin SystemParam.
    let body_frame_e = app.world().get::<FrameEntityC>(body_e).unwrap().0;
    let root_frame_e = app.world().resource::<RootFrameEntityR>().0;
    let (ecs_pos, ecs_vel) = app
        .world_mut()
        .run_system_cached_with(
            |In((root, body)): In<(Entity, Entity)>,
             origin: FrameOrigin|
             -> (Position<RootInertial>, Velocity<RootInertial>) {
                origin.origin_in_root(root, body)
            },
            (root_frame_e, body_frame_e),
        )
        .expect("run_system_cached_with should succeed");

    assert_dvec3_bits_eq("origin_in_root.pos", arena_pos, ecs_pos.raw_si());
    assert_dvec3_bits_eq("origin_in_root.vel", arena_vel, ecs_vel.raw_si());
}

/// `FrameOrigin::origin_in_root(root, root)` returns
/// `(DVec3::ZERO, DVec3::ZERO)` — the same identity short-circuit
/// `jeod_sim::frame_origin(tree, root, root)` carries. The typed
/// return wraps the zeros at the `Position<RootInertial>` /
/// `Velocity<RootInertial>` phantom.
#[test]
fn frame_origin_in_root_of_root_is_zero() {
    let (mut app, _planet_e, _body_e) = build_app("Earth", &EARTH);
    step_once(&mut app);

    let root_frame_e = app.world().resource::<RootFrameEntityR>().0;
    let (pos, vel) = app
        .world_mut()
        .run_system_cached_with(
            |In(root): In<Entity>,
             origin: FrameOrigin|
             -> (Position<RootInertial>, Velocity<RootInertial>) {
                origin.origin_in_root(root, root)
            },
            root_frame_e,
        )
        .expect("run_system_cached_with should succeed");
    assert_eq!(pos.raw_si(), DVec3::ZERO);
    assert_eq!(vel.raw_si(), DVec3::ZERO);
}

/// `FrameOrigin::origin_in(ancestor, frame)` for a non-root ancestor:
/// query the body's origin in the planet's pfix-frame coordinates.
/// The arena equivalent is
/// `compute_relative_state(pfix_fid, body_fid).trans.{position,velocity}`,
/// which is what `RelativeFrameState::position_velocity(pfix, body)`
/// returns under the hood.
#[test]
fn frame_origin_in_pfix_matches_relative_frame_state() {
    let (mut app, planet_e, body_e) = build_app("Earth", &EARTH);
    step_once(&mut app);

    let body_frame_e = app.world().get::<FrameEntityC>(body_e).unwrap().0;
    let pfix_frame_e = app.world().get::<PfixFrameEntityC>(planet_e).unwrap().0;

    // Arena answer.
    let (arena_pos, arena_vel) = {
        let world = app.world();
        let body_fid = world.get::<BodyFrameIdC>(body_e).unwrap().0;
        let pfix_fid = world.get::<SourcePfixFrameIdC>(planet_e).unwrap().0;
        let frame_tree = world.resource::<FrameTreeR>();
        let state = frame_tree.0.compute_relative_state(pfix_fid, body_fid);
        (state.trans.position, state.trans.velocity)
    };

    // ECS answer via the FrameOrigin SystemParam (raw ancestor form).
    let (origin_pos, origin_vel) = app
        .world_mut()
        .run_system_cached_with(
            |In((ancestor, frame)): In<(Entity, Entity)>, origin: FrameOrigin| -> (DVec3, DVec3) {
                origin.origin_in(ancestor, frame)
            },
            (pfix_frame_e, body_frame_e),
        )
        .expect("run_system_cached_with should succeed");

    // ECS answer via RelativeFrameState (the FrameOrigin sibling).
    // FrameOrigin::origin_in is sugar over
    // RelativeFrameState::position_velocity, so they must agree.
    let (rel_pos, rel_vel) = app
        .world_mut()
        .run_system_cached_with(
            |In((ancestor, frame)): In<(Entity, Entity)>,
             rel: RelativeFrameState|
             -> (DVec3, DVec3) { rel.position_velocity(ancestor, frame) },
            (pfix_frame_e, body_frame_e),
        )
        .expect("run_system_cached_with should succeed");

    assert_dvec3_bits_eq("FrameOrigin.pos vs arena", arena_pos, origin_pos);
    assert_dvec3_bits_eq("FrameOrigin.vel vs arena", arena_vel, origin_vel);
    assert_dvec3_bits_eq("FrameOrigin.pos vs RelativeFrameState", origin_pos, rel_pos);
    assert_dvec3_bits_eq("FrameOrigin.vel vs RelativeFrameState", origin_vel, rel_vel);
}

/// Compile-time assertion that the design-doc "After" mission-code
/// shape compiles cleanly: a system that uses `FrameOrigin` to read a
/// body's origin in the root inertial frame, holds no
/// `Res<FrameTreeR>`, never names a `FrameId`, and returns a typed
/// `Position<RootInertial>`. The system is registered (not run) — the
/// purpose is to exercise the SystemParam wiring against the
/// resources / queries `JeodPlugin` installs.
#[test]
fn after_diff_mission_code_shape_compiles() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
    app.world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH));

    fn read_body_origin_in_root(
        origin: FrameOrigin,
        root: Res<RootFrameEntityR>,
        bodies: Query<&FrameEntityC, With<TranslationalStateC>>,
    ) {
        for body_e in &bodies {
            let (_pos, _vel): (Position<RootInertial>, Velocity<RootInertial>) =
                origin.origin_in_root(root.0, body_e.0);
        }
    }

    // Registering the system is enough to type-check the SystemParam
    // shape; we don't need to actually run it for this compile-time
    // gate.
    let _id = app.world_mut().register_system(read_body_origin_in_root);
}
