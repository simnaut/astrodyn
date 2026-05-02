//! Issue #268 prototype validation: bit-identity between
//! `FrameTreeR.compute_relative_state` (the arena read path) and the
//! new `RelativeFrameState` `SystemParam` (the ECS-native read path).
//!
//! This test exists on the `study/268-frame-tree-ecs-native` branch
//! to demonstrate that the dual-write infrastructure (issue #268 Phase
//! B) keeps the two storage backends numerically equivalent — and to
//! document the **mission-code ergonomic improvement** by showing
//! both API surfaces side by side.
//!
//! ## Before / After diff (the load-bearing design-doc evidence)
//!
//! **Before** (arena read through `FrameTreeR` + `FrameId` components):
//! ```ignore
//! fn read_via_arena(
//!     frame_tree: Res<FrameTreeR>,
//!     bodies: Query<&BodyFrameIdC, With<MyBody>>,
//!     planets: Query<&SourcePfixFrameIdC, With<MyPlanet>>,
//! ) -> DVec3 {
//!     let body_fid = bodies.single().unwrap().0;
//!     let pfix_fid = planets.single().unwrap().0;
//!     frame_tree.0.compute_relative_state(pfix_fid, body_fid).trans.position
//! }
//! ```
//!
//! **After** (`RelativeFrameState` SystemParam wrapping ECS queries):
//! ```ignore
//! fn read_via_systemparam(
//!     rel: RelativeFrameState,
//!     bodies: Query<&FrameEntityC, With<MyBody>>,
//!     planets: Query<&PfixFrameEntityC, With<MyPlanet>>,
//! ) -> DVec3 {
//!     let body_e = bodies.single().unwrap().0;
//!     let pfix_e = planets.single().unwrap().0;
//!     rel.position(pfix_e, body_e)
//! }
//! ```
//!
//! The "After" surface never names a `FrameId`, never holds a
//! `Res<FrameTreeR>`, and reads like any other Bevy `SystemParam`.
//! This is the ergonomic delta the design doc commits to.

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::frame_param::RelativeFrameState;
use bevy_jeod::{
    BodyFrameIdC, DynamicsConfigC, FrameEntityC, FrameTreeR, GravityControlsC, JeodPlugin,
    MassPropertiesC, PfixFrameEntityC, PlanetBundle, RotationalStateC, SourcePfixFrameIdC,
    TranslationalStateC,
};
use glam::DVec3;
use jeod_sim::{
    DynamicsConfig, GravityControls, JeodQuat, MassProperties, PlanetConfig, RotationalState,
    TranslationalState, EARTH,
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

    // A simple rotational + translational body in low Earth orbit.
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

/// Validate that the dual-write (T6) keeps `FrameTransC` exactly in
/// sync with the arena's `state.trans` for the body frame.
#[test]
fn body_trans_arena_matches_ecs_after_step() {
    let (mut app, _planet_e, body_e) = build_app("Earth", &EARTH);
    step_once(&mut app);

    let world = app.world();

    // Arena read.
    let body_fid = world.get::<BodyFrameIdC>(body_e).unwrap().0;
    let frame_tree = world.resource::<FrameTreeR>();
    let arena_trans = frame_tree.0.get(body_fid).state.trans;

    // ECS read via the dual-written FrameTransC component.
    let body_frame_e = world.get::<FrameEntityC>(body_e).unwrap().0;
    let ecs_trans = *world
        .get::<bevy_jeod::FrameTransC>(body_frame_e)
        .expect("body frame entity should carry FrameTransC after dual-write");

    assert_dvec3_bits_eq(
        "body trans.position",
        arena_trans.position,
        ecs_trans.position,
    );
    assert_dvec3_bits_eq(
        "body trans.velocity",
        arena_trans.velocity,
        ecs_trans.velocity,
    );
}

/// Validate that the dual-write (T6) keeps the pfix frame entity's
/// `FrameRotC` + `FrameAngVelC` exactly in sync with the arena's
/// `state.rot` after `planet_fixed_rotation_system` runs.
#[test]
fn pfix_rot_arena_matches_ecs_after_step() {
    let (mut app, planet_e, _body_e) = build_app("Earth", &EARTH);
    step_once(&mut app);

    let world = app.world();

    // Arena read.
    let pfix_fid = world.get::<SourcePfixFrameIdC>(planet_e).unwrap().0;
    let frame_tree = world.resource::<FrameTreeR>();
    let arena_rot = frame_tree.0.get(pfix_fid).state.rot;

    // ECS read.
    let pfix_frame_e = world.get::<PfixFrameEntityC>(planet_e).unwrap().0;
    let ecs_rot = world
        .get::<bevy_jeod::FrameRotC>(pfix_frame_e)
        .expect("pfix frame entity should carry FrameRotC");
    let ecs_ang_vel = world
        .get::<bevy_jeod::FrameAngVelC>(pfix_frame_e)
        .expect("pfix frame entity should carry FrameAngVelC");

    // Quaternion components.
    for i in 0..4 {
        assert_bits_eq(
            &format!("pfix q_parent_this[{i}]"),
            arena_rot.q_parent_this.data[i],
            ecs_rot.q_parent_this.data[i],
        );
    }
    // Rotation matrix.
    for col in 0..3 {
        for row in 0..3 {
            assert_bits_eq(
                &format!("pfix t_parent_this[{col}][{row}]"),
                arena_rot.t_parent_this.col(col)[row],
                ecs_rot.t_parent_this.col(col)[row],
            );
        }
    }
    assert_dvec3_bits_eq("pfix ang_vel_this", arena_rot.ang_vel_this, ecs_ang_vel.0);
}

/// Validate that `RelativeFrameState::position(planet_inertial, body)`
/// returns the same value as `FrameTreeR.compute_relative_state(planet_inertial_id, body_id).trans.position`.
///
/// This is the **load-bearing** prototype validation: it proves the
/// new SystemParam read path produces bit-identical numerics to the
/// arena read path. The design doc cites this test as the "After"
/// surface's correctness evidence.
#[test]
fn relative_frame_state_matches_arena() {
    let (mut app, planet_e, body_e) = build_app("Earth", &EARTH);
    step_once(&mut app);

    // Arena answer: compute relative state (planet inertial → body)
    // via the existing FrameTreeR.compute_relative_state.
    let arena_pos = {
        let world = app.world();
        let body_fid = world.get::<BodyFrameIdC>(body_e).unwrap().0;
        let planet_inertial_fid = world.get::<bevy_jeod::SourceFrameIdC>(planet_e).unwrap().0;
        let frame_tree = world.resource::<FrameTreeR>();
        frame_tree
            .0
            .compute_relative_state(planet_inertial_fid, body_fid)
            .trans
            .position
    };

    // ECS answer: same query through RelativeFrameState SystemParam.
    let body_frame_e = app.world().get::<FrameEntityC>(body_e).unwrap().0;
    let planet_frame_e = app.world().get::<FrameEntityC>(planet_e).unwrap().0;
    let ecs_pos = app.world_mut().run_system_cached_with(
        |In((from, to)): In<(Entity, Entity)>, rel: RelativeFrameState| -> DVec3 {
            rel.position(from, to)
        },
        (planet_frame_e, body_frame_e),
    );
    let ecs_pos = ecs_pos.expect("run_system_cached_with should succeed");

    assert_dvec3_bits_eq("planet→body relative position", arena_pos, ecs_pos);
    println!("RelativeFrameState SystemParam vs FrameTreeR arena: bit-identical");
}

/// Same as above but for a deeper hierarchy: planet → pfix child →
/// (no body underneath, but exercises the compose-up path through
/// pfix). Body is parented under planet inertial as before.
///
/// Computes (pfix → body) which traverses pfix up to planet inertial
/// (1 step) then down to body (1 step) — a 2-segment compose with
/// rotation involved.
#[test]
fn relative_frame_state_matches_arena_through_pfix() {
    let (mut app, planet_e, body_e) = build_app("Earth", &EARTH);
    step_once(&mut app);

    // Arena: pfix → body relative state through compute_relative_state.
    let arena_pos = {
        let world = app.world();
        let body_fid = world.get::<BodyFrameIdC>(body_e).unwrap().0;
        let pfix_fid = world.get::<SourcePfixFrameIdC>(planet_e).unwrap().0;
        let frame_tree = world.resource::<FrameTreeR>();
        frame_tree
            .0
            .compute_relative_state(pfix_fid, body_fid)
            .trans
            .position
    };

    // ECS: same via SystemParam.
    let body_frame_e = app.world().get::<FrameEntityC>(body_e).unwrap().0;
    let pfix_frame_e = app.world().get::<PfixFrameEntityC>(planet_e).unwrap().0;
    let ecs_pos = app.world_mut().run_system_cached_with(
        |In((from, to)): In<(Entity, Entity)>, rel: RelativeFrameState| -> DVec3 {
            rel.position(from, to)
        },
        (pfix_frame_e, body_frame_e),
    );
    let ecs_pos = ecs_pos.expect("run_system_cached_with should succeed");

    assert_dvec3_bits_eq("pfix→body relative position", arena_pos, ecs_pos);
    println!("RelativeFrameState through pfix SystemParam vs FrameTreeR arena: bit-identical");
}
