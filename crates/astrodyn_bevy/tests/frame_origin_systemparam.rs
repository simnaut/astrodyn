//! Mission-code shape tests for the [`FrameOrigin`] /
//! [`RelativeFrameState`] `SystemParam`s.
//!
//! These tests document the supported mission-facing surface for
//! cross-frame state queries:
//!
//! ```ignore
//! fn read_body_origin_in_root(
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
//! `FrameOrigin` returns
//! `(Position<RootInertial>, Velocity<RootInertial>)` directly — the
//! caller never names a frame id, never holds a frame-tree
//! resource, and the typed phantoms are stamped by the
//! `SystemParam` itself (no `.from_raw_si` at the call site).

use std::time::Duration;

use astrodyn::{
    DynamicsConfig, GravityControls, JeodQuat, MassProperties, PlanetConfig, Position,
    RootInertial, RotationalState, TranslationalState, Velocity, EARTH,
};
use astrodyn_bevy::frame_param::{FrameOrigin, RelativeFrameState};
use astrodyn_bevy::{
    AstrodynPlugin, DynamicsConfigC, FrameEntityC, GravityControlsC, IntegrationDtR,
    MassPropertiesC, PfixFrameEntityC, PlanetBundle, RootFrameEntityR, RotationalStateC,
    TranslationalStateC,
};
use bevy::prelude::*;
use glam::DVec3;

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
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);

    let planet_e = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass(
            planet_name,
            planet,
        ))
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
            TranslationalStateC::<astrodyn::Earth>::from_untyped(trans),
            RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(&(rot))),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(1.0)),
            )),
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

fn assert_dvec3_eq(label: &str, a: DVec3, b: DVec3) {
    // Per-component `to_bits()` equality so the assertion really is a
    // bit-identity fence — `DVec3::abs_diff_eq(.., 0.0)` only checks
    // numeric equality (`+0.0 == -0.0`, NaN-payload-insensitive) which
    // would let bit-distinct `FrameOrigin` / `RelativeFrameState`
    // results pass here despite the panic message claiming
    // bit-identical. The two helpers must produce the same bits for
    // every code path, not just the same numeric value.
    let bits_eq = a.x.to_bits() == b.x.to_bits()
        && a.y.to_bits() == b.y.to_bits()
        && a.z.to_bits() == b.z.to_bits();
    assert!(bits_eq, "{label} not bit-identical: a={a:?}, b={b:?}");
}

/// `FrameOrigin::origin_in_root(root, root)` returns
/// `(DVec3::ZERO, DVec3::ZERO)` typed at the root inertial phantom —
/// the identity short-circuit. Same contract as
/// `astrodyn::frame_origin(tree, root, root)`.
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

/// `FrameOrigin::origin_in` and
/// `RelativeFrameState::position_velocity` are siblings — the
/// former is sugar over the latter, so the two must agree on every
/// `(ancestor, frame)` query. Exercise both against the same
/// hierarchy walk (planet pfix → body) and assert numeric equality.
#[test]
fn frame_origin_and_relative_frame_state_agree() {
    let (mut app, planet_e, body_e) = build_app("Earth", &EARTH);
    step_once(&mut app);

    let body_frame_e = app.world().get::<FrameEntityC>(body_e).unwrap().0;
    let pfix_frame_e = app.world().get::<PfixFrameEntityC>(planet_e).unwrap().0;

    let (origin_pos, origin_vel) = app
        .world_mut()
        .run_system_cached_with(
            |In((ancestor, frame)): In<(Entity, Entity)>, origin: FrameOrigin| -> (DVec3, DVec3) {
                origin.origin_in(ancestor, frame)
            },
            (pfix_frame_e, body_frame_e),
        )
        .expect("run_system_cached_with should succeed");

    let (rel_pos, rel_vel) = app
        .world_mut()
        .run_system_cached_with(
            |In((ancestor, frame)): In<(Entity, Entity)>,
             rel: RelativeFrameState|
             -> (DVec3, DVec3) { rel.position_velocity(ancestor, frame) },
            (pfix_frame_e, body_frame_e),
        )
        .expect("run_system_cached_with should succeed");

    assert_dvec3_eq("FrameOrigin.pos vs RelativeFrameState", origin_pos, rel_pos);
    assert_dvec3_eq("FrameOrigin.vel vs RelativeFrameState", origin_vel, rel_vel);
}

/// Compile-time assertion that the mission-code shape compiles
/// cleanly: a system that uses `FrameOrigin` to read a body's origin
/// in the root inertial frame, never names a frame id, never holds a
/// frame-tree resource, and binds typed `Position<RootInertial>` /
/// `Velocity<RootInertial>` values directly (no `.from_raw_si` /
/// phantom-stamping at the call site). The system is registered
/// (not run) — the purpose is to exercise the SystemParam wiring
/// against the resources / queries `AstrodynPlugin` installs.
#[test]
fn after_diff_mission_code_shape_compiles() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);
    app.world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH));

    fn read_body_origin_in_root(
        origin: FrameOrigin,
        root: Res<RootFrameEntityR>,
        bodies: Query<&FrameEntityC, With<TranslationalStateC<astrodyn::Earth>>>,
    ) {
        for body_e in &bodies {
            let (_pos, _vel): (Position<RootInertial>, Velocity<RootInertial>) =
                origin.origin_in_root(root.0, body_e.0);
        }
    }

    let _id = app.world_mut().register_system(read_body_origin_in_root);
}
