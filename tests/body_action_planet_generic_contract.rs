//! Planet-generic body-action contract: with `BodyActionsR<P>` per
//! planet and `spawn_bevy::<P>` carrying the witness, a Mars-orbit
//! mission can use both surfaces end-to-end without any manual
//! `<Earth>` fix-up.
//!
//! This file replaces the Earth-pinned negative-shape regression
//! tests that previously locked in the queue-side Earth-only
//! contract. The two checks here mirror the two writers that used
//! to mint `<Earth>`-tagged storage:
//!
//! - `spawn_bevy::<P>` inserts `TranslationalStateC<P>` for the
//!   planet `P` chosen at the call site. A Mars-orbit body spawned
//!   via `cfg.spawn_bevy::<jeod_sim::Mars>(...)` carries the
//!   `<Mars>` slot, not the historical `<Earth>` one.
//! - `body_action_system::<P>` is registered per planet by
//!   `register_planet_systems::<P>`. A translational `BodyAction`
//!   queued via `BodyActionEvent::add_for::<jeod_sim::Mars>(...)`
//!   lands in `BodyActionsR<Mars>` and the matching apply pass
//!   mutates `TranslationalStateC<Mars>` on the entity.

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    register_planet_systems, BodyActionEvent, JeodPlugin, MassPropertiesC, PlanetBundle,
    SourceInertialPositionC, TranslationalStateC, VehicleConfigBevyExt,
};
use glam::DVec3;
use jeod_sim::{
    BodyAction, GravityControl, JeodQuat, MassProperties, RotationalState, TranslationalState,
    VehicleBuilder, MARS,
};

const DT: f64 = 0.1;

fn body_state_initial() -> TranslationalState {
    // 4_000 km circular-ish state around Mars (Mars radius ≈ 3389.5
    // km; this is a low Mars orbit). Numerics aren't load-bearing —
    // the test only needs a non-degenerate state to confirm the
    // queue path overwrites it.
    TranslationalState {
        position: DVec3::new(4_000_000.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 3500.0, 0.0),
    }
}

fn body_state_replacement() -> TranslationalState {
    // Distinct second state so the post-apply assertion can
    // distinguish the queued-action result from the spawn-time value.
    TranslationalState {
        position: DVec3::new(0.0, 4_500_000.0, 0.0),
        velocity: DVec3::new(-3300.0, 0.0, 0.0),
    }
}

fn vehicle_mass() -> MassProperties {
    MassProperties::with_inertia(
        1_000.0,
        glam::DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0)),
        DVec3::ZERO,
    )
}

fn initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ..Default::default()
    }
}

#[test]
fn spawn_bevy_inserts_planet_tagged_translational_storage_for_mars() {
    // `spawn_bevy::<Mars>` lands the planet witness in the
    // translational-state component tag. The matching `<Earth>` slot
    // must NOT be present — that would be the silent miswire the
    // spawn-side refactor exists to forbid.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);

    let mars = app
        .world_mut()
        .spawn(PlanetBundle::<jeod_sim::Mars>::point_mass("Mars", &MARS))
        .id();

    let cfg = VehicleBuilder::new()
        .with_state(body_state_initial())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let vehicle_id = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg.spawn_bevy::<jeod_sim::Mars>(&mut commands_queue, &[mars]);
        world.flush();
        id
    };

    assert!(
        app.world()
            .entity(vehicle_id)
            .contains::<TranslationalStateC<jeod_sim::Mars>>(),
        "spawn_bevy::<Mars> must insert TranslationalStateC<Mars> on the vehicle entity",
    );
    assert!(
        !app.world()
            .entity(vehicle_id)
            .contains::<TranslationalStateC<jeod_sim::Earth>>(),
        "spawn_bevy::<Mars> must NOT insert a residual <Earth>-tagged \
         translational slot — that would silently miswire a Mars body \
         into the Earth-tagged downstream consumer pipeline",
    );
}

#[test]
fn body_action_for_mars_writes_through_mars_tagged_storage() {
    // Wire a Mars planet pipeline via `register_planet_systems::<Mars>`,
    // spawn a body via `spawn_bevy::<Mars>`, then queue a translational
    // `BodyAction` for Mars via `BodyActionEvent::add_for::<Mars>`.
    // After one FixedUpdate tick, the body's
    // `TranslationalStateC<Mars>` must reflect the queued state — no
    // panic, no silent skip. This is the end-to-end check that the
    // queue is genuinely planet-generic.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
    register_planet_systems::<jeod_sim::Mars>(&mut app);

    // Mars as the gravity source for the vehicle. `register_planet_systems::<Mars>`
    // wires `register_source_frames_system::<Mars>` so the Mars
    // entity's frame hierarchy is set up before EphemerisUpdate.
    let mars = app
        .world_mut()
        .spawn(PlanetBundle::<jeod_sim::Mars>::point_mass("Mars", &MARS))
        .id();
    // The Mars source needs a SourceInertialPositionC for the
    // gravity / ephemeris path; PlanetBundle already includes it via
    // `point_mass` but make the dependency explicit.
    app.world_mut()
        .entity_mut(mars)
        .insert(SourceInertialPositionC::default());

    let cfg = VehicleBuilder::new()
        .with_state(body_state_initial())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let vehicle = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg.spawn_bevy::<jeod_sim::Mars>(&mut commands_queue, &[mars]);
        world.flush();
        id
    };

    // Sanity: spawn_bevy still inserts a Mars-tagged slot the apply
    // pass can mutate. (Defends against a spawn-time regression that
    // would make the second assertion pass for the wrong reason —
    // an Earth-tagged slot left behind by spawn_bevy would be
    // mutated by the Earth apply pass instead.)
    assert!(
        app.world()
            .entity(vehicle)
            .contains::<TranslationalStateC<jeod_sim::Mars>>(),
        "preconditions: spawn_bevy::<Mars> placed the Mars-tagged slot",
    );

    // Spawn a vehicle in Mars then verify the spawn-time state
    // matches `body_state_initial()` before the queue overrides it.
    let pre_state = app
        .world()
        .entity(vehicle)
        .get::<TranslationalStateC<jeod_sim::Mars>>()
        .unwrap()
        .0
        .to_untyped();
    assert_eq!(pre_state.position, body_state_initial().position);

    // Confirm a `MassPropertiesC` is present (required by the
    // body_action_system query's With<DynamicsConfigC> filter and the
    // mass-update system's downstream consumers).
    assert!(app.world().entity(vehicle).contains::<MassPropertiesC>());

    // Queue a Mars-tagged translational init. With the queue
    // genuinely planet-generic, the matching `body_action_system::<Mars>`
    // pass mutates the Mars-tagged slot on the next FixedUpdate.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>()
        .write(BodyActionEvent::add_for::<jeod_sim::Mars>(
            vehicle,
            BodyAction::InitTrans {
                state: body_state_replacement(),
            },
            Some("init_trans_planet_generic"),
        ));

    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    let post_state = app
        .world()
        .entity(vehicle)
        .get::<TranslationalStateC<jeod_sim::Mars>>()
        .expect("Mars-tagged translational state must remain present after the apply pass")
        .0
        .to_untyped();

    // The body_action apply pass runs before integration in the same
    // FixedUpdate tick. The post-tick state is therefore one DT of
    // integration starting from `body_state_replacement()`. With DT
    // = 0.1 s and orbital speed ~3500 m/s, the position drifts at
    // most ~400 m from the replacement state — well under the
    // 4500 km Y-component magnitude that distinguishes the
    // replacement from the spawn state. We assert closeness to the
    // replacement and farness from the spawn state to confirm the
    // queue overwrote storage and the integrator started from the
    // overwritten value.
    let drift = (post_state.position - body_state_replacement().position).length();
    assert!(
        drift < 1_000.0,
        "body_action_system::<Mars> must overwrite the Mars-tagged \
         translational position before integration runs — observed a \
         drift of {drift} m from the replacement state, expected < 1 km \
         (one DT of orbital propagation). Post-state: {post_state:?}",
    );
    let from_initial = (post_state.position - body_state_initial().position).length();
    assert!(
        from_initial > 1_000_000.0,
        "body_action_system::<Mars> must move the body away from its \
         spawn-time position — observed only {from_initial} m of drift, \
         which would indicate the apply pass never ran. Post-state: {post_state:?}",
    );
    // Velocity is overwritten too; integration applies a small change
    // per DT but the bulk magnitude must follow the replacement.
    let v_drift = (post_state.velocity - body_state_replacement().velocity).length();
    assert!(
        v_drift < 100.0,
        "body_action_system::<Mars> must overwrite the Mars-tagged \
         translational velocity — observed velocity drift of {v_drift} m/s \
         from the replacement value, expected < 100 m/s after one DT. \
         Post-state: {post_state:?}",
    );
}

#[test]
fn earth_tagged_action_does_not_mutate_mars_body() {
    // Cross-planet routing fence: an `add_for::<Earth>` against a
    // body whose translational storage is `<Mars>` must NOT mutate
    // the Mars slot. The Earth apply pass queries
    // `&mut TranslationalStateC<Earth>` and never sees the body;
    // the body's own (Mars) slot stays untouched until a
    // Mars-tagged action is queued. The Earth apply pass *will*
    // panic (missing `TranslationalStateC<Earth>` on the body),
    // which is the correct fail-loud diagnostic — but the panic is
    // not the contract under test here. We instead verify that a
    // Mars-tagged add against the same body works in the same App
    // *and* leaves the Earth pipeline's view of the body
    // unchanged. To exercise both pipelines without the Earth
    // panic, route the Earth add against a *different* (Earth-only)
    // body and check that the Mars body's state is untouched by the
    // Earth apply pass.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
    register_planet_systems::<jeod_sim::Mars>(&mut app);

    let earth_planet = app
        .world_mut()
        .spawn(PlanetBundle::<jeod_sim::Earth>::point_mass(
            "Earth",
            &jeod_sim::EARTH,
        ))
        .id();
    let mars_planet = app
        .world_mut()
        .spawn(PlanetBundle::<jeod_sim::Mars>::point_mass("Mars", &MARS))
        .id();

    let cfg_mars = VehicleBuilder::new()
        .with_state(body_state_initial())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();
    let cfg_earth = VehicleBuilder::new()
        .with_state(TranslationalState {
            position: DVec3::new(7_000_000.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7000.0, 0.0),
        })
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let mars_body = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg_mars.spawn_bevy::<jeod_sim::Mars>(&mut commands_queue, &[mars_planet]);
        world.flush();
        id
    };
    let earth_body = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg_earth.spawn_bevy::<jeod_sim::Earth>(&mut commands_queue, &[earth_planet]);
        world.flush();
        id
    };

    // Queue *only* an Earth-tagged action against the Earth body.
    // The Mars body has no pending action and must stay at its
    // spawn-time state across the FixedUpdate tick.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>()
        .write(BodyActionEvent::add(
            earth_body,
            BodyAction::InitTrans {
                state: TranslationalState {
                    position: DVec3::new(8_000_000.0, 0.0, 0.0),
                    velocity: DVec3::new(0.0, 6500.0, 0.0),
                },
            },
            Some("earth_init"),
        ));

    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    // Mars body untouched by the Earth-tagged action: the post-tick
    // position is the propagation result starting from the spawn
    // state, NOT the Earth replacement state. Numerically the Mars
    // body propagates one DT under Mars gravity; we just need to
    // confirm it didn't snap to the Earth replacement state.
    let mars_post = app
        .world()
        .entity(mars_body)
        .get::<TranslationalStateC<jeod_sim::Mars>>()
        .unwrap()
        .0
        .to_untyped();
    // One DT of orbital propagation drifts the Mars body by at most
    // ~400 m (3500 m/s velocity × 0.1 s DT). 10 km is a generous
    // bound that still excludes any silent route through an
    // Earth-tagged apply (which would teleport the body 4 Mm to the
    // Earth replacement position).
    let mars_drift = (mars_post.position - body_state_initial().position).length();
    assert!(
        mars_drift < 10_000.0,
        "Mars body must NOT see the Earth-tagged init's position — \
         after one DT under Mars gravity it should still be near its \
         spawn-time initial position. Observed drift: {mars_drift} m, \
         post-state {mars_post:?}",
    );
    assert_ne!(
        mars_post.position,
        DVec3::new(8_000_000.0, 0.0, 0.0),
        "Mars body must NOT pick up the Earth-tagged action's position",
    );

    // Earth body got the Earth-tagged init applied. After one DT of
    // integration starting from the Earth replacement state, the
    // body has drifted at most ~700 m (7000 m/s × 0.1 s); the
    // 10 km bound easily excludes any path that left the Earth body
    // at its spawn position.
    let earth_post = app
        .world()
        .entity(earth_body)
        .get::<TranslationalStateC<jeod_sim::Earth>>()
        .unwrap()
        .0
        .to_untyped();
    let earth_drift = (earth_post.position - DVec3::new(8_000_000.0, 0.0, 0.0)).length();
    assert!(
        earth_drift < 10_000.0,
        "Earth-tagged action must overwrite the Earth body's \
         translational position before the same-tick integrator runs. \
         Observed drift {earth_drift} m from the replacement state, \
         post-state {earth_post:?}",
    );
}
