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
//!   via `cfg.spawn_bevy::<astrodyn::Mars>(...)` carries the
//!   `<Mars>` slot, not the historical `<Earth>` one.
//! - `body_action_system::<P>` is registered per planet by
//!   `register_planet_systems::<P>`. A translational `BodyAction`
//!   queued via `BodyActionEvent::add_for::<astrodyn::Mars>(...)`
//!   lands in `BodyActionsR<Mars>` and the matching apply pass
//!   mutates `TranslationalStateC<Mars>` on the entity.

use std::time::Duration;

use astrodyn::{
    BodyAction, GravityControl, GravityRole, JeodQuat, MassProperties, RootInertial,
    RotationalState, TranslationalState, TranslationalStateTyped, Vec3Ext, VehicleBuilder, MARS,
};
use astrodyn_bevy::{
    register_planet_systems, AstrodynPlugin, BodyActionCommandsExt, BodyActionEvent,
    MassPropertiesC, PlanetBundle, SourceInertialPositionC, TranslationalStateC,
    VehicleConfigBevyExt,
};
use bevy::prelude::*;
use glam::DVec3;

const DT: f64 = 0.1;

fn body_state_initial() -> TranslationalStateTyped<RootInertial> {
    // 4_000 km circular-ish state around Mars (Mars radius ≈ 3389.5
    // km; this is a low Mars orbit). Numerics aren't load-bearing —
    // the test only needs a non-degenerate state to confirm the
    // queue path overwrites it.
    TranslationalStateTyped::<RootInertial> {
        position: DVec3::new(4_000_000.0, 0.0, 0.0).m_at::<RootInertial>(),
        velocity: DVec3::new(0.0, 3500.0, 0.0).m_per_s_at::<RootInertial>(),
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
        .spawn(PlanetBundle::<astrodyn::Mars>::point_mass("Mars", &MARS))
        .id();

    let cfg = VehicleBuilder::new()
        .with_translational(body_state_initial())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();

    let vehicle_id = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg.spawn_bevy::<astrodyn::Mars>(&mut commands_queue, &[mars]);
        world.flush();
        id
    };

    assert!(
        app.world()
            .entity(vehicle_id)
            .contains::<TranslationalStateC<astrodyn::Mars>>(),
        "spawn_bevy::<Mars> must insert TranslationalStateC<Mars> on the vehicle entity",
    );
    assert!(
        !app.world()
            .entity(vehicle_id)
            .contains::<TranslationalStateC<astrodyn::Earth>>(),
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
    app.add_plugins(AstrodynPlugin);
    register_planet_systems::<astrodyn::Mars>(&mut app);

    // Mars as the gravity source for the vehicle. `register_planet_systems::<Mars>`
    // wires `register_source_frames_system::<Mars>` so the Mars
    // entity's frame hierarchy is set up before EphemerisUpdate.
    let mars = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Mars>::point_mass("Mars", &MARS))
        .id();
    // The Mars source needs a SourceInertialPositionC for the
    // gravity / ephemeris path; PlanetBundle already includes it via
    // `point_mass` but make the dependency explicit.
    app.world_mut()
        .entity_mut(mars)
        .insert(SourceInertialPositionC::default());

    let cfg = VehicleBuilder::new()
        .with_translational(body_state_initial())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();

    let vehicle = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg.spawn_bevy::<astrodyn::Mars>(&mut commands_queue, &[mars]);
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
            .contains::<TranslationalStateC<astrodyn::Mars>>(),
        "preconditions: spawn_bevy::<Mars> placed the Mars-tagged slot",
    );

    // Spawn a vehicle in Mars then verify the spawn-time state
    // matches `body_state_initial()` before the queue overrides it.
    let pre_state = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .entity(vehicle)
            .get::<TranslationalStateC<astrodyn::Mars>>()
            .unwrap()
            .0,
    );
    assert_eq!(pre_state.position, body_state_initial().position.raw_si());

    // Confirm a `MassPropertiesC` is present (required by the
    // body_action_system query's With<DynamicsConfigC> filter and the
    // mass-update system's downstream consumers).
    assert!(app.world().entity(vehicle).contains::<MassPropertiesC>());

    // Queue a Mars-tagged translational init. With the queue
    // genuinely planet-generic, the matching `body_action_system::<Mars>`
    // pass mutates the Mars-tagged slot on the next FixedUpdate.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>()
        .write(BodyActionEvent::add_for::<astrodyn::Mars>(
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

    let post_state = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .entity(vehicle)
            .get::<TranslationalStateC<astrodyn::Mars>>()
            .expect("Mars-tagged translational state must remain present after the apply pass")
            .0,
    );

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
    let from_initial = (post_state.position - body_state_initial().position.raw_si()).length();
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
    app.add_plugins(AstrodynPlugin);
    register_planet_systems::<astrodyn::Mars>(&mut app);

    let earth_planet = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass(
            "Earth",
            &astrodyn::EARTH,
        ))
        .id();
    let mars_planet = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Mars>::point_mass("Mars", &MARS))
        .id();

    let cfg_mars = VehicleBuilder::new()
        .with_translational(body_state_initial())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();
    let cfg_earth = VehicleBuilder::new()
        .with_translational(TranslationalStateTyped::<RootInertial> {
            position: DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<RootInertial>(),
            velocity: DVec3::new(0.0, 7000.0, 0.0).m_per_s_at::<RootInertial>(),
        })
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();

    let mars_body = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg_mars.spawn_bevy::<astrodyn::Mars>(&mut commands_queue, &[mars_planet]);
        world.flush();
        id
    };
    let earth_body = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg_earth.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth_planet]);
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
    let mars_post = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .entity(mars_body)
            .get::<TranslationalStateC<astrodyn::Mars>>()
            .unwrap()
            .0,
    );
    // One DT of orbital propagation drifts the Mars body by at most
    // ~400 m (3500 m/s velocity × 0.1 s DT). 10 km is a generous
    // bound that still excludes any silent route through an
    // Earth-tagged apply (which would teleport the body 4 Mm to the
    // Earth replacement position).
    let mars_drift = (mars_post.position - body_state_initial().position.raw_si()).length();
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
    let earth_post = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .entity(earth_body)
            .get::<TranslationalStateC<astrodyn::Earth>>()
            .unwrap()
            .0,
    );
    let earth_drift = (earth_post.position - DVec3::new(8_000_000.0, 0.0, 0.0)).length();
    assert!(
        earth_drift < 10_000.0,
        "Earth-tagged action must overwrite the Earth body's \
         translational position before the same-tick integrator runs. \
         Observed drift {earth_drift} m from the replacement state, \
         post-state {earth_post:?}",
    );
}

#[test]
fn planet_agnostic_remove_cancels_mars_pending_without_disturbing_earth() {
    // Cross-planet `Remove` contract: a `BodyActionEvent::Remove`
    // sent from code that holds no `<P>` witness must reach every
    // per-planet `BodyActionsR<P>` queue and drop matching entries.
    // The fan-out is the implementation of the docstring on
    // `BodyActionEvent::Remove` ("a name-based remove from
    // Earth-orbit code reaches a Mars-tagged add on the same name
    // even when the calling system holds no `<P>` witness").
    //
    // This test:
    //   - registers Earth (via `AstrodynPlugin`) and Mars (via
    //     `register_planet_systems::<Mars>`),
    //   - spawns one Earth body and one Mars body, each with a
    //     queued `Add` whose name shares a unique tag for the Mars
    //     body and a separate one for the Earth body,
    //   - sends a planet-agnostic `Remove` for the Mars body's tag,
    //   - ticks once, then asserts:
    //     (a) the Mars body's translational state is the spawn state
    //         (the queued Mars `Add` was dropped before `body_action_system::<Mars>`
    //         ran), and
    //     (b) the Earth body's translational state reflects the
    //         Earth `Add`'s replacement (the Remove targeted only the
    //         Mars-named action).
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    register_planet_systems::<astrodyn::Mars>(&mut app);

    let earth_planet = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass(
            "Earth",
            &astrodyn::EARTH,
        ))
        .id();
    let mars_planet = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Mars>::point_mass("Mars", &MARS))
        .id();

    let cfg_mars = VehicleBuilder::new()
        .with_translational(body_state_initial())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();
    let cfg_earth = VehicleBuilder::new()
        .with_translational(TranslationalStateTyped::<RootInertial> {
            position: DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<RootInertial>(),
            velocity: DVec3::new(0.0, 7000.0, 0.0).m_per_s_at::<RootInertial>(),
        })
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();

    let mars_body = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg_mars.spawn_bevy::<astrodyn::Mars>(&mut commands_queue, &[mars_planet]);
        world.flush();
        id
    };
    let earth_body = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg_earth.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth_planet]);
        world.flush();
        id
    };

    // Earth replacement state — the post-tick Earth body should
    // land near this (one DT of integration from this point).
    let earth_replacement = TranslationalState {
        position: DVec3::new(8_000_000.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 6500.0, 0.0),
    };
    // Mars replacement state — the Mars body must NOT land near
    // this; the planet-agnostic Remove should drop the queued Mars
    // Add before its apply pass runs.
    let mars_replacement = body_state_replacement();

    {
        let mut messages = app
            .world_mut()
            .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>();
        // Earth Add with one name.
        messages.write(BodyActionEvent::add(
            earth_body,
            BodyAction::InitTrans {
                state: earth_replacement,
            },
            Some("earth_only_init"),
        ));
        // Mars Add with a *different* name — the Remove targets
        // only the Mars-named action.
        messages.write(BodyActionEvent::add_for::<astrodyn::Mars>(
            mars_body,
            BodyAction::InitTrans {
                state: mars_replacement,
            },
            Some("mars_init_to_be_cancelled"),
        ));
        // Planet-agnostic Remove for the Mars-named action. The
        // sender holds no `<P>` witness — this is the documented
        // cross-planet contract under test.
        messages.write(BodyActionEvent::remove("mars_init_to_be_cancelled"));
    }

    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    // (a) Mars body's queued action did NOT execute: its position
    //     stays near the spawn-time `body_state_initial()` (one DT
    //     of integration drift, well under 10 km), NOT near
    //     `mars_replacement` (which is ~6.4 Mm away).
    let mars_post = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .entity(mars_body)
            .get::<TranslationalStateC<astrodyn::Mars>>()
            .unwrap()
            .0,
    );
    let mars_drift_from_initial =
        (mars_post.position - body_state_initial().position.raw_si()).length();
    assert!(
        mars_drift_from_initial < 10_000.0,
        "Planet-agnostic Remove must drop the Mars-tagged Add before \
         `body_action_system::<Mars>` runs — Mars body should still be near \
         its spawn-time position. Observed drift {mars_drift_from_initial} m, \
         post-state {mars_post:?}",
    );
    let mars_dist_to_replacement = (mars_post.position - mars_replacement.position).length();
    assert!(
        mars_dist_to_replacement > 1_000_000.0,
        "Mars body must NOT reflect the cancelled replacement state — \
         observed only {mars_dist_to_replacement} m from the replacement, \
         which would indicate the Remove failed to reach `BodyActionsR<Mars>`. \
         Post-state {mars_post:?}",
    );

    // (b) Earth body's queued action DID execute: its post-tick
    //     position is near `earth_replacement` (one DT of integration
    //     drift).
    let earth_post = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .entity(earth_body)
            .get::<TranslationalStateC<astrodyn::Earth>>()
            .unwrap()
            .0,
    );
    let earth_drift = (earth_post.position - earth_replacement.position).length();
    assert!(
        earth_drift < 10_000.0,
        "Planet-agnostic Remove must NOT cancel a differently-named Earth Add — \
         the Earth body should land near its replacement state. Observed drift \
         {earth_drift} m from `earth_replacement`, post-state {earth_post:?}",
    );
}

#[test]
#[should_panic(expected = "no per-planet body-action pipeline is registered for that planet")]
fn add_for_unregistered_planet_panics_with_named_diagnostic() {
    // Concern 1 / Fail-Loudly: queuing a `BodyActionEvent::Add`
    // tagged for a planet whose pipeline was never registered must
    // panic with a diagnostic that names the planet (via
    // `std::any::type_name::<P>()`) and points to
    // `register_planet_systems::<P>` as the fix. Without the guard,
    // the message would land in the unified `Messages<BodyActionEvent>`
    // buffer, be skipped by every existing per-planet intake (TypeId
    // mismatch), and silently age out of the double-buffer with no
    // observable effect — the regression the deleted Earth-pinned
    // `should_panic` test was guarding against.
    //
    // The mission here calls `AstrodynPlugin::build` (which registers
    // Earth) and spawns a Mars body but *never* calls
    // `register_planet_systems::<Mars>`. The `add_for::<Mars>`
    // message must trip the
    // `body_action_unregistered_planet_fence_system` panic — the
    // direct-`MessageWriter` path (the `Commands` path has its own
    // call-site assertion, exercised by a separate test).
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    // NOTE: deliberately NOT calling `register_planet_systems::<Mars>`.

    let earth_planet = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass(
            "Earth",
            &astrodyn::EARTH,
        ))
        .id();

    // Spawn an Earth body so the App is otherwise well-formed —
    // the Mars `Add` message is the only misconfiguration on this
    // tick.
    let cfg_earth = VehicleBuilder::new()
        .with_translational(TranslationalStateTyped::<RootInertial> {
            position: DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<RootInertial>(),
            velocity: DVec3::new(0.0, 7000.0, 0.0).m_per_s_at::<RootInertial>(),
        })
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();
    let earth_body = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg_earth.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth_planet]);
        world.flush();
        id
    };

    // Direct-writer path: an `add_for::<Mars>` lands in the message
    // buffer. The fence system, registered by `AstrodynPlugin::build`
    // and chained between the per-planet intakes and apply passes,
    // must observe it and panic.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>()
        .write(BodyActionEvent::add_for::<astrodyn::Mars>(
            earth_body,
            BodyAction::InitTrans {
                state: body_state_replacement(),
            },
            Some("mars_init_unregistered"),
        ));

    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

#[test]
#[should_panic(expected = "no per-planet body-action pipeline is registered for that planet")]
fn add_body_action_for_unregistered_planet_panics_at_commands_flush() {
    // Companion to `add_for_unregistered_planet_panics_with_named_diagnostic`:
    // exercises the `Commands` path
    // (`BodyActionCommandsExt::add_body_action_for::<P>`) instead of
    // a direct `MessageWriter` write. The `Commands::queue` closure
    // checks `RegisteredPlanetsR` at flush time and panics with the
    // same diagnostic shape (named planet, named call-site fix). Two
    // separate tests because the two surfaces panic at different
    // sites — the direct-writer path is caught by the
    // post-intake fence system, the `Commands` path is caught at
    // queue-flush time, before the message ever reaches the buffer.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    // NOTE: deliberately NOT calling `register_planet_systems::<Mars>`.

    let earth_planet = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass(
            "Earth",
            &astrodyn::EARTH,
        ))
        .id();
    let cfg_earth = VehicleBuilder::new()
        .with_translational(TranslationalStateTyped::<RootInertial> {
            position: DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<RootInertial>(),
            velocity: DVec3::new(0.0, 7000.0, 0.0).m_per_s_at::<RootInertial>(),
        })
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();
    let earth_body = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg_earth.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth_planet]);
        world.flush();
        id
    };

    // Queue a Mars action via the `Commands` extension. Bevy's
    // `Commands::queue` runs the closure at the next flush; the
    // closure consults `RegisteredPlanetsR`, finds Mars unregistered,
    // and panics. We perform the queue + flush by spawning a
    // `CommandQueue` directly (avoids the ZST-only restriction on
    // `run_system_cached`).
    {
        use bevy::ecs::world::CommandQueue;
        let mut queue = CommandQueue::default();
        let world = app.world_mut();
        {
            let mut commands = bevy::ecs::system::Commands::new(&mut queue, world);
            commands.add_body_action_for::<astrodyn::Mars>(
                earth_body,
                BodyAction::InitTrans {
                    state: body_state_replacement(),
                },
                Some("mars_init_via_commands"),
            );
        }
        // Apply the queued commands: the deferred closure consults
        // `RegisteredPlanetsR`, finds Mars unregistered, and panics
        // before the message buffer is touched.
        queue.apply(world);
    }

    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}
