//! Regression tests pinning the Earth-only contract for two writers
//! that mint `TranslationalStateC` storage with the concrete
//! `<jeod_sim::Earth>` tag (rather than a generic `<P>`):
//!
//! - `body_action_system`'s query asks for
//!   `Option<&mut TranslationalStateC<jeod_sim::Earth>>` and writes
//!   through it on `apply_translational`. A queued translational
//!   `BodyAction` against a body whose translational storage is
//!   `TranslationalStateC<Mars>` (i.e. the body integrates against a
//!   non-Earth source) finds no matching `<Earth>` slot and panics
//!   with a diagnostic that *forbids* the silent-miswire workaround
//!   ("just add an `<Earth>` slot") and instead directs the operator
//!   to use the planet-typed direct-write path.
//! - `VehicleConfigBevyExt::spawn_bevy` inserts the translational
//!   state slot as `TranslationalStateC<jeod_sim::Earth>` regardless
//!   of which planet pipeline is registered. The contract test
//!   asserts the post-spawn entity carries the `<Earth>` tag — a
//!   future change that flips the helper to wildcard `<P>` would
//!   trip this test loudly.
//!
//! Both tests are deliberately negative-shape: they document the
//! current restriction so the queue-side / spawn-side refactor that
//! lifts it (tracked separately) cannot land without updating these
//! locked-in expectations.

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    BodyActionEvent, DynamicsConfigC, GravityControlsC, GravitySourceC, JeodPlugin,
    MassPropertiesC, PlanetBundle, SourceInertialPositionC, TranslationalStateC,
    VehicleConfigBevyExt,
};
use glam::DVec3;
use jeod_sim::{
    BodyAction, DynamicsConfig, GravityControl, GravityControls, GravityModel, GravitySource,
    MassProperties, RotationalState, TranslationalState, VehicleBuilder, EARTH,
};

const DT: f64 = 0.1;

fn body_state() -> TranslationalState {
    // Arbitrary non-degenerate state — the test only cares that the
    // panic fires, not the numerics.
    TranslationalState {
        position: DVec3::new(7_000_000.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7000.0, 0.0),
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
    RotationalState::default()
}

#[test]
#[should_panic(expected = "queued-action path is currently Earth-only")]
fn body_action_translational_panics_when_only_mars_slot_present() {
    // Verifies that queuing a translational `BodyAction` against a
    // body whose translational storage is `TranslationalStateC<Mars>`
    // panics with a diagnostic that names the Earth-only restriction.
    // The diagnostic must *not* tell the caller to add an
    // `<Earth>`-tagged slot to a non-Earth body — that workaround
    // would silently land the action in a wrong-planet storage.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    // Earth as the gravity source (so the body has *some* gravity
    // wiring; the test focuses on the `BodyAction` translational
    // path, not on integration). The body's *integration* frame is
    // the Earth source's, but the body's translational *storage* is
    // tagged `<Mars>` — that mismatch is the situation under test.
    let earth = app
        .world_mut()
        .spawn((
            Name::new("Earth"),
            GravitySourceC(GravitySource {
                mu: EARTH.shape.mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<jeod_sim::Earth>::default(),
        ))
        .id();

    let vehicle = app
        .world_mut()
        .spawn((
            Name::new("MarsBody"),
            // Only a `<Mars>`-tagged slot; no `<Earth>`-tagged slot.
            TranslationalStateC::<jeod_sim::Mars>::from(body_state()),
            MassPropertiesC::from(vehicle_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false)],
            }),
        ))
        .id();

    // Queue a translational init action and drive the schedule. The
    // `body_action_system` runs on `FixedUpdate`, so a single
    // `advance_by + run_schedule` is enough to land us in the
    // `apply_translational` arm.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<BodyActionEvent>>()
        .write(BodyActionEvent::add(
            vehicle,
            BodyAction::InitTrans {
                state: body_state(),
            },
            Some("init_trans_panic_test"),
        ));

    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

#[test]
fn spawn_bevy_inserts_earth_tagged_translational_storage() {
    // Pins the current `<Earth>`-only contract on
    // `VehicleConfigBevyExt::spawn_bevy`. A future change that flips
    // the helper to wildcard `<P>` (or parameterizes
    // `VehicleConfig<P>`) must update this test to match the new
    // contract — silently changing the inserted tag would otherwise
    // produce non-Earth bodies with a stale `<Earth>` storage slot.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);

    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<jeod_sim::Earth>::point_mass("Earth", &EARTH))
        .id();

    let cfg = VehicleBuilder::new()
        .with_state(body_state())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let vehicle_id = {
        let world = app.world_mut();
        let mut commands_queue = world.commands();
        let id = cfg.spawn_bevy(&mut commands_queue, &[earth]);
        world.flush();
        id
    };

    assert!(
        app.world()
            .entity(vehicle_id)
            .contains::<TranslationalStateC<jeod_sim::Earth>>(),
        "spawn_bevy must insert TranslationalStateC<jeod_sim::Earth> \
         (current Earth-only contract). If this assertion ever fires \
         because the helper was made planet-generic, update this test \
         to assert the new contract — but do not silently change the \
         inserted tag to `<P>` for a non-Earth body without rewiring \
         the public API."
    );
}
