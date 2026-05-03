//! `MassPointRef` back-pointer wiring (PR #283 review thread
//! `PRRT_kwDORtae6c5_KiLK`).
//!
//! `register_body_frames_system` auto-inserts the frame-side
//! [`MassPointRef`](bevy_jeod::MassPointRef) back-pointer for every
//! body entity that participates in the mass tree (carries
//! [`MassPropertiesC`]). In the current Bevy adapter the body, mass,
//! and frame are colocated on a single ECS entity, so the
//! back-pointer resolves to `MassPointRef(self_entity)` — mirroring
//! JEOD's `BodyRefFrame::mass_point` connection from a kinematic
//! frame to its mass-tree origin (`models/dynamics/dyn_body/include
//! /body_ref_frame.hh`).
//!
//! The component is **omitted** for kinematic-only bodies (no
//! `MassPropertiesC`), matching the "absent for kinematic-only
//! attaches" contract documented on `MassPointRef`.

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    BodyFrameIdC, DynamicsConfigC, JeodPlugin, MassPointRef, MassPropertiesC, RotationalStateC,
    TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_sim::{DynamicsConfig, MassProperties, RotationalState, TranslationalState};

const DT: f64 = 60.0;

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
    app
}

fn trans_state() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7.5e3, 0.0),
    }
}

fn mass() -> MassProperties {
    MassProperties::with_inertia(
        100.0,
        DMat3::from_diagonal(DVec3::new(50.0, 50.0, 50.0)),
        DVec3::ZERO,
    )
}

fn dynamics() -> DynamicsConfig {
    DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    }
}

#[test]
fn mass_point_ref_inserted_on_body_with_mass_properties() {
    let mut app = build_app();
    let body = app
        .world_mut()
        .spawn((
            Name::new("vehicle"),
            TranslationalStateC::from(trans_state()),
            RotationalStateC::from(RotationalState::default()),
            MassPropertiesC::from(mass()),
            DynamicsConfigC(dynamics()),
        ))
        .id();

    // Run startup so register_body_frames_system fires.
    app.world_mut().run_schedule(Startup);
    // Drive one fixed-update tick so any Commands deferred inside
    // Startup are applied.
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    // Body frame registered.
    assert!(
        app.world().get::<BodyFrameIdC>(body).is_some(),
        "BodyFrameIdC missing — register_body_frames_system did not run"
    );
    // MassPointRef back-pointer wired and points at the body itself
    // (frame entity == mass entity == body entity in current Bevy
    // adapter).
    let mass_point_ref = app
        .world()
        .get::<MassPointRef>(body)
        .copied()
        .expect("MassPointRef must be inserted on a body that has MassPropertiesC");
    assert_eq!(
        mass_point_ref.0, body,
        "MassPointRef should point at the body entity itself"
    );
}

#[test]
fn mass_point_ref_inserted_when_mass_acquired_after_registration() {
    // PR #283 review thread `PRRT_kwDORtae6c5_K7qF`:
    // `register_body_frames_system` filters by `Without<BodyFrameIdC>`
    // and only sees each body once. A body that starts kinematic-only
    // (no `MassPropertiesC`) and later acquires it must still get the
    // `MassPointRef` back-pointer wired up — otherwise the "frame
    // entity participates in the mass tree" invariant would only hold
    // at spawn time.
    let mut app = build_app();
    let body = app
        .world_mut()
        .spawn((
            Name::new("late_mass"),
            TranslationalStateC::from(trans_state()),
            RotationalStateC::from(RotationalState::default()),
            // Note: no MassPropertiesC at spawn — kinematic-only.
            DynamicsConfigC(dynamics()),
        ))
        .id();

    app.world_mut().run_schedule(Startup);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    // After the first tick: body is registered (has BodyFrameIdC),
    // but no back-pointer yet because it had no mass.
    assert!(
        app.world().get::<BodyFrameIdC>(body).is_some(),
        "BodyFrameIdC must be inserted on first tick"
    );
    assert!(
        app.world().get::<MassPointRef>(body).is_none(),
        "MassPointRef should be absent before mass is acquired"
    );

    // Mission code attaches `MassPropertiesC` mid-sim (e.g. the body
    // was a kinematic placeholder that just received its real mass
    // properties).
    app.world_mut()
        .entity_mut(body)
        .insert(MassPropertiesC::from(mass()));

    // Drive another tick so `sync_body_mass_point_ref_system` fires.
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    let mass_point_ref =
        app.world().get::<MassPointRef>(body).copied().expect(
            "MassPointRef must be inserted after MassPropertiesC is added post-registration",
        );
    assert_eq!(
        mass_point_ref.0, body,
        "MassPointRef should point at the body entity itself"
    );
}

#[test]
fn mass_point_ref_removed_when_mass_lost_after_registration() {
    // PR #283 review thread `PRRT_kwDORtae6c5_K7qF`: the inverse of
    // the previous test. A body that initially has `MassPropertiesC`
    // (and hence `MassPointRef`) but later has its mass component
    // removed must shed the now-stale back-pointer — keeping it would
    // lie about whether the frame still participates in the mass tree.
    let mut app = build_app();
    let body = app
        .world_mut()
        .spawn((
            Name::new("loses_mass"),
            TranslationalStateC::from(trans_state()),
            RotationalStateC::from(RotationalState::default()),
            MassPropertiesC::from(mass()),
            DynamicsConfigC(dynamics()),
        ))
        .id();

    app.world_mut().run_schedule(Startup);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    // First-pass registration wired up MassPointRef.
    assert!(
        app.world().get::<MassPointRef>(body).is_some(),
        "MassPointRef should be wired at first sight"
    );

    // Mission code removes `MassPropertiesC` (e.g. body retired to
    // kinematic-only behaviour for the rest of the sim).
    app.world_mut().entity_mut(body).remove::<MassPropertiesC>();

    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    assert!(
        app.world().get::<MassPointRef>(body).is_none(),
        "MassPointRef should be removed once MassPropertiesC is gone — \
         keeping it would lie about mass-tree participation"
    );
}

#[test]
fn mass_point_ref_omitted_for_kinematic_only_body() {
    // A body without MassPropertiesC (kinematic-only — sensor mount,
    // station-keeping vehicle attached via attach_to_frame) must
    // *not* receive a MassPointRef. The "absent for kinematic-only
    // attaches" contract is documented on MassPointRef itself.
    let mut app = build_app();
    let body = app
        .world_mut()
        .spawn((
            Name::new("kinematic"),
            TranslationalStateC::from(trans_state()),
            RotationalStateC::from(RotationalState::default()),
            DynamicsConfigC(dynamics()),
        ))
        .id();

    app.world_mut().run_schedule(Startup);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    assert!(
        app.world().get::<BodyFrameIdC>(body).is_some(),
        "BodyFrameIdC should still be inserted on a kinematic-only body"
    );
    assert!(
        app.world().get::<MassPointRef>(body).is_none(),
        "MassPointRef should be absent on a body without MassPropertiesC"
    );
}
