// JEOD_INV: TS.01 — `<SelfRef>` is a runtime-resolved storage-boundary
// wildcard; this test reaches the canonical Bevy adapter
// `AttachEvent<SelfRef, SelfRef>` Message slot directly. See
// `docs/JEOD_invariants.md` row TS.01 and the lint at
// `tests/self_ref_self_planet_discipline.rs`.
//! Fail-loudly contract for [`astrodyn_bevy::systems::staging_system`]:
//! a pending `AttachEvent` / `DetachEvent` observed without
//! `MassTreeR` registered as a Bevy resource must panic with a
//! diagnostic that names both fix paths.
//!
//! Without this, an `AttachEvent` issued before `MassTreeR` is
//! inserted would be silently drained — the staging system has no
//! arena to mutate, the targeted body propagates unattached, and the
//! mass topology is wrong with no surface signal. That is the
//! "wrong physics that still runs" failure mode the *Fail Loudly*
//! rule (CLAUDE.md) forbids.
//!
//! See `docs/JEOD_invariants.md` row `MA.24` and the staging-system
//! source comment in
//! `crates/astrodyn_bevy/src/systems/integration.rs`.

use std::time::Duration;

use astrodyn_bevy::{AstrodynPlugin, AttachEvent, DetachEvent};
use bevy::prelude::*;
use glam::DVec3;

const DT: f64 = 1.0;

fn run_one_fixed_tick(app: &mut App) {
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

fn build_minimal_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);
    app
}

#[test]
#[should_panic(expected = "AttachEvent received but `MassTreeR` is not registered")]
fn staging_system_panics_when_attach_event_arrives_without_mass_tree() {
    let mut app = build_minimal_app();
    // Two bare placeholder entities are enough to construct an
    // `AttachEvent`. The staging system's `MassTreeR` precondition
    // fires before any per-entity validation, so the participants do
    // not need any body components for this contract test.
    let parent = app.world_mut().spawn_empty().id();
    let child = app.world_mut().spawn_empty().id();
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<
            AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>,
        >>()
        .write(AttachEvent {
            child,
            parent,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    run_one_fixed_tick(&mut app);
}

#[test]
#[should_panic(expected = "DetachEvent received but `MassTreeR` is not registered")]
fn staging_system_panics_when_detach_event_arrives_without_mass_tree() {
    let mut app = build_minimal_app();
    let child = app.world_mut().spawn_empty().id();
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent { child });
    run_one_fixed_tick(&mut app);
}

#[test]
fn staging_system_no_op_when_mass_tree_absent_and_no_events() {
    // The fail-loudly contract is gated on the *event being pending*:
    // a step with no staging events must remain a no-op even when
    // `MassTreeR` is absent. Otherwise every Bevy app would have to
    // pre-install `MassTreeR` for empty-staging missions.
    let mut app = build_minimal_app();
    run_one_fixed_tick(&mut app);
    run_one_fixed_tick(&mut app);
}
