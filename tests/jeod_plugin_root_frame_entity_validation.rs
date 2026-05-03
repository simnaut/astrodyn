//! `JeodPlugin` accepts caller-supplied [`RootFrameEntityR`] so a
//! mission can share the root frame entity with another subsystem
//! (or so a second `JeodPlugin::build` call doesn't leak the root
//! entity and re-parent future frame entities under a different
//! root). When a caller pre-installs the resource, the plugin must
//! validate that the referenced entity is fit for the role: still
//! alive, carrying [`InertialFrameMarker`], and carrying the three
//! frame components ([`FrameTransC`], [`FrameRotC`], [`FrameAngVelC`])
//! that source/body registration and frame-tree consumers depend on.
//!
//! These tests exercise the rejection paths that the plugin must
//! cover loudly:
//!
//! 1. `RootFrameEntityR` references an entity that doesn't exist
//!    (stale handle from a previous `App`, or the entity was
//!    despawned before `JeodPlugin::build` ran).
//! 2. `RootFrameEntityR` references an entity missing the
//!    `InertialFrameMarker`.
//! 3. `RootFrameEntityR` references an entity missing one of the
//!    required frame components.
//!
//! And the happy paths:
//!
//! 4. Not pre-installed: plugin spawns and inserts.
//! 5. Pre-installed and valid: plugin preserves the caller's entity.
//!
//! This mirrors `jeod_plugin_frame_tree_validation.rs` for the
//! arena-side `FrameTreeR` / `RootFrameIdR` pair.
//!
//! Issue #277 PR 1 round-5 review fixup (Copilot, threadId
//! `PRRT_kwDORtae6c5_K7a4`): without these checks, a pre-installed
//! `RootFrameEntityR` was silently overwritten, leaking the original
//! root entity and producing a frame hierarchy split across two
//! disconnected roots — exactly the kind of "wrong physics that
//! still runs" failure the "Fail Loudly" rule forbids.

use bevy::prelude::*;
use bevy_jeod::{
    FrameAngVelC, FrameRotC, FrameTransC, InertialFrameMarker, JeodPlugin, RootFrameEntityR,
};

#[test]
#[should_panic(expected = "references an entity that no longer exists in the world")]
fn jeod_plugin_rejects_dangling_root_frame_entity() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    // Spawn a placeholder, capture its id, then despawn it. The
    // resource now points at a tombstone — exactly the situation a
    // careless mission could create by spawning the root entity in
    // one `App` and re-using the id in another, or by despawning
    // before `JeodPlugin::build` runs.
    let stale = app.world_mut().spawn_empty().id();
    app.world_mut().entity_mut(stale).despawn();
    app.insert_resource(RootFrameEntityR(stale));
    app.add_plugins(JeodPlugin);
}

#[test]
#[should_panic(expected = "is missing `InertialFrameMarker`")]
fn jeod_plugin_rejects_root_frame_entity_without_inertial_marker() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    // All three frame components present, but the marker is missing
    // — the entity could be a body frame, a pfix frame, or just a
    // bare placeholder. Either way the plugin's assumption that the
    // root is inertial is broken.
    let entity = app
        .world_mut()
        .spawn((
            Name::new("not_actually_inertial"),
            FrameTransC::default(),
            FrameRotC::default(),
            FrameAngVelC::default(),
        ))
        .id();
    app.insert_resource(RootFrameEntityR(entity));
    app.add_plugins(JeodPlugin);
}

#[test]
#[should_panic(expected = "missing one or more of the required frame components")]
fn jeod_plugin_rejects_root_frame_entity_missing_frame_components() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    // Marker is present but `FrameRotC` / `FrameAngVelC` are not —
    // frame-tree consumers reading the root would either silently
    // read defaults (if Bevy treated the absence as default) or
    // panic deep in an unrelated query. Catch it at the boundary.
    let entity = app
        .world_mut()
        .spawn((
            Name::new("partial_root"),
            InertialFrameMarker,
            FrameTransC::default(),
        ))
        .id();
    app.insert_resource(RootFrameEntityR(entity));
    app.add_plugins(JeodPlugin);
}

#[test]
fn jeod_plugin_spawns_root_frame_entity_when_absent() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(JeodPlugin);
    let entity = app.world().resource::<RootFrameEntityR>().0;
    let world = app.world();
    assert!(
        world.get_entity(entity).is_ok(),
        "plugin must spawn a real entity"
    );
    let entity_ref = world.entity(entity);
    assert!(entity_ref.contains::<InertialFrameMarker>());
    assert!(entity_ref.contains::<FrameTransC>());
    assert!(entity_ref.contains::<FrameRotC>());
    assert!(entity_ref.contains::<FrameAngVelC>());
}

#[test]
fn jeod_plugin_preserves_valid_preseeded_root_frame_entity() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let preseeded = app
        .world_mut()
        .spawn((
            Name::new("mission_owned_root"),
            InertialFrameMarker,
            FrameTransC::default(),
            FrameRotC::default(),
            FrameAngVelC::default(),
        ))
        .id();
    app.insert_resource(RootFrameEntityR(preseeded));
    app.add_plugins(JeodPlugin);

    let after = app.world().resource::<RootFrameEntityR>().0;
    assert_eq!(
        after, preseeded,
        "plugin must not replace the caller's pre-installed root frame entity"
    );
}
