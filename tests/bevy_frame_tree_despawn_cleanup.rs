//! `FrameTreeR` despawn cleanup.
//!
//! The frame tree is append-only — `jeod_frames::FrameTree` exposes no
//! removal API because arena indices are stable handles other state may
//! hold. Without explicit cleanup, a despawned source's frame node would
//! sit in the arena indefinitely with the canonical name, eventually
//! shadowing a future re-spawn of the same name via
//! `FrameTree::find_by_name` and growing memory monotonically.
//!
//! These tests verify the per-component `Despawn` observers retire the
//! orphan nodes (rename + reset state) so the canonical name is no
//! longer findable and any stale `frame_origin` query returns identity.
//! See `src/systems.rs` "Frame-tree despawn cleanup" for the design.

// `FrameTreeR` is `#[deprecated]` for mission-code use. These
// despawn-observer tests validate the arena's per-node cleanup, which
// is the *internal* dual-write infrastructure that will be replaced
// wholesale once the resource is removed (the ECS frame-entity
// despawn observers stay). File-level `#![allow(deprecated)]` to keep
// the cleanup validation operating against the still-live arena until
// then.
#![allow(deprecated)]

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    BodyFrameIdC, BodyFrameMarker, DynamicsConfigC, FrameEntityC, FrameTreeR, GravityControlsC,
    InertialFrameMarker, JeodPlugin, MassPropertiesC, PfixFrameEntityC, PlanetBundle,
    PlanetFixedFrameMarker, RetiredPfixFrameIdC, RotationModelC, SourceFrameIdC,
    SourcePfixFrameIdC, TranslationalStateC,
};
use glam::DVec3;
use jeod_sim::{
    DynamicsConfig, GravityControls, MassProperties, RotationModel, TranslationalState, EARTH,
};

const DT: f64 = 60.0;

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);
    app
}

#[test]
fn source_despawn_retires_inertial_and_pfix_nodes() {
    let mut app = build_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    app.update();

    {
        let tree = app.world().resource::<FrameTreeR>();
        assert!(
            tree.0.find_by_name("Earth.inertial").is_some(),
            "Earth.inertial should exist before despawn"
        );
        assert!(
            tree.0.find_by_name("Earth.pfix").is_some(),
            "Earth.pfix should exist before despawn (PlanetBundle::point_mass adds PlanetFixedRotationC)"
        );
    }

    let inertial_fid = app.world().get::<SourceFrameIdC>(earth).unwrap().0;
    let pfix_fid = app.world().get::<SourcePfixFrameIdC>(earth).unwrap().0;

    app.world_mut().entity_mut(earth).despawn();
    app.update();

    let tree = app.world().resource::<FrameTreeR>();
    assert!(
        tree.0.find_by_name("Earth.inertial").is_none(),
        "Earth.inertial must not be findable after despawn — it would shadow a future re-spawn"
    );
    assert!(
        tree.0.find_by_name("Earth.pfix").is_none(),
        "Earth.pfix must not be findable after despawn"
    );
    assert!(
        tree.0.get(inertial_fid).name.ends_with(".despawned"),
        "inertial node must be renamed with '.despawned' sentinel, got {:?}",
        tree.0.get(inertial_fid).name
    );
    assert!(
        tree.0.get(pfix_fid).name.ends_with(".despawned"),
        "pfix node must be renamed with '.despawned' sentinel, got {:?}",
        tree.0.get(pfix_fid).name
    );
    assert_eq!(
        tree.0.get(inertial_fid).state.trans.position,
        DVec3::ZERO,
        "retired inertial node state must be reset so stale frame_origin queries return identity"
    );
    assert_eq!(
        tree.0.get(pfix_fid).state.rot.ang_vel_this,
        DVec3::ZERO,
        "retired pfix node state must be reset"
    );
}

#[test]
fn source_respawn_after_despawn_does_not_shadow() {
    let mut app = build_app();
    let earth1 = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    app.update();
    let original_inertial_fid = app.world().get::<SourceFrameIdC>(earth1).unwrap().0;

    app.world_mut().entity_mut(earth1).despawn();
    app.update();

    let earth2 = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    app.update();

    let tree = app.world().resource::<FrameTreeR>();
    let new_inertial_fid = app.world().get::<SourceFrameIdC>(earth2).unwrap().0;
    assert_ne!(
        new_inertial_fid, original_inertial_fid,
        "re-spawn must allocate a fresh frame node, not reuse the retired slot"
    );
    let found = tree
        .0
        .find_by_name("Earth.inertial")
        .expect("Earth.inertial must be findable after re-spawn");
    assert_eq!(
        found, new_inertial_fid,
        "find_by_name('Earth.inertial') must resolve to the new node, not the retired one"
    );
}

#[test]
fn body_despawn_retires_body_node() {
    let mut app = build_app();
    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    let body = app
        .world_mut()
        .spawn((
            Name::new("vehicle"),
            TranslationalStateC::from_untyped(TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7.5e3, 0.0),
            }),
            MassPropertiesC::from(MassProperties::new(1000.0)),
            DynamicsConfigC(DynamicsConfig::default()),
            GravityControlsC(GravityControls::<Entity> {
                controls: Vec::new(),
            }),
        ))
        .id();
    app.update();

    let body_fid = app.world().get::<BodyFrameIdC>(body).unwrap().0;
    {
        let tree = app.world().resource::<FrameTreeR>();
        assert!(
            tree.0.find_by_name("vehicle.body").is_some(),
            "vehicle.body should exist before despawn"
        );
    }

    app.world_mut().entity_mut(body).despawn();
    app.update();

    let tree = app.world().resource::<FrameTreeR>();
    assert!(
        tree.0.find_by_name("vehicle.body").is_none(),
        "vehicle.body must not be findable after despawn"
    );
    assert!(
        tree.0.get(body_fid).name.ends_with(".despawned"),
        "body node must be renamed with '.despawned' sentinel"
    );
}

#[test]
fn retired_pfix_node_despawn_retires_orphan() {
    let mut app = build_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    app.update();

    app.world_mut()
        .entity_mut(earth)
        .insert(RotationModelC(RotationModel::None));
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    let retired_fid = app
        .world()
        .get::<RetiredPfixFrameIdC>(earth)
        .expect("toggle to RotationModel::None should insert RetiredPfixFrameIdC")
        .0;
    {
        let tree = app.world().resource::<FrameTreeR>();
        assert!(
            tree.0.get(retired_fid).name.ends_with(".retired"),
            "retired pfix node should carry the '.retired' sentinel before despawn"
        );
    }

    app.world_mut().entity_mut(earth).despawn();
    app.update();

    let tree = app.world().resource::<FrameTreeR>();
    assert!(
        tree.0.get(retired_fid).name.ends_with(".despawned"),
        "retired pfix node must be re-renamed to '.despawned' when its owning entity despawns, got {:?}",
        tree.0.get(retired_fid).name
    );
}

// ── ECS frame-entity cleanup ──
//
// `register_source_frames_system` and `register_body_frames_system`
// dual-write a frame *entity* alongside the arena `FrameId`. Without
// a parallel observer on `FrameEntityC` / `PfixFrameEntityC`,
// despawning the source / body entity left the dual-write frame
// entities alive forever, growing the world's entity count over
// time and shadowing future re-spawns of the same `Name`. The
// observers tested below close that gap.

#[test]
fn source_despawn_despawns_inertial_and_pfix_frame_entities() {
    let mut app = build_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    app.update();

    let inertial_frame_entity = app.world().get::<FrameEntityC>(earth).unwrap().0;
    let pfix_frame_entity = app.world().get::<PfixFrameEntityC>(earth).unwrap().0;
    assert!(
        app.world().get_entity(inertial_frame_entity).is_ok(),
        "source inertial frame entity should be alive before despawn"
    );
    assert!(
        app.world().get_entity(pfix_frame_entity).is_ok(),
        "source pfix frame entity should be alive before despawn"
    );

    app.world_mut().entity_mut(earth).despawn();
    app.update();

    assert!(
        app.world().get_entity(inertial_frame_entity).is_err(),
        "source inertial frame entity must be despawned when the source despawns"
    );
    assert!(
        app.world().get_entity(pfix_frame_entity).is_err(),
        "source pfix frame entity must be despawned when the source despawns (recursively or via the pfix observer)"
    );

    // No `InertialFrameMarker` (besides the root) and no
    // `PlanetFixedFrameMarker` should remain after the source
    // despawn — the only inertial-tagged entity left is `root.frame`.
    let mut q_inertial = app.world_mut().query::<&InertialFrameMarker>();
    assert_eq!(
        q_inertial.iter(app.world()).count(),
        1,
        "only the root frame should remain tagged InertialFrameMarker after source despawn"
    );
    let mut q_pfix = app.world_mut().query::<&PlanetFixedFrameMarker>();
    assert_eq!(
        q_pfix.iter(app.world()).count(),
        0,
        "no PlanetFixedFrameMarker entities should remain after source despawn"
    );
}

#[test]
fn body_despawn_despawns_body_frame_entity() {
    let mut app = build_app();
    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    let body = app
        .world_mut()
        .spawn((
            Name::new("vehicle"),
            TranslationalStateC::from_untyped(TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7.5e3, 0.0),
            }),
            MassPropertiesC::from(MassProperties::new(1000.0)),
            DynamicsConfigC(DynamicsConfig::default()),
            GravityControlsC(GravityControls::<Entity> {
                controls: Vec::new(),
            }),
        ))
        .id();
    app.update();

    let body_frame_entity = app.world().get::<FrameEntityC>(body).unwrap().0;
    assert!(
        app.world().get_entity(body_frame_entity).is_ok(),
        "body frame entity should be alive before despawn"
    );

    app.world_mut().entity_mut(body).despawn();
    app.update();

    assert!(
        app.world().get_entity(body_frame_entity).is_err(),
        "body frame entity must be despawned when the body despawns"
    );
    let mut q_body = app.world_mut().query::<&BodyFrameMarker>();
    assert_eq!(
        q_body.iter(app.world()).count(),
        0,
        "no BodyFrameMarker entities should remain after body despawn"
    );
}

#[test]
fn rotation_none_then_source_despawn_does_not_leak_retired_pfix_entity() {
    // Combination of the two retirement tracks: toggle to
    // `RotationModel::None` (which retires the live pfix entity into
    // `RetiredPfixFrameEntityC` but keeps it alive) and then despawn
    // the source. Both the live source frame entity and the retired
    // pfix entity must be gone afterward.
    let mut app = build_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    app.update();

    let inertial_frame_entity = app.world().get::<FrameEntityC>(earth).unwrap().0;
    let original_pfix_entity = app.world().get::<PfixFrameEntityC>(earth).unwrap().0;

    app.world_mut()
        .entity_mut(earth)
        .insert(RotationModelC(RotationModel::None));
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    // The pfix entity is now stashed in `RetiredPfixFrameEntityC`
    // and `PfixFrameEntityC` has been removed.
    assert!(
        app.world().get::<PfixFrameEntityC>(earth).is_none(),
        "PfixFrameEntityC should be removed on toggle to RotationModel::None"
    );
    assert!(
        app.world().get_entity(original_pfix_entity).is_ok(),
        "retired pfix entity should still be alive (stashed for reuse)"
    );

    app.world_mut().entity_mut(earth).despawn();
    app.update();

    assert!(
        app.world().get_entity(inertial_frame_entity).is_err(),
        "source frame entity must be despawned when the source despawns"
    );
    assert!(
        app.world().get_entity(original_pfix_entity).is_err(),
        "retired pfix entity must be despawned when its owning source despawns"
    );
}
