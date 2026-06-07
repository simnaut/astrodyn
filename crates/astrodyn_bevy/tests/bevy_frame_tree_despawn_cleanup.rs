//! Frame-entity despawn cleanup.
//!
//! Frame entities are owned by the source / body / pfix entity that
//! references them via `FrameEntityC` / `PfixFrameEntityC` /
//! `RetiredPfixFrameEntityC`. When the owner despawns, the referenced
//! frame entity must be despawned alongside so the world's entity
//! count stays bounded and future re-spawns of the same `Name` aren't
//! shadowed by an orphan. These tests exercise the per-component
//! `Despawn` observers that close that gap. See the module-level
//! comment in `src/systems.rs` ("Frame-tree despawn cleanup") for
//! the design.

use std::time::Duration;

use astrodyn::{
    DynamicsConfig, GravityControls, MassProperties, RotationModel, TranslationalState, EARTH,
};
use astrodyn_bevy::{
    AstrodynPlugin, BodyFrameMarker, DynamicsConfigC, FrameEntityC, GravityControlsC,
    InertialFrameMarker, IntegrationDtR, MassPropertiesC, PfixFrameEntityC, PlanetBundle,
    PlanetFixedFrameMarker, RotationModelC, TranslationalStateC,
};
use bevy::prelude::*;
use glam::DVec3;

const DT: f64 = 60.0;

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);
    app
}

#[test]
fn source_despawn_despawns_inertial_and_pfix_frame_entities() {
    let mut app = build_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
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
fn source_respawn_after_despawn_does_not_shadow() {
    let mut app = build_app();
    let earth1 = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    app.update();
    let original_inertial_frame_entity = app.world().get::<FrameEntityC>(earth1).unwrap().0;

    app.world_mut().entity_mut(earth1).despawn();
    app.update();

    let earth2 = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    app.update();

    let new_inertial_frame_entity = app.world().get::<FrameEntityC>(earth2).unwrap().0;
    assert_ne!(
        new_inertial_frame_entity, original_inertial_frame_entity,
        "re-spawn must allocate a fresh frame entity, not reuse the despawned slot"
    );
    // The re-spawned source's frame entity must be alive.
    assert!(
        app.world().get_entity(new_inertial_frame_entity).is_ok(),
        "re-spawned source frame entity must be alive"
    );
}

#[test]
fn body_despawn_despawns_body_frame_entity() {
    let mut app = build_app();
    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    let body = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(
                "bevy-frame-tree-despawn-cleanup-b1",
            )),
            Name::new("vehicle"),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: DVec3::new(7e6, 0.0, 0.0),
                velocity: DVec3::new(0.0, 7.5e3, 0.0),
            }),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(1000.0)),
            )),
            DynamicsConfigC(DynamicsConfig::default()),
            GravityControlsC(GravityControls {
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
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
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
