//! `FrameTreeR` despawn cleanup (PR #260 reviewer-flagged gap).
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

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    BodyFrameIdC, DynamicsConfigC, FrameTreeR, GravityControlsC, JeodPlugin, MassPropertiesC,
    PlanetBundle, RetiredPfixFrameIdC, RotationModelC, SourceFrameIdC, SourcePfixFrameIdC,
    TranslationalStateC,
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
