//! Identity flow through the ECS frame store (issue #664): every frame
//! entity carries a required [`FrameUidC`], the [`FrameUidIndexR`] maps
//! identity → frame entity (rejecting duplicates, RF.14), and the
//! pfix retire/reuse lifecycle keeps the index consistent.

use astrodyn::{FrameUid, PlanetFixed, PlanetInertial, RootInertial, EARTH};
use astrodyn_bevy::{
    AstrodynPlugin, FrameEntityC, FrameUidC, FrameUidIndexR, IntegrationDtR, PfixFrameEntityC,
    PlanetBundle, RootFrameEntityR, RotationModelC,
};
use bevy::prelude::*;

const DT: f64 = 1.0;

fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);
    app
}

#[test]
fn root_frame_carries_root_uid_and_is_indexed() {
    let mut app = test_app();
    app.update();
    let root = app.world().resource::<RootFrameEntityR>().0;
    let uid = app
        .world()
        .get::<FrameUidC>(root)
        .expect("root frame entity carries FrameUidC");
    assert_eq!(uid.0, FrameUid::of::<RootInertial>());
    assert_eq!(
        app.world()
            .resource::<FrameUidIndexR>()
            .get(&FrameUid::of::<RootInertial>()),
        Some(root),
        "the index resolves the root identity to the root frame entity"
    );
}

#[test]
fn source_and_pfix_frames_carry_planet_uids() {
    let mut app = test_app();
    let planet = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    app.update();

    let fe = app
        .world()
        .get::<FrameEntityC>(planet)
        .expect("registered")
        .0;
    assert_eq!(
        app.world().get::<FrameUidC>(fe).expect("stamped").0,
        FrameUid::of::<PlanetInertial<astrodyn::Earth>>(),
        "the source's carried identity lands on its frame entity"
    );
    let pfix_fe = app
        .world()
        .get::<PfixFrameEntityC>(planet)
        .expect("rotating source has a pfix frame")
        .0;
    assert_eq!(
        app.world().get::<FrameUidC>(pfix_fe).expect("stamped").0,
        FrameUid::of::<PlanetFixed<astrodyn::Earth>>(),
        "the pfix sibling derivation equals the typed mint"
    );
    let index = app.world().resource::<FrameUidIndexR>();
    assert_eq!(
        index.get(&FrameUid::of::<PlanetInertial<astrodyn::Earth>>()),
        Some(fe)
    );
    assert_eq!(
        index.get(&FrameUid::of::<PlanetFixed<astrodyn::Earth>>()),
        Some(pfix_fe)
    );
}

#[test]
fn body_frame_inherits_carried_identity() {
    let mut app = test_app();
    let uid = astrodyn::named_body_frame_uid("identity-flow-body");
    let body = app
        .world_mut()
        .spawn((
            FrameUidC(uid.clone()),
            astrodyn_bevy::TranslationalStateC::<astrodyn::Earth>::default(),
            astrodyn_bevy::DynamicsConfigC(astrodyn::DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
        ))
        .id();
    app.update();
    let fe = app.world().get::<FrameEntityC>(body).expect("registered").0;
    assert_eq!(
        app.world().get::<FrameUidC>(fe).expect("stamped").0,
        uid,
        "the body's mission-supplied identity lands on its frame entity"
    );
    assert_eq!(app.world().resource::<FrameUidIndexR>().get(&uid), Some(fe));
}

#[test]
// JEOD_INV: RF.14 — negative test: two frame entities carrying the same
// identity must be rejected at indexing, mirroring the arena's
// register_uid.
#[should_panic(expected = "duplicate frame identity")]
fn frame_uid_index_rejects_duplicate() {
    let mut app = test_app();
    // Two distinct gravity sources carrying the SAME inertial identity:
    // registration spawns two frame entities with one uid.
    for name in ["Earth", "Earth-impostor"] {
        app.world_mut().spawn((
            Name::new(name),
            FrameUidC(FrameUid::of::<PlanetInertial<astrodyn::Earth>>()),
            astrodyn_bevy::GravitySourceC(astrodyn::GravitySource {
                mu: EARTH.shape.mu,
                model: astrodyn::GravityModel::PointMass,
            }),
            astrodyn_bevy::SourceInertialPositionC::default(),
            astrodyn_bevy::TranslationalStateC::<astrodyn::Earth>::default(),
        ));
    }
    app.update();
}

#[test]
#[should_panic(expected = "has no FrameUidC")]
fn register_body_frames_panics_without_frame_uid() {
    let mut app = test_app();
    // A dynamic body with no carried identity: required, never minted by
    // accident (#662/#664).
    app.world_mut().spawn((
        astrodyn_bevy::TranslationalStateC::<astrodyn::Earth>::default(),
        astrodyn_bevy::DynamicsConfigC(astrodyn::DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        }),
    ));
    app.update();
}

#[test]
fn retired_pfix_leaves_index_and_reuse_reacquires_it() {
    let mut app = test_app();
    let planet = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    app.update();
    let pfix_uid = FrameUid::of::<PlanetFixed<astrodyn::Earth>>();
    let pfix_fe = app.world().get::<PfixFrameEntityC>(planet).expect("pfix").0;
    assert_eq!(
        app.world().resource::<FrameUidIndexR>().get(&pfix_uid),
        Some(pfix_fe)
    );

    // Toggle the rotation model to None: the pfix frame retires, its
    // identity leaves the store and the index.
    app.world_mut()
        .entity_mut(planet)
        .insert(RotationModelC(astrodyn::RotationModel::None));
    // One FixedUpdate tick so planet_fixed_rotation_system retires it
    // and the deindex system observes the removal.
    run_fixed_ticks(&mut app, 2);
    assert!(
        app.world().get::<PfixFrameEntityC>(planet).is_none(),
        "retired pfix handle removed from the source"
    );
    assert!(
        app.world()
            .resource::<FrameUidIndexR>()
            .get(&pfix_uid)
            .is_none(),
        "retired identity must leave the index so re-minting cannot collide"
    );

    // Toggle back: the retired entity is reused and re-acquires the
    // identity + index entry.
    app.world_mut()
        .entity_mut(planet)
        .insert(RotationModelC(astrodyn::RotationModel::EarthRNP));
    run_fixed_ticks(&mut app, 2);
    let reused = app
        .world()
        .get::<PfixFrameEntityC>(planet)
        .expect("reused")
        .0;
    assert_eq!(reused, pfix_fe, "toggle-back reuses the retired entity");
    assert_eq!(
        app.world().resource::<FrameUidIndexR>().get(&pfix_uid),
        Some(reused),
        "re-minted identity is re-indexed"
    );
}

/// Drive `n` FixedUpdate ticks directly (the parity-suite idiom): the
/// rotation system retires/reuses pfix frames inside FixedUpdate, and
/// the index/deindex systems run in the same schedule.
fn run_fixed_ticks(app: &mut App, n: usize) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(std::time::Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(bevy::app::FixedUpdate);
    }
}
