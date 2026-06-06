//! Frame-entity sync systems must panic with a fail-loud diagnostic
//! — not silently no-op — when a `FrameEntityC` /
//! `PfixFrameEntityC` / `RetiredPfixFrameEntityC` handle is stale
//! (the referenced frame entity has been despawned or stripped of
//! its required `FrameTransC` / `FrameRotC` / `FrameAngVelC`
//! components).
//!
//! Per the project's "Fail Loudly" rule, silently swallowing
//! `Query::get_mut` failures here would let the source/body's ECS
//! components drift out of sync with the frame entity's stored
//! state while `RelativeFrameState` consumers read the stale value
//! — producing wrong physics with no diagnostic. These
//! `#[should_panic]` tests pin the panic messages so future
//! regressions are caught.
//!
//! See `src/systems.rs::sync_source_to_frame_system`,
//! `sync_body_to_frame_system`, `register_pfix_frames_system`, and
//! `planet_fixed_rotation_system`.

use std::time::Duration;

use astrodyn::{
    GravityControl, GravityGradient, JeodQuat, MassProperties, RootInertial, RotationModel,
    RotationalState, TranslationalStateTyped, Vec3Ext, VehicleBuilder, EARTH,
};
use astrodyn_bevy::{
    AstrodynPlugin, FrameEntityC, FrameRotC, FrameTransC, IntegrationDtR, PfixFrameEntityC,
    PlanetBundle, RotationModelC, VehicleConfigBevyExt,
};
use bevy::prelude::*;
use glam::DVec3;

const DT: f64 = 60.0;

/// Advance Fixed time by one DT and run the FixedUpdate schedule.
/// `app.update()` only runs `Update`, but `AstrodynPlugin`'s dual-write
/// systems live in `FixedUpdate`, so the bare update never exercises
/// them.
fn step_fixed(app: &mut App) {
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);
    app
}

fn initial_trans() -> TranslationalStateTyped<RootInertial> {
    TranslationalStateTyped::<RootInertial> {
        position: DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<RootInertial>(),
        velocity: DVec3::new(0.0, 7500.0, 0.0).m_per_s_at::<RootInertial>(),
    }
}

fn initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    }
}

fn vehicle_mass() -> MassProperties {
    MassProperties::with_inertia(
        1_000.0,
        glam::DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0)),
        DVec3::ZERO,
    )
}

/// Spawn an Earth source + a 6-DoF body via `VehicleConfig::spawn_bevy`,
/// then return both entities. After one update the dual-write
/// components are wired and the body's frame entity exists.
fn spawn_earth_and_body(app: &mut App) -> (Entity, Entity) {
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();

    let cfg = VehicleBuilder::new()
        .vehicle_named("frame-entity-dual-write-fail-loud-0")
        .with_translational(initial_trans())
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        ))
        .build();

    let body = {
        let mut commands_queue = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth])
    };
    app.world_mut().flush();
    // Update tick lets the Startup-equivalent registration systems
    // flush their Commands so FrameEntityC etc. are attached. Then
    // a FixedUpdate tick exercises the dual-write path under
    // intact wiring so the test gates on a stable starting state.
    app.update();
    step_fixed(app);

    (earth, body)
}

/// Smoke check: with an intact dual-write wiring, no panic occurs.
/// Establishes the baseline against which the `#[should_panic]`
/// tests below isolate the stale-handle path.
#[test]
fn intact_dual_write_does_not_panic() {
    let mut app = build_app();
    let (_earth, body) = spawn_earth_and_body(&mut app);

    let frame_entity = app
        .world()
        .get::<FrameEntityC>(body)
        .expect("body must carry FrameEntityC after register_body_frames_system runs")
        .0;
    assert!(
        app.world().get::<FrameTransC>(frame_entity).is_some(),
        "body's frame entity must carry FrameTransC after registration"
    );

    // Step a few FixedUpdate ticks — the dual-write systems should
    // run without panicking.
    for _ in 0..3 {
        step_fixed(&mut app);
    }
}

/// Stale `FrameEntityC` on a body: strip `FrameTransC` off the
/// body's frame entity, leaving the body's `FrameEntityC` pointing
/// at a still-alive entity that no longer satisfies the dual-write
/// contract. The next tick's `sync_body_to_frame_system` must
/// panic with a diagnostic naming the broken assumption.
#[test]
#[should_panic(expected = "sync_body_to_frame_system: body has FrameEntityC")]
fn sync_body_to_frame_panics_on_stale_frame_entity() {
    let mut app = build_app();
    let (_earth, body) = spawn_earth_and_body(&mut app);

    let frame_entity = app
        .world()
        .get::<FrameEntityC>(body)
        .expect("body must carry FrameEntityC after register_body_frames_system runs")
        .0;
    // Strip FrameTransC off the frame entity to simulate either a
    // partial despawn or external component removal — either way
    // the handle is stale w.r.t. the dual-write contract. We do
    // NOT despawn the entity itself because the recursive child
    // despawn observers would also remove FrameEntityC from the
    // body, hiding the bug. Removing just FrameTransC is the
    // tightest reproduction of "handle present but write target
    // missing required component."
    app.world_mut()
        .entity_mut(frame_entity)
        .remove::<FrameTransC>();

    // Next FixedUpdate tick must panic in sync_body_to_frame_system.
    step_fixed(&mut app);
}

/// Stale `FrameEntityC` on a source: same shape as above but for
/// `sync_source_to_frame_system`. Strip `FrameTransC` off the
/// source's frame entity and assert the next tick panics.
#[test]
#[should_panic(expected = "sync_source_to_frame_system: source has FrameEntityC")]
fn sync_source_to_frame_panics_on_stale_frame_entity() {
    let mut app = build_app();
    // We need a body too so that the system actually iterates over
    // a source whose FrameEntityC is stale; the source-side dual-
    // write fires regardless, but spawning a body keeps the test
    // shape symmetric with the body-side test above.
    let (earth, _body) = spawn_earth_and_body(&mut app);

    let frame_entity = app
        .world()
        .get::<FrameEntityC>(earth)
        .expect("source must carry FrameEntityC after PlanetBundle spawn")
        .0;
    // Same surgical strip as in the body test — leaves
    // `FrameEntityC` pointing at a still-alive entity that no
    // longer carries `FrameTransC`, which is exactly the silent
    // `Query::get_mut` failure mode round-6 review flagged.
    app.world_mut()
        .entity_mut(frame_entity)
        .remove::<FrameTransC>();

    // Next FixedUpdate tick must panic in sync_source_to_frame_system.
    step_fixed(&mut app);
}

/// Stale `PfixFrameEntityC` on a rotating-Earth source: strip
/// `FrameRotC` off the pfix frame entity. Next tick's
/// `planet_fixed_rotation_system` must panic when it tries to
/// write the new rotation matrix.
#[test]
#[should_panic(expected = "planet_fixed_rotation_system")]
fn planet_fixed_rotation_panics_on_stale_pfix_frame_entity() {
    let mut app = build_app();
    // PlanetBundle::point_mass attaches PlanetFixedRotationC + the
    // default RotationModel::EarthRNP, so register_pfix_frames_system
    // will spawn a pfix frame entity for this source.
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    // Force EarthRNP explicitly so the rotated branch (not the None
    // branch) is exercised on the next tick.
    app.world_mut()
        .entity_mut(earth)
        .insert(RotationModelC(RotationModel::EarthRNP));
    app.update();
    // First FixedUpdate tick lets register_pfix_frames_system attach
    // PfixFrameEntityC and spawn the pfix frame entity.
    step_fixed(&mut app);

    let pfix_frame_entity = app
        .world()
        .get::<PfixFrameEntityC>(earth)
        .expect("source must carry PfixFrameEntityC after register_pfix_frames_system runs")
        .0;
    // Strip FrameRotC off the pfix frame entity. The next tick of
    // planet_fixed_rotation_system must panic when writing the
    // matrix; we do NOT despawn the entity itself because that
    // would tear down the source dual-write chain via the despawn
    // observers, masking the targeted failure mode.
    app.world_mut()
        .entity_mut(pfix_frame_entity)
        .remove::<FrameRotC>();

    // Next FixedUpdate tick must panic in planet_fixed_rotation_system.
    step_fixed(&mut app);
}
