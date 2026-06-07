//! Focused unit coverage for [`BevySimContext::attach_to_frame`].
//!
//! The integration-test parity wrapper that exercises this surface
//! end-to-end (`bevy_parity_ref_attach.rs::bevy_parity_ref_attach_matrix`)
//! is currently `#[ignore]`'d behind a sub-ULP Bevy-side schedule
//! investigation, so without a dedicated unit test the adapter method
//! is structurally unreachable from any non-ignored test. This file
//! covers the routing contract directly:
//!
//! - `SourceFrameKind::Inertial` resolves the parent frame to the
//!   source's [`FrameEntityC`] (the inertial child of the root frame).
//! - `SourceFrameKind::Pfix` resolves to the source's
//!   [`PfixFrameEntityC`] (the rotating child of the inertial frame).
//! - In both cases a [`FrameAttachEvent`] lands on the message bus and
//!   the next `FixedUpdate` tick processes it via `frame_attach_system`,
//!   inserting [`FrameAttachedC { parent_frame, .. }`] on the body with
//!   the resolved frame entity recorded.
//!
//! The bit-identity argument lives in the integration parity wrapper;
//! this test only proves the Bevy-side plumbing (kind → entity resolution,
//! message dispatch, schedule drain) is wired correctly, so the adapter
//! cannot silently regress while the trajectory wrapper stays ignored.

// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers.

use std::time::Duration;

use astrodyn::{
    DynamicsConfig, GravityControl, GravityControls, GravityGradient, MassPropertiesTyped,
    RotationalStateTyped, SelfRef, TranslationalState,
};
use astrodyn_bevy::{
    AstrodynPlugin, DynamicsConfigC, FrameAttachedC, FrameEntityC, GravityAccelerationC,
    GravityControlsC, GravitySourceC, IntegrationDtR, MassPropertiesC, PfixFrameEntityC,
    PlanetFixedRotationC, RotationalStateC, SourceInertialPositionC, TranslationalStateC,
};
use astrodyn_verif_jeod::verification::{SimContext, SourceFrameKind};
use astrodyn_verif_parity::BevySimContext;
use bevy::prelude::*;
use glam::{DMat3, DVec3};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

const DT: f64 = 1.0;

fn earth_source() -> astrodyn::GravitySource {
    astrodyn::GravitySource {
        mu: astrodyn::EARTH.shape.mu,
        model: astrodyn::GravityModel::PointMass,
    }
}

/// Build a minimal Bevy app with one rotating-Earth source and one body,
/// run `Startup` to register the source's inertial + pfix frame entities,
/// and return `(app, source_entity, body_entity)`.
///
/// The source carries `PlanetFixedRotationC` (the indicator that gates
/// pfix-frame registration in `register_source_frames_system`) and
/// omits `RotationModelC` — the registration path defaults that to
/// `RotationModel::EarthRNP`, which is non-`None` and therefore
/// triggers `PfixFrameEntityC` insertion. The actual per-tick rotation
/// behavior doesn't matter to this test (we only compare entity IDs
/// recorded on `FrameAttachedC`), but a non-`None` model is required
/// for the pfix entity to exist at all.
fn build_test_app() -> (App, Entity, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);

    let source = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::FrameUid::of::<
                astrodyn::PlanetInertial<astrodyn::Earth>,
            >()),
            Name::new("Earth"),
            GravitySourceC(earth_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
            PlanetFixedRotationC::<astrodyn::Earth>(astrodyn::FrameTransform::from_matrix(
                DMat3::IDENTITY,
            )),
        ))
        .id();

    // Spawn a body with default-zero state. `propagate_frame_attached_state_system`
    // overwrites the state from the parent frame + captured offset each
    // tick, so the starting value is irrelevant — what matters is that
    // the body carries `TranslationalStateC<Earth>` (the
    // `frame_attach_system` reject path filters on that) and is not
    // tagged as a mass-tree child (no `MassChildOf`).
    let body = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(&format!(
                "bevy-context-attach-to-frame-b1-{}",
                NEXT_BODY_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ))),
            Name::new("body"),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState::default()),
            RotationalStateC::from(RotationalStateTyped::<SelfRef>::default()),
            MassPropertiesC::from(MassPropertiesTyped::<SelfRef>::new(Mass::new::<kilogram>(
                1_000.0,
            ))),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Skip,
                )],
            }),
            GravityAccelerationC::default(),
        ))
        .id();

    // Run Startup so `register_source_frames_system` registers
    // `FrameEntityC` + `PfixFrameEntityC` on the source. `MinimalPlugins`
    // does not auto-run Startup; without this step the
    // `BevySimContext::attach_to_frame` calls below would resolve to
    // missing components.
    app.world_mut().run_schedule(Startup);

    (app, source, body)
}

/// Advance one `FixedUpdate` tick. `frame_attach_system` drains
/// `FrameAttachEvent` messages at the top of the schedule (pinned
/// between `EphemerisUpdate` and `Environment`), so this is the minimal
/// progression that converts a queued event into a `FrameAttachedC`
/// insertion.
fn step_one_tick(app: &mut App) {
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

/// `BevySimContext::attach_to_frame(SourceFrameKind::Inertial)` resolves
/// the parent frame entity to the source's [`FrameEntityC`] and the
/// next tick produces a `FrameAttachedC` on the body pointing at that
/// entity.
#[test]
fn attach_to_frame_inertial_routes_to_source_frame_entity() {
    let (mut app, source, body) = build_test_app();

    let expected_parent = app
        .world()
        .get::<FrameEntityC>(source)
        .expect("Startup must have registered FrameEntityC on the source")
        .0;

    // Use a non-identity rotation and a non-zero offset so the captured
    // values are recognizable downstream — if the system passed the
    // wrong fields through, a default-zero offset / identity rotation
    // would silently match.
    let captured_offset = DVec3::new(6_778_137.0, 0.0, 0.0);
    let captured_rot = DMat3::from_rotation_z(std::f64::consts::FRAC_PI_3);

    {
        let world = app.world_mut();
        let source_entities = [source];
        let body_entities = [body];
        let mut ctx =
            BevySimContext::<astrodyn::Earth>::new(world, &source_entities, &body_entities);
        ctx.attach_to_frame(
            0,
            0,
            SourceFrameKind::Inertial,
            captured_offset,
            captured_rot,
        );
    }

    step_one_tick(&mut app);

    let attached = app
        .world()
        .get::<FrameAttachedC>(body)
        .expect("frame_attach_system must insert FrameAttachedC after one tick");
    assert_eq!(
        attached.parent_frame, expected_parent,
        "SourceFrameKind::Inertial must resolve to the source's FrameEntityC"
    );
    assert_eq!(
        attached.offset, captured_offset,
        "FrameAttachEvent.offset must be carried through to FrameAttachedC.offset unchanged"
    );
    assert_eq!(
        attached.t_parent_body, captured_rot,
        "FrameAttachEvent.t_parent_body must be carried through unchanged"
    );
}

/// `BevySimContext::attach_to_frame(SourceFrameKind::Pfix)` resolves the
/// parent frame entity to the source's [`PfixFrameEntityC`] — distinct
/// from the inertial frame entity — and the next tick records that
/// pfix entity on the body's [`FrameAttachedC`].
#[test]
fn attach_to_frame_pfix_routes_to_pfix_frame_entity() {
    let (mut app, source, body) = build_test_app();

    let inertial_parent = app
        .world()
        .get::<FrameEntityC>(source)
        .expect("Startup must have registered FrameEntityC on the source")
        .0;
    let expected_parent = app
        .world()
        .get::<PfixFrameEntityC>(source)
        .expect(
            "Startup must register PfixFrameEntityC when the source carries PlanetFixedRotationC",
        )
        .0;
    assert_ne!(
        expected_parent, inertial_parent,
        "PfixFrameEntityC must be a distinct entity from FrameEntityC — \
         otherwise this test would not actually exercise the pfix routing branch"
    );

    let captured_offset = DVec3::new(0.0, 6_778_137.0, 0.0);
    let captured_rot = DMat3::from_rotation_x(std::f64::consts::FRAC_PI_4);

    {
        let world = app.world_mut();
        let source_entities = [source];
        let body_entities = [body];
        let mut ctx =
            BevySimContext::<astrodyn::Earth>::new(world, &source_entities, &body_entities);
        ctx.attach_to_frame(0, 0, SourceFrameKind::Pfix, captured_offset, captured_rot);
    }

    step_one_tick(&mut app);

    let attached = app
        .world()
        .get::<FrameAttachedC>(body)
        .expect("frame_attach_system must insert FrameAttachedC after one tick");
    assert_eq!(
        attached.parent_frame, expected_parent,
        "SourceFrameKind::Pfix must resolve to the source's PfixFrameEntityC, \
         not the inertial FrameEntityC"
    );
    assert_eq!(attached.offset, captured_offset);
    assert_eq!(attached.t_parent_body, captured_rot);
}

/// Per-call unique suffix for swept test-body identities (#664): helpers
/// spawning multiple bodies per App must mint distinct identities.
static NEXT_BODY_UID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
