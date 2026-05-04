//! Bevy ECS frame-attach parity for bodies whose `IntegSourceC` is a
//! non-root planet (lunar-orbit integ frame). Pins the
//! root→integ-frame lower the per-step propagation system applies
//! before writing back to `TranslationalStateC`.
//!
//! The runner's `propagate_frame_attached_state` lowers the kernel's
//! root-inertial output through each body's `IntegOrigin` at the
//! writeback boundary
//! (`crates/jeod_runner/src/simulation/frame_attach.rs:335-339`):
//!
//! ```ignore
//! TranslationalStateTyped::<IntegrationFrame>::from_inertial(trans_root, integ_origin)
//! ```
//!
//! For a body integrated in `PlanetInertial<P>` (a non-root planet),
//! the body's `TranslationalStateC` storage is planet-relative —
//! root-inertial coords minus the planet's offset. A propagation
//! system that copies the kernel's root-inertial output straight into
//! `TranslationalStateC` mislabels the frame: the typed phantom says
//! "integration frame" but the numeric coordinates are in root.
//! Downstream consumers that read `TranslationalStateC` (gravity,
//! drag-velocity, LVLH, geodetic, orbital elements, …) then operate
//! on a body offset by the inter-source separation distance.
//!
//! What this test pins:
//!
//! 1. The body's `FrameTransC` (the frame-tree node, which is
//!    `ChildOf(integ_frame)`) carries the parent-frame composition in
//!    root-inertial coordinates.
//! 2. The body's `TranslationalStateC` carries the same composition
//!    lowered through the body's `IntegOrigin` — i.e. root coords
//!    minus the integ frame's origin.
//! 3. The two storage values differ by exactly `MOON_OFFSET` (lower
//!    symmetry). A regression that drops the lower would put
//!    root-inertial coords into the integration-frame slot, breaking
//!    every `PlanetInertial<P>` consumer of the body state.
//!
//! Mirrors `bevy_parity_detach_non_root_integ_source` from the detach
//! path: same lift/lower contract, applied at the per-step
//! frame-attach writeback rather than the detach handler. For
//! root-integrated bodies (no `IntegSourceC`) the origin is
//! identically zero and the lower is a numerical no-op; for non-root
//! integ_source it is load-bearing.

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, FrameAttachEvent, FrameAttachedC, FrameDerivativesC, FrameEntityC,
    FrameTransC, GravityControlsC, IntegSourceC, JeodPlugin, MassPropertiesC, PlanetBundle,
    RootFrameEntityR, RotationalStateC, SourceMutator, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_sim::{
    DynamicsConfig, GravityControls, MassProperties, RotationalState, TranslationalState, EARTH,
    MOON,
};
use std::time::Duration;

const DT: f64 = 60.0;
const MOON_OFFSET: DVec3 = DVec3::new(3.844e8, 0.0, 0.0);
/// Captured attach offset, in the parent frame's coordinates. Picked
/// well above any conceivable f64 round-off so a `MOON_OFFSET`-sized
/// regression is unambiguous in the failure messages.
const ATTACH_OFFSET: DVec3 = DVec3::new(7.0e6, 0.0, 0.0);

fn body_mass() -> MassProperties {
    MassProperties::with_inertia(
        1_000.0,
        DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0)),
        DVec3::ZERO,
    )
}

fn six_dof_config() -> DynamicsConfig {
    DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    }
}

fn initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(2.0e6, 0.0, 0.0),
        velocity: DVec3::ZERO,
    }
}

fn initial_rot() -> RotationalState {
    RotationalState::default()
}

/// Frame-attach a body whose `IntegSourceC` points at a non-root
/// planet (Moon at `MOON_OFFSET`), then step once. The per-tick
/// propagation must lower the kernel's root-inertial output through
/// the body's `IntegOrigin` before stamping `TranslationalStateC`.
#[test]
fn bevy_parity_frame_attach_non_root_integ_source_lowers_to_integ_frame() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Moon", &MOON))
        .id();

    let body = app
        .world_mut()
        .spawn((
            Name::new("Lunar"),
            DynamicsConfigC(six_dof_config()),
            MassPropertiesC::from(body_mass()),
            TranslationalStateC::from(initial_trans()),
            RotationalStateC::from(initial_rot()),
            FrameDerivativesC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
            IntegSourceC(Some(moon)),
        ))
        .id();

    // Run startup so register_source_frames + register_body_frames fire.
    // After this, the body's frame entity is `ChildOf(moon.frame_entity)`.
    app.world_mut().run_schedule(Startup);

    // Park the Moon at MOON_OFFSET in root inertial. This is the
    // body's integration-frame origin.
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator| {
            m.set_source_position(moon, MOON_OFFSET);
        });
    app.world_mut().run_system(sys).unwrap();

    // Attach the body to the *root* frame at `ATTACH_OFFSET`. The
    // parent-frame composition gives the body root-inertial coords =
    // ATTACH_OFFSET (the root frame is identity). After the lower,
    // `TranslationalStateC` (in Moon-inertial integ-frame coords)
    // must equal `ATTACH_OFFSET - MOON_OFFSET`.
    let parent_frame = **app.world().resource::<RootFrameEntityR>();
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
        .write(FrameAttachEvent {
            body,
            parent_frame,
            offset: ATTACH_OFFSET,
            t_parent_body: DMat3::IDENTITY,
        });

    // Step once so frame_attach_system inserts FrameAttachedC and
    // propagate_frame_attached_state_system runs the writeback.
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    // Sanity: the body must now carry the FrameAttachedC marker.
    assert!(
        app.world().get::<FrameAttachedC>(body).is_some(),
        "frame_attach_system must insert FrameAttachedC after a FrameAttachEvent"
    );

    // 1. The body's frame entity carries the *parent-relative*
    //    composition: position relative to its `ChildOf` parent in
    //    parent-frame coordinates (the convention `FrameTransC`'s
    //    doc on `components.rs` and `sync_body_to_frame_system` /
    //    the runner's `node.state.trans = bodies[idx].trans` line
    //    in `crates/jeod_runner/src/simulation/frame_attach.rs:369`
    //    both establish). The body frame is `ChildOf(moon.frame)`
    //    (set by `register_body_frames_system` because this body's
    //    `IntegSourceC = Some(moon)`; frame-attach does not reparent
    //    the frame node), so the parent-relative coordinates are
    //    Moon-relative — identical to the lowered integ-frame value
    //    written into `TranslationalStateC` below: `ATTACH_OFFSET -
    //    MOON_OFFSET`. A regression that wrote the kernel's
    //    pre-lower root-inertial value into `FrameTransC.position`
    //    (the bug this assertion pins) would put `ATTACH_OFFSET`
    //    here — off by `MOON_OFFSET` and inconsistent with the
    //    body's `TranslationalStateC`, breaking every frame-tree
    //    walker that reads through this node (`compute_relative_state`
    //    consumers in gravity / drag / LVLH / geodetic / frame
    //    switch).
    let body_frame_entity = app
        .world()
        .get::<FrameEntityC>(body)
        .expect("register_body_frames_system must insert FrameEntityC")
        .0;
    let frame_trans = *app
        .world()
        .get::<FrameTransC>(body_frame_entity)
        .expect("body's frame entity must carry FrameTransC");
    let expected_frame_pos = ATTACH_OFFSET - MOON_OFFSET;
    let frame_pos_err = (frame_trans.position - expected_frame_pos).length();
    let frame_tol = 1e-6;
    assert!(
        frame_pos_err < frame_tol,
        "FrameTransC.position not lowered to parent-relative (Moon-inertial) coords:\n  \
         got {:?}\n  expected {expected_frame_pos:?} (= ATTACH_OFFSET - MOON_OFFSET)\n  \
         delta {:?} (length {frame_pos_err:.3e}, tol {frame_tol:.3e})\n\n\
         A regression that copied the kernel's pre-lower root-inertial output \
         into FrameTransC would put ATTACH_OFFSET here, breaking the parent- \
         relative invariant `compute_relative_state` walks rely on.",
        frame_trans.position,
        frame_trans.position - expected_frame_pos,
    );
    assert!(
        frame_trans.velocity.length() < 1e-9,
        "FrameTransC.velocity must be zero (frame-attached to a stationary root \
         frame, integ frame at rest in root): got {:?}",
        frame_trans.velocity,
    );

    // 2. The body's `TranslationalStateC` must carry the lowered
    //    integration-frame coordinates. Storage convention: typed
    //    `<PlanetInertial<SelfPlanet>>`, numerically planet-relative
    //    (Moon in this test). Without the lower, this slot would
    //    carry root-inertial coords (ATTACH_OFFSET), i.e. off by
    //    `MOON_OFFSET` (~3.84e8 m).
    let body_trans = app
        .world()
        .get::<TranslationalStateC>(body)
        .expect("body must keep TranslationalStateC after frame attach")
        .0
        .position
        .raw_si();
    let expected_integ_pos = ATTACH_OFFSET - MOON_OFFSET;
    let err = (body_trans - expected_integ_pos).length();
    let tol = 1e-6;
    assert!(
        err < tol,
        "TranslationalStateC.position not lowered to Moon-inertial integ frame:\n  \
         got {body_trans:?}\n  expected {expected_integ_pos:?} (= ATTACH_OFFSET - MOON_OFFSET)\n  \
         delta {:?} (length {err:.3e}, tol {tol:.3e})\n\n\
         A regression that drops the IntegOrigin lower would write the kernel's \
         root-inertial output straight into TranslationalStateC, leaving the \
         numeric coords off by MOON_OFFSET (~3.84e8 m).",
        body_trans - expected_integ_pos,
    );

    // The same body's velocity must lower symmetrically. With the
    // Moon at rest in root-inertial and an attach to the (also at
    // rest) root frame, the body velocity in the integ frame is
    // identically zero — both branches of the lower must agree.
    let body_vel = app
        .world()
        .get::<TranslationalStateC>(body)
        .unwrap()
        .0
        .velocity
        .raw_si();
    assert!(
        body_vel.length() < 1e-9,
        "TranslationalStateC.velocity must be zero (frame-attached to a stationary \
         root frame, integ frame at rest in root): got {body_vel:?}"
    );
}
