//! Schedule-order regression: a frame-attached body's
//! `AstrodynSet::Environment` consumers (gravity, atmosphere) must observe
//! the post-`propagate_frame_attached_state_system` body state, not a
//! one-tick-stale composition.
//!
//! Pre-fix: the per-tick frame-attached propagation lived in
//! `AstrodynSet::ForceCollection`, so gravity (in `AstrodynSet::Environment`,
//! which runs strictly before ForceCollection) and the
//! `AstrodynSet::Interaction` force producers (drag, SRP, gravity-torque)
//! read the body's pre-tick `TranslationalStateC` — for a body that
//! had just been frame-attached this tick, that meant the default-zero
//! state, not the parent-frame-derived offset. The fix moves the
//! propagation pass between `AstrodynSet::EphemerisUpdate` and
//! `AstrodynSet::Environment` so all downstream consumers see the
//! freshly-derived state on the same tick.
//!
//! This test exercises the gravity path because it is the simplest
//! Environment consumer to assert against directly:
//! `gravity_computation_system` writes the central-source gravity
//! acceleration into `GravityAccelerationC.grav_accel`. With the
//! attach offset placed at a known LEO position the expected
//! acceleration is the analytical `-mu * r / |r|^3`. Pre-fix, the
//! gravity write would reflect the body's default-zero position
//! (which `gravity_computation_system` would either skip or evaluate
//! at a degenerate point) on the very first tick of an attach;
//! post-fix it matches the analytical value to roundoff.

mod common;

use astrodyn::{
    DynamicsConfig, GravityControl, GravityControls, GravityRole, MassProperties, RotationalState,
    TranslationalState,
};
use astrodyn_bevy::{
    DynamicsConfigC, FrameAttachEvent, GravityAccelerationC, GravityControlsC, MassPropertiesC,
    RootFrameEntityR, RotationalStateC, TranslationalStateC,
};
use bevy::prelude::*;

use common::*;

/// A body frame-attached at a non-zero offset must have its gravity
/// acceleration computed from the post-propagation root-inertial
/// position on the same tick the attach event is dispatched.
///
/// The expected analytical acceleration is `-mu * r / |r|^3` for the
/// central Earth source at the captured offset (LEO altitude). Pre-fix
/// the gravity write would either be zero (default-zero body state ⇒
/// `gravity_computation_system` skips singularity) or evaluate at the
/// origin and produce garbage; post-fix it agrees with the analytical
/// value to numerical roundoff.
#[test]
fn bevy_parity_frame_attach_gravity_sees_propagated_state() {
    let mut app = new_bevy_app(DT);

    let earth = spawn_earth_source(&mut app);

    // Spawn a body with default-zero state. Gravity will be computed
    // *after* `propagate_frame_attached_state_system` sets its
    // position to the captured offset.
    let body = app
        .world_mut()
        .spawn((
            Name::new("frame_attached_body"),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState::default()),
            RotationalStateC::from(astrodyn::typed_bridge::rot_raw_to_self_ref(
                &(RotationalState::default()),
            )),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(1_000.0)),
            )),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                // Re-use the central Earth source spawned above. The
                // helper hands back its Entity; we rebuild the
                // controls here so the test owns its own setup.
                controls: vec![GravityControl::new_spherical(earth, GravityRole::Central)],
            }),
            GravityAccelerationC::default(),
        ))
        .id();

    // Capture an LEO-radius offset. The root frame is at the origin
    // and stationary, so the captured offset *is* the body's
    // root-inertial position after `propagate_frame_attached_state_system`.
    let parent_frame = **app.world().resource::<RootFrameEntityR>();
    let attach_offset = iss_trans().position;

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<FrameAttachEvent>>()
        .write(FrameAttachEvent {
            body,
            parent_frame,
            offset: attach_offset,
            t_parent_body: glam::DMat3::IDENTITY,
        });

    step_bevy_dt(&mut app, 1, DT);

    // Sanity: post-fix, the body's `TranslationalStateC` reflects the
    // captured offset (the propagation pass ran).
    let body_pos = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(body)
        .expect("frame-attached body must keep TranslationalStateC")
        .0
        .position
        .raw_si();
    assert!(
        (body_pos - attach_offset).length() < 1e-9,
        "frame-attached body must end the tick at the captured offset \
         (got {body_pos:?}, expected {attach_offset:?}); the propagation \
         pass either did not run or ran after the read site."
    );

    // The analytical central-Earth gravity at the attach offset.
    let r = attach_offset;
    let r_mag = r.length();
    let expected_accel = -MU_EARTH * r / (r_mag * r_mag * r_mag);

    let grav_accel = app
        .world()
        .get::<GravityAccelerationC>(body)
        .expect("body must carry GravityAccelerationC")
        .0
        .grav_accel
        .raw_si();

    // Pre-fix this would be zero (or near-singular) because gravity
    // ran before propagation. Post-fix it matches the analytical
    // central-mass acceleration to numerical roundoff.
    let err = (grav_accel - expected_accel).length();
    let tol = 1e-6 * expected_accel.length();
    assert!(
        err < tol,
        "gravity_computation_system must observe the post-propagation \
         body state. expected_accel={expected_accel:?}, got={grav_accel:?}, \
         err={err:.3e} (tol={tol:.3e}). If gravity ran before \
         propagate_frame_attached_state_system the acceleration would \
         reflect the body's pre-tick default-zero position."
    );
}
