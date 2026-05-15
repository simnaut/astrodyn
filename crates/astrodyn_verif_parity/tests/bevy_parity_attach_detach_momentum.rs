// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy adapter parity for the JEOD attach/detach
//! momentum-conservation port + detached-subtree ballistic tracking.

#![allow(
    clippy::float_cmp,
    reason = "bevy-parity tests assert bit-exact identity between runner and Bevy state fields"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test step counts fit exactly in f64 mantissa and usize"
)]
//! Mirrors `models/dynamics/dyn_body/src/dyn_body_attach.cc` and
//! `dyn_body_detach.cc` in JEOD.
//!
//! Three layers of coverage:
//!
//! 1. **Attach momentum conservation.** Spawn two free-flying bodies
//!    with non-trivial relative state, fire `AttachEvent`, verify the
//!    parent's `TranslationalStateC` / `RotationalStateC` post-event
//!    matches `astrodyn::stage_attach_combine` byte-for-byte. Linear
//!    momentum about the integration-frame origin and angular momentum
//!    about the new combined CoM are preserved across the merge.
//!
//! 2. **Detach captures ballistic state.** Spawn a parent + child
//!    pair, attach, propagate, fire `DetachEvent`, verify the
//!    detached entity now carries `DetachedSubtreeStateC` whose
//!    composite-body state matches the child's pre-detach
//!    `TranslationalStateC` / `RotationalStateC`.
//!
//! 3. **Detached subtree drifts ballistically.** After detach, run a
//!    few ticks and verify the detached entity's state advances under
//!    free-flight kinematics — `position += velocity·dt`, attitude
//!    rotates under `ang_vel`, velocity and `ang_vel` unchanged. The
//!    detached entity is no longer integrated by the wrench-aggregation
//!    walk; `step_detached_system` owns its propagation.

use astrodyn::{
    GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource, JeodQuat,
    MassProperties, MassTree, RotationalState, StageAttachInputs, TranslationalState,
};
use astrodyn_bevy::frame_param::RelativeFrameState;
use astrodyn_bevy::{
    AstrodynPlugin, AttachEvent, DetachEvent, DetachedSubtreeStateC, DynamicsConfigC, FrameAngVelC,
    FrameDerivativesC, FrameEntityC, FrameRotC, FrameTransC, GravityAccelerationC,
    GravityControlsC, GravitySourceC, IntegSourceC, IntegrationDtR, MassBodyIdC, MassPropertiesC,
    MassTreeR, RootFrameEntityR, RotationalStateC, SourceInertialPositionC, TranslationalStateC,
};
use bevy::prelude::*;
use glam::{DMat3, DVec3};
use std::time::Duration;

/// Build a Bevy app with two free-flying bodies registered in a shared
/// `MassTreeR`, both carrying full 6-DOF state. No gravity sources —
/// the bodies are inertial coasters so attach/detach math is the only
/// thing under test.
fn build_two_body_world(
    sim_dt: f64,
    parent_mass: MassProperties,
    parent_trans: TranslationalState,
    parent_rot: RotationalState,
    child_mass: MassProperties,
    child_trans: TranslationalState,
    child_rot: RotationalState,
) -> (
    App,
    Entity,
    Entity,
    astrodyn::MassBodyId,
    astrodyn::MassBodyId,
) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.insert_resource(IntegrationDtR(sim_dt));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(parent_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(child_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    (app, parent_entity, child_entity, id_a, id_b)
}

fn step(app: &mut App, n: usize, dt: f64) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

fn read_position(world: &World, entity: Entity) -> DVec3 {
    world
        .get::<TranslationalStateC<astrodyn::Earth>>(entity)
        .expect("entity has TranslationalStateC")
        .0
        .position
        .raw_si()
}

fn read_velocity(world: &World, entity: Entity) -> DVec3 {
    world
        .get::<TranslationalStateC<astrodyn::Earth>>(entity)
        .expect("entity has TranslationalStateC")
        .0
        .velocity
        .raw_si()
}

fn read_ang_vel(world: &World, entity: Entity) -> DVec3 {
    astrodyn::typed_bridge::rot_typed_to_raw(
        &world
            .get::<RotationalStateC>(entity)
            .expect("entity has RotationalStateC")
            .0,
    )
    .ang_vel_body
}

fn read_mass(world: &World, entity: Entity) -> f64 {
    astrodyn::typed_bridge::mass_typed_to_raw(
        &world
            .get::<MassPropertiesC>(entity)
            .expect("entity has MassPropertiesC")
            .0,
    )
    .mass
}

/// Attach with relative translational velocity at non-zero offset
/// induces angular momentum (the JEOD "magical merge" — see
/// `dyn_body_attach.cc` / `combine_states_at_attach`). Verify the
/// parent's post-attach `TranslationalStateC` / `RotationalStateC`
/// match the kernel's output byte-for-byte.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_conserves_linear_and_angular_momentum() {
    let parent_mass = MassProperties::with_inertia(
        1000.0,
        DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        1000.0,
        DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
        DVec3::ZERO,
    );

    let parent_trans = TranslationalState {
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
    };
    let child_trans = TranslationalState {
        position: DVec3::new(2.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 1.0, 0.0),
    };
    let parent_rot = RotationalState::default();
    let child_rot = RotationalState::default();

    let (mut app, parent_entity, child_entity, _id_a, _id_b) = build_two_body_world(
        1.0,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );

    // Capture pre-attach mass properties so we can independently
    // compute the kernel's expected output without going through the
    // tree.
    let combined_mass_expected = MassProperties::with_inertia(
        2000.0,
        // Parallel-axis: each body contributes I = I_self + m·d² about
        // the new CoM at (1, 0, 0). Both bodies' offsets are 1 m along
        // x; mass·d² = 1000·1·1 = 1000 each on the y- and z-diagonal
        // (no contribution along the offset axis). Plus the original
        // 500 each gives 500 (xx) and 500+1000+1000 = 2500-ish — but
        // `tree.attach` does this for us; we just feed the kernel the
        // same combined inertia the tree computes.
        DMat3::from_diagonal(DVec3::new(1000.0, 2500.0, 2500.0)),
        DVec3::new(1.0, 0.0, 0.0),
    );
    // We don't rely on the exact combined inertia diagonal above — the
    // kernel reads `combined_mass.position` (CoM) and
    // `combined_mass.inverse_inertia` for the angular-momentum solve.
    // Instead, fire the `AttachEvent`, let the tree compute the
    // composite, and use the tree's combined mass as the kernel's
    // ground-truth input.
    let _ = combined_mass_expected; // unused — kept for documentation

    // Fire the attach event.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::new(2.0, 0.0, 0.0),
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });

    step(&mut app, 1, 1.0);

    // Compute the kernel's expected output independently. We use the
    // tree's post-attach combined mass (the same value `staging_system`
    // feeds the kernel) so this is a true parity check, not a tautology.
    let combined_mass = app
        .world()
        .resource::<MassTreeR>()
        .0
        .get(_id_a)
        .composite_properties;

    let q = JeodQuat::identity();
    let expected = astrodyn::stage_attach_combine(StageAttachInputs {
        parent_position: parent_trans.position,
        parent_velocity: parent_trans.velocity,
        parent_quaternion: q,
        parent_ang_vel_body: DVec3::ZERO,
        parent_mass,
        orig_parent_cm_struct: parent_mass.position,
        parent_t_inertial_struct: DMat3::IDENTITY,
        child_position: child_trans.position,
        child_velocity: child_trans.velocity,
        child_quaternion: q,
        child_ang_vel_body: DVec3::ZERO,
        child_mass,
        combined_mass,
    });

    // Verify the parent entity now holds the merged state.
    let pos = read_position(app.world(), parent_entity);
    let vel = read_velocity(app.world(), parent_entity);
    let omega = read_ang_vel(app.world(), parent_entity);
    assert!(
        (pos - expected.position).length() < 1e-9,
        "post-attach position: bevy {pos:?} vs kernel {:?}",
        expected.position
    );
    assert!(
        (vel - expected.velocity).length() < 1e-9,
        "post-attach velocity: bevy {vel:?} vs kernel {:?}",
        expected.velocity
    );
    assert!(
        (omega - expected.ang_vel_body).length() < 1e-9,
        "post-attach ang_vel: bevy {omega:?} vs kernel {:?}",
        expected.ang_vel_body
    );

    // Sanity: linear momentum is conserved about the integration-frame
    // origin. p_pre = m_p·v_p + m_c·v_c = (0, 1000, 0); p_post = m_t·v_t.
    let p_post = combined_mass.mass * vel;
    let p_pre = parent_mass.mass * parent_trans.velocity + child_mass.mass * child_trans.velocity;
    assert!(
        (p_post - p_pre).length() < 1e-6,
        "linear momentum not conserved: pre {p_pre:?} vs post {p_post:?}"
    );

    // Sanity: composite mass on the parent matches the tree.
    assert!(
        (read_mass(app.world(), parent_entity) - combined_mass.mass).abs() < 1e-12,
        "composite mass on parent should match the tree's post-attach value"
    );
}

/// `combine_states_at_attach` should leave a "soft" merge (no relative
/// motion) untouched: parent state preserved, no spurious spin.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_no_relative_motion_preserves_parent_state() {
    let parent_mass = MassProperties::with_inertia(
        100.0,
        DMat3::from_diagonal(DVec3::new(50.0, 50.0, 50.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        50.0,
        DMat3::from_diagonal(DVec3::new(20.0, 20.0, 20.0)),
        DVec3::ZERO,
    );

    let v = DVec3::new(0.0, 7600.0, 0.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: v,
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: v,
    };
    let parent_rot = RotationalState::default();
    let child_rot = RotationalState::default();

    let (mut app, parent_entity, child_entity, _, _) = build_two_body_world(
        1.0,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    let vel = read_velocity(app.world(), parent_entity);
    let omega = read_ang_vel(app.world(), parent_entity);
    assert!((vel - v).length() < 1e-9, "soft merge changed velocity");
    assert!(
        omega.length() < 1e-12,
        "soft merge induced spurious spin: {omega:?}"
    );
}

/// Detach captures the about-to-be-detached body's instantaneous
/// composite-body state into `DetachedSubtreeStateC`. After an attach,
/// the child's own `TranslationalStateC` / `RotationalStateC` are
/// stale (only the parent carries the merged composite). The detach
/// handler must derive the child's instantaneous state from the
/// parent's composite-body inertial state at the detach instant —
/// rigid-body composition via `propagate_forward` over the mass-tree
/// offset chain — not from the child's own (stale) component values.
///
/// This test exercises that real path end-to-end. The attach is a
/// soft merge (identical pre-state on parent and child, zero offset)
/// so the kernel's expected child instantaneous state is the parent's
/// composite-body state at detach. The test does NOT manually reset
/// the child's `TranslationalStateC` — if a regression dropped that
/// real derivation, the captured state would diverge from the
/// expected and this test would fail.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_detach_captures_subtree_state() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);

    // Both bodies share the same orbital state and identical attitude
    // / spin. Soft-merge invariant: the merged composite has the same
    // translational and rotational state as the inputs, so at the
    // detach instant the rigid-body composition recovers exactly the
    // shared pre-attach state. Crucially this remains true *without*
    // patching the child's stale `TranslationalStateC` by hand — the
    // detach handler derives the child's state from the parent's
    // composite via the mass-tree offsets.
    let initial_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 1.0),
    };
    let omega = DVec3::new(0.001, 0.0, 0.0);
    let initial_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: omega,
    };

    let (mut app, parent_entity, child_entity, _id_a, _id_b) = build_two_body_world(
        1.0,
        parent_mass,
        initial_trans,
        initial_rot,
        child_mass,
        initial_trans,
        initial_rot,
    );

    // Attach (soft merge, zero offset, identity rotation).
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // Snapshot the parent's post-attach composite-body state. This is
    // the live state the detach handler will derive the child from —
    // *no* manual reset of the child's TranslationalStateC.
    let parent_pos_at_detach = read_position(app.world(), parent_entity);
    let parent_vel_at_detach = read_velocity(app.world(), parent_entity);
    let parent_omega_at_detach = read_ang_vel(app.world(), parent_entity);

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent {
            child: child_entity,
        });
    step(&mut app, 1, 1.0);

    // The child entity must now carry `DetachedSubtreeStateC`.
    let detached = app
        .world()
        .get::<DetachedSubtreeStateC>(child_entity)
        .expect("DetachEvent should have inserted DetachedSubtreeStateC on the child entity");

    // The captured state must match the parent's composite-body state
    // at the detach instant (soft-merge: child's instantaneous state
    // == parent's composite). We tolerate one tick of
    // `step_detached_system` advance since the `step(&mut app, 1, 1.0)`
    // after `DetachEvent` runs both `staging_system` (which captures +
    // inserts) and `step_detached_system` (which advances by `dt`).
    let dt = 1.0;
    let expected_pos = parent_pos_at_detach + parent_vel_at_detach * dt;
    let detached_pos = detached.0.composite_position.raw_si();
    let detached_vel = detached.0.composite_velocity.raw_si();
    assert!(
        (detached_pos - expected_pos).length() < 1e-9,
        "detached pos: {:?} expected {:?} (= parent_composite_at_detach + vel·dt)",
        detached_pos,
        expected_pos
    );
    assert!(
        (detached_vel - parent_vel_at_detach).length() < 1e-12,
        "detached velocity should match parent composite: {:?} vs {:?}",
        detached_vel,
        parent_vel_at_detach
    );
    assert!(
        (detached.0.composite_ang_vel_body - parent_omega_at_detach).length() < 1e-12,
        "detached ang_vel should match parent composite: {:?} vs {:?}",
        detached.0.composite_ang_vel_body,
        parent_omega_at_detach
    );
}

/// Stronger variant: with a NON-zero attach offset, the child's
/// instantaneous state at the detach instant is the parent's
/// composite-body state shifted by the rigid-body offset (and any
/// rotational contribution). Verifies the detach handler's
/// `propagate_forward` walk is actually being applied — not just
/// "happens to be zero in the soft case".
#[test]
fn bevy_parity_attach_detach_momentum_bevy_detach_derives_child_state_via_rigid_body_composition() {
    let parent_mass = MassProperties::with_inertia(
        1000.0,
        DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        500.0,
        DMat3::from_diagonal(DVec3::new(200.0, 200.0, 200.0)),
        DVec3::ZERO,
    );

    // Parent is rotating about Z at a noticeable rate so the
    // child's composition picks up a velocity-from-rotation term that
    // wouldn't show up at zero ang_vel.
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let parent_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::new(0.0, 0.0, 0.01),
    };
    // Child shares the parent's rigid-body motion (same translational
    // velocity + same angular velocity) but is offset along the
    // parent's structure x-axis. The resulting attach is a soft merge
    // in the shared-rigid-body sense (no relative momentum, no induced
    // spin) — so the merged composite-body state at detach equals the
    // parent's pre-attach state shifted by the structure-frame offset
    // to the new CoM (which sits between the two CoMs). At the detach
    // instant, the child's instantaneous state is recoverable by
    // applying `propagate_forward` from the merged composite using
    // the mass-tree's `composite_wrt_pstr` offset.
    let attach_offset = DVec3::new(2.0, 0.0, 0.0);
    let child_trans = TranslationalState {
        position: parent_trans.position + attach_offset,
        // For shared rigid-body motion, child velocity = parent.vel +
        // ω × r (in inertial). With identity attitude, body axes ==
        // inertial.
        velocity: parent_trans.velocity + parent_rot.ang_vel_body.cross(attach_offset),
    };
    let child_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: parent_rot.ang_vel_body,
    };

    let (mut app, parent_entity, child_entity, _id_a, _id_b) = build_two_body_world(
        1.0,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                attach_offset,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // After attach: parent carries the merged composite-body state.
    // Child's TranslationalStateC may be stale — we deliberately do
    // NOT touch it. The detach handler is responsible for deriving
    // the child's instantaneous state from the merged composite.
    let parent_pos_at_detach = read_position(app.world(), parent_entity);
    let parent_vel_at_detach = read_velocity(app.world(), parent_entity);
    let parent_omega_at_detach = read_ang_vel(app.world(), parent_entity);

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent {
            child: child_entity,
        });
    step(&mut app, 1, 1.0);

    let detached = app
        .world()
        .get::<DetachedSubtreeStateC>(child_entity)
        .expect("DetachEvent should have inserted DetachedSubtreeStateC on the child entity");

    // Expected: rigid-body shared-motion invariant — every point on
    // the rigid body has velocity = v_cm + ω × (r − r_cm). The child's
    // composite-CoM sits at child_trans.position in inertial. After
    // detach, the captured state should satisfy this rigid-body
    // composition exactly (no momentum was injected — both bodies were
    // already moving together pre-attach), then advance ballistically
    // by one tick.
    let dt = 1.0;
    let r_child_rel_parent = child_trans.position - parent_pos_at_detach;
    let expected_child_vel =
        parent_vel_at_detach + parent_omega_at_detach.cross(r_child_rel_parent);
    let expected_child_pos_at_detach = child_trans.position;
    let expected_pos_after_step = expected_child_pos_at_detach + expected_child_vel * dt;

    let detached_pos = detached.0.composite_position.raw_si();
    let detached_vel = detached.0.composite_velocity.raw_si();
    assert!(
        (detached_pos - expected_pos_after_step).length() < 1e-6,
        "detached pos via rigid-body composition: got {:?}, expected {:?}",
        detached_pos,
        expected_pos_after_step
    );
    assert!(
        (detached_vel - expected_child_vel).length() < 1e-6,
        "detached velocity via rigid-body composition: got {:?}, expected {:?}",
        detached_vel,
        expected_child_vel
    );
}

/// After detach, the entity's `TranslationalStateC` advances under
/// free-flight kinematics each tick. Position drifts at velocity,
/// velocity unchanged.
///
/// This test exercises the real attach → propagate → detach → drift
/// path without manually patching the child's state in between. The
/// parent and child share rigid-body motion pre-attach (so the
/// soft-merge invariant gives the kernel the same input as if no
/// momentum were exchanged), and the post-detach drift is asserted
/// against the parent's composite-body state at the detach instant
/// — which is what the live detach handler must derive.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_detached_subtree_propagates_ballistically() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    // Both bodies share orbital velocity + zero spin. After attach +
    // soft-merge, the parent's composite-body state is the same as
    // either input, so the post-detach drift baseline is well-defined
    // without needing to hand-patch the child's state.
    let initial_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let (mut app, parent_entity, child_entity, _, _) = build_two_body_world(
        1.0,
        parent_mass,
        initial_trans,
        initial_rot,
        child_mass,
        initial_trans,
        initial_rot,
    );

    // Attach then detach to put the child in DetachedSubtreeStateC.
    // No manual state reset between the two — the detach handler must
    // derive the child's instantaneous state from the parent's
    // composite at the detach instant via `propagate_forward`.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // Capture the parent's composite-body state at the detach instant
    // — this is the live source of truth for the detach handler.
    let parent_pos_at_detach = read_position(app.world(), parent_entity);
    let parent_vel_at_detach = read_velocity(app.world(), parent_entity);

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent {
            child: child_entity,
        });

    // Run 5 ticks of free-flight propagation.
    let n_steps = 5;
    let dt = 1.0;
    step(&mut app, n_steps, dt);

    let post_pos = read_position(app.world(), child_entity);
    let post_vel = read_velocity(app.world(), child_entity);
    // The detach itself runs in the first of these 5 ticks, then
    // step_detached_system advances the body for `n_steps` ticks
    // (5 here). Expected: child starts at parent's composite state
    // (zero offset → identical to parent at detach instant) and drifts
    // at parent's velocity for the full block of ticks.
    let expected = parent_pos_at_detach + parent_vel_at_detach * (n_steps as f64) * dt;
    assert!(
        (post_pos - expected).length() < 1e-6,
        "detached body should drift at parent composite velocity: got {post_pos:?}, expected {expected:?}"
    );
    assert!(
        (post_vel - parent_vel_at_detach).length() < 1e-9,
        "detached velocity should be unchanged: got {post_vel:?}, expected {parent_vel_at_detach:?}"
    );

    // The DetachedSubtreeStateC's internal composite state should
    // mirror the synced TranslationalStateC.
    let detached = app
        .world()
        .get::<DetachedSubtreeStateC>(child_entity)
        .expect("child still has DetachedSubtreeStateC during free flight");
    assert!(
        (detached.0.composite_position.raw_si() - post_pos).length() < 1e-12,
        "DetachedSubtreeStateC position must mirror TranslationalStateC after step_detached_system"
    );
}

/// Re-attach after detach: the captured `DetachedSubtreeStateC` must
/// be removed when the entity is re-attached. The combine kernel must
/// run on the recaptured state.
///
/// The test exercises the real attach → detach → re-attach cycle
/// without manually resetting the child's `TranslationalStateC` in
/// between. The detach handler is responsible for deriving the
/// child's instantaneous state from the parent's composite via
/// `propagate_forward` — and the re-attach handler then consumes
/// that captured ballistic state.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_re_attach_consumes_detached_state() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    // Shared rigid-body state so soft-merge invariant lets us
    // exercise the cycle without touching child state by hand.
    let initial_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let (mut app, parent_entity, child_entity, _, _) = build_two_body_world(
        1.0,
        parent_mass,
        initial_trans,
        initial_rot,
        child_mass,
        initial_trans,
        initial_rot,
    );

    // Attach.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // Detach — handler derives child state from parent's composite.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent {
            child: child_entity,
        });
    step(&mut app, 1, 1.0);
    assert!(
        app.world()
            .get::<DetachedSubtreeStateC>(child_entity)
            .is_some(),
        "child should be detached after DetachEvent"
    );

    // Re-attach — handler consumes the captured DetachedSubtreeStateC.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    assert!(
        app.world()
            .get::<DetachedSubtreeStateC>(child_entity)
            .is_none(),
        "AttachEvent should have removed DetachedSubtreeStateC from re-attached child"
    );
}

/// Exercise the kernel parity directly via `stage_attach_combine` —
/// covers the Tier 1 surface for the orchestration helper.
#[test]
fn bevy_parity_attach_detach_momentum_stage_attach_combine_parity_smoke() {
    use astrodyn::stage_attach_combine;
    let parent_mass = MassProperties::with_inertia(
        100.0,
        DMat3::from_diagonal(DVec3::new(50.0, 60.0, 70.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        50.0,
        DMat3::from_diagonal(DVec3::new(20.0, 25.0, 30.0)),
        DVec3::ZERO,
    );

    // Build the same topology in the runner's arena tree to obtain
    // the post-attach combined mass — the kernel input the orchestration
    // helper takes.
    let mut tree = MassTree::new();
    let pid = tree.add_root("p".into(), parent_mass);
    let cid = tree.add_body("c".into(), child_mass);
    tree.attach(cid, pid, DVec3::new(2.0, 0.0, 0.0), DMat3::IDENTITY);
    let combined = tree.get(pid).composite_properties;

    let q = JeodQuat::identity();
    let out = stage_attach_combine(StageAttachInputs {
        parent_position: DVec3::ZERO,
        parent_velocity: DVec3::ZERO,
        parent_quaternion: q,
        parent_ang_vel_body: DVec3::ZERO,
        parent_mass,
        orig_parent_cm_struct: parent_mass.position,
        parent_t_inertial_struct: DMat3::IDENTITY,
        child_position: DVec3::new(2.0, 0.0, 0.0),
        child_velocity: DVec3::new(0.0, 1.0, 0.0),
        child_quaternion: q,
        child_ang_vel_body: DVec3::ZERO,
        child_mass,
        combined_mass: combined,
    });

    // Linear-momentum check.
    let v_t_expected = (parent_mass.mass * DVec3::ZERO
        + child_mass.mass * DVec3::new(0.0, 1.0, 0.0))
        / combined.mass;
    assert!(
        (out.velocity - v_t_expected).length() < 1e-9,
        "combined velocity should be mass-weighted average"
    );
    // Angular momentum check (about the new combined CoM).
    // Pre: only the child has translational momentum; child sits at
    // r = (2,0,0) - new_cm. Compute the kernel's L and divide by I.
    let new_cm_inertial = combined.position; // T_inertial_to_struct = I → CoM in inertial = struct
    let r_c_rel = DVec3::new(2.0, 0.0, 0.0) - new_cm_inertial;
    let p_c_rel = child_mass.mass * (DVec3::new(0.0, 1.0, 0.0) - v_t_expected);
    let l_c = r_c_rel.cross(p_c_rel);
    let r_p_rel = -new_cm_inertial;
    let p_p_rel = parent_mass.mass * (DVec3::ZERO - v_t_expected);
    let l_p = r_p_rel.cross(p_p_rel);
    let l_total = l_c + l_p;
    let omega_expected = combined.inverse_inertia * l_total;
    assert!(
        (out.ang_vel_body - omega_expected).length() < 1e-9,
        "combined ang_vel should solve I·ω = L: got {:?}, expected {:?}",
        out.ang_vel_body,
        omega_expected
    );
}

/// `step_detached_system` must run before the frame-tree sync /
/// frame-switch evaluation. Detached bodies still carry
/// `FrameEntityC`, so without an explicit ordering constraint the
/// schedule is free to run `sync_body_to_frame_system` against the
/// pre-step body state, then have `step_detached_system` overwrite
/// `TranslationalStateC` afterwards — leaving the body's frame
/// entity desynced from the body for one tick.
///
/// After detach + one `App::update()`, the body's frame entity's
/// `FrameTransC` must reflect the *post-step* body position
/// (advanced by one ballistic `dt`), not the pre-step position the
/// detached subtree carried at the start of the tick.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_step_detached_runs_before_frame_tree_sync() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let initial_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let dt = 1.0;
    let (mut app, parent_entity, child_entity, _, _) = build_two_body_world(
        dt,
        parent_mass,
        initial_trans,
        initial_rot,
        child_mass,
        initial_trans,
        initial_rot,
    );

    // Attach so the child enters the parent's composite, then detach
    // to put the child in `DetachedSubtreeStateC` with a known
    // ballistic state (matches the parent's composite at the detach
    // instant — same as `bevy_parity_attach_detach_momentum_bevy_detached_subtree_propagates_ballistically`).
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, dt);

    let parent_pos_at_detach = read_position(app.world(), parent_entity);
    let parent_vel_at_detach = read_velocity(app.world(), parent_entity);

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent {
            child: child_entity,
        });

    // Run a single tick. `step_detached_system` advances the
    // detached body by `dt`; `sync_body_to_frame_system` then
    // mirrors the post-step body state into the frame-tree node.
    step(&mut app, 1, dt);

    let body_pos = read_position(app.world(), child_entity);
    let expected_body_pos = parent_pos_at_detach + parent_vel_at_detach * dt;
    assert!(
        (body_pos - expected_body_pos).length() < 1e-6,
        "detached TranslationalStateC must advance one ballistic step: got {body_pos:?}, expected {expected_body_pos:?}"
    );

    let body_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("detached body retains FrameEntityC during free flight")
        .0;
    let frame_pos = app
        .world()
        .get::<FrameTransC>(body_frame_entity)
        .expect("body frame entity must carry FrameTransC")
        .position;

    // The crux of the regression: the body frame entity's
    // FrameTransC must mirror the post-`step_detached_system` body
    // state. If `sync_body_to_frame_system` raced ahead of
    // `step_detached_system`, the FrameTransC would still hold the
    // pre-step position and this assertion would fail.
    assert!(
        (frame_pos - body_pos).length() < 1e-12,
        "body frame entity FrameTransC must reflect post-step body position \
         (sync_body_to_frame_system must run after step_detached_system): \
         frame {frame_pos:?}, body {body_pos:?}"
    );
}

/// **Same-integration-frame attach only.** When parent and child
/// already share an integration frame (the common case — both bodies
/// integrate in root inertial), the child body's frame entity must
/// remain parented under its original integration frame entity, NOT
/// under the parent body's frame entity.
///
/// This pins JEOD's frame-tree semantics from
/// `models/dynamics/dyn_body/src/dyn_body_integration.cc::set_integ_frame`
/// and `dyn_body_attach.cc::attach_establish_links`: a body's three
/// reference frames (`structure`, `composite_body`, `core_body`) plus
/// vehicle points are children of its `integ_frame` (an
/// `EphemerisRefFrame`), not of any parent body. The dyn-parent
/// relationship is captured by the mass tree (and its
/// `MassChildOf` ECS-native sibling), independent of the frame tree.
/// JEOD only reparents these frames when the child's `integ_frame`
/// changes — and the same-integ-frame attach case here leaves the
/// integ frame unchanged.
///
/// In this same-integ-frame regime, reparenting the child's frame
/// entity under the parent body's frame entity would *invert* JEOD's
/// invariant: the child's `FrameTransC` is in the integration
/// frame's coordinates, and re-parenting it under the parent body's
/// frame entity would relabel that storage as "relative to the
/// parent body" without converting the numbers — silently corrupting
/// every downstream `RelativeFrameState` walk that reads the child's
/// frame entity to compute a cross-frame state. The
/// kinematic-propagation rewrite
/// (`propagate_state_from_root_system` + `sync_body_to_frame_system`)
/// keeps the child's `FrameTransC` in lockstep with its
/// `TranslationalStateC` each tick, but only as long as the child's
/// frame-tree node is parented under its own integ frame.
///
/// This regression test pins the no-op behaviour for the
/// same-integ-frame case so a future change that wires
/// `commands.entity(child_frame).insert(ChildOf(parent_frame))`
/// blindly into the attach handler would fail loudly.
///
/// **Out of scope for this regression**: the cross-integration-frame
/// case (parent and child carrying different `IntegSourceC` values),
/// where JEOD's `attach_establish_links` *does* call `set_integ_frame`
/// to reparent the child's frame tree under the parent's integ frame.
/// Our `staging_system` implements the matching reparent — and (unlike
/// JEOD's reparent-only `set_integ_frame`) also rewrites each
/// reparented body's stored `TranslationalStateC` / `FrameTransC` by
/// `(old_integ_origin - new_integ_origin)` in the same staging tick.
/// JEOD relies on its immediately-following `propagate_state` to refill
/// descendants' parent-relative storage, but our adapter has no
/// equivalent same-call propagation (the next tick's
/// `propagate_state_from_root_system` runs many systems later) and the
/// `TranslationalStateC`-is-already-integ-frame-relative storage
/// contract would otherwise leave every reparented descendant's
/// numerics inconsistent with the post-attach frame-tree topology for
/// every consumer in the staging→propagate window. The cross-integ
/// reparent + rewrite is exercised by the companion regressions
/// `bevy_parity_attach_detach_momentum_bevy_attach_cross_integ_frame_runs_combine_and_reparents_child_frame`
/// and `bevy_parity_attach_detach_momentum_bevy_attach_cross_integ_frame_rewrites_child_state_into_new_integ_frame`
/// below.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_does_not_reparent_child_frame_under_parent_frame()
{
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let (mut app, parent_entity, child_entity, _, _) = build_two_body_world(
        1.0,
        parent_mass,
        parent_trans,
        initial_rot,
        child_mass,
        child_trans,
        initial_rot,
    );

    // First step: `register_body_frames_system` spawns each body's
    // frame entity and parents it under the root frame entity. No
    // attach yet — both children are independent free-flying bodies.
    step(&mut app, 1, 1.0);

    let root_frame_entity = app.world().resource::<RootFrameEntityR>().0;

    // Sanity: both bodies registered, both frame entities under the
    // root frame entity.
    let parent_frame_entity = app
        .world()
        .get::<FrameEntityC>(parent_entity)
        .expect("parent registered FrameEntityC after first step")
        .0;
    let child_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("child registered FrameEntityC after first step")
        .0;
    assert_eq!(
        app.world()
            .get::<ChildOf>(parent_frame_entity)
            .expect("parent frame entity has ChildOf parent")
            .parent(),
        root_frame_entity,
        "pre-attach: parent frame entity must be parented under the root frame entity"
    );
    assert_eq!(
        app.world()
            .get::<ChildOf>(child_frame_entity)
            .expect("child frame entity has ChildOf parent")
            .parent(),
        root_frame_entity,
        "pre-attach: child frame entity must be parented under the root frame entity"
    );

    // Fire the attach event and step once so `staging_system`
    // processes it.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // The crux of the regression: the child's frame entity's ChildOf
    // parent must STILL be the root frame entity — `staging_system`
    // mutates the mass tree (and writes the merged composite-body
    // state into the parent), but it does not (and per JEOD must not)
    // reparent the child's frame entity under the parent body's frame
    // entity.
    let child_frame_parent_post = app
        .world()
        .get::<ChildOf>(child_frame_entity)
        .expect("child frame entity still alive post-attach")
        .parent();
    assert_eq!(
        child_frame_parent_post, root_frame_entity,
        "post-attach: child frame entity ({child_frame_entity:?}) must remain parented under the \
         root frame entity ({root_frame_entity:?}) — not reparented under the parent body's \
         frame entity ({parent_frame_entity:?}). JEOD `dyn_body_integration.cc::set_integ_frame` \
         only reparents body frames on integ-frame change, never on attach to a same-integ-frame \
         parent."
    );
    assert_ne!(
        child_frame_parent_post, parent_frame_entity,
        "post-attach: child frame entity must not be reparented under the parent body's frame \
         entity ({parent_frame_entity:?}); the parent-body relationship lives in the mass tree, \
         not in the frame tree."
    );

    // Pair the parentage check with a relative-state readback so a
    // future regression that keeps the `ChildOf` link correct but
    // silently corrupts the stored `FrameTransC` / `TranslationalStateC`
    // would also fail here. `RelativeFrameState::position(root, child)`
    // walks the frame tree using the same algorithm downstream
    // consumers use — if the child's frame-entity coordinates were
    // overwritten with parent-relative numbers (or zeroed by a stray
    // sync), this readback would diverge from the child entity's
    // `TranslationalStateC` (which is the integ-frame storage and, in
    // the same-integ-frame case, the same coordinate system as
    // `RelativeFrameState` returns).
    let child_trans_post = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .get::<TranslationalStateC<astrodyn::Earth>>(child_entity)
            .expect("child still has TranslationalStateC post-attach")
            .0,
    );
    let child_pos_via_frame_tree = app
        .world_mut()
        .run_system_cached_with(
            |In((from, to)): In<(Entity, Entity)>, rel: RelativeFrameState| -> DVec3 {
                rel.position(from, to)
            },
            (root_frame_entity, child_frame_entity),
        )
        .expect("RelativeFrameState run_system_cached_with");
    for i in 0..3 {
        assert_eq!(
            child_pos_via_frame_tree[i], child_trans_post.position[i],
            "post-attach: child's frame-entity position via RelativeFrameState (axis {i}) must \
             equal the child entity's TranslationalStateC — same-integ-frame attach must leave \
             both in root-inertial coordinates"
        );
    }

    // Companion check: the parent's own frame entity is also
    // unchanged (the attach is a child-side operation; the parent's
    // frame-tree node is identity-mapped through it).
    assert_eq!(
        app.world()
            .get::<ChildOf>(parent_frame_entity)
            .expect("parent frame entity has ChildOf parent post-attach")
            .parent(),
        root_frame_entity,
        "post-attach: parent frame entity must remain parented under the root frame entity"
    );
}

/// **Cross-integration-frame attach: positive parity.**
///
/// Spawn parent and child carrying **different** `IntegSourceC`
/// values so each body's frame entity is parented under a different
/// integration-frame entity. Source A and source B are placed at
/// distinct inertial positions so the two integ frames are not
/// numerically equivalent — a regression that reparents the child
/// under source A but forgets the integ-origin lift would produce a
/// merged composite that's wrong by `source_b - source_a`
/// (~2.7e8 m), well clear of the per-component tolerances below.
///
/// JEOD's `dyn_body_attach.cc::attach_establish_links` calls
/// `dyn_body_integration.cc::set_integ_frame` whenever the child's
/// `integ_frame` differs from the parent's: the child's primary
/// frames are reparented under the parent's integ frame and all
/// kinematic descendants follow recursively. JEOD's
/// `dyn_body_integration.cc::set_integ_frame` (lines 64-117) uses
/// the low-level `RefFrame::reset_parent` and explicitly does NOT
/// rewrite stored state ("It does not update state") — JEOD relies
/// on `attach_update_properties` calling `propagate_state()`
/// immediately afterwards to refill descendants' parent-relative
/// storage from the merged root. Our adapter has no equivalent
/// same-call propagation, so each reparented descendant's
/// `TranslationalStateC` and body-frame `FrameTransC` are shifted
/// in-place by `(old_origin - new_origin)` during the same staging
/// tick (same physical pose, just relabeled into the new integration
/// frame's coordinates) — exercised in detail by
/// `bevy_parity_attach_detach_momentum_bevy_attach_cross_integ_frame_rewrites_child_state_into_new_integ_frame`
/// further below. This test focuses on the parent's merged
/// composite-body state computed by `combine_states_at_attach`
/// (lifted to root inertial via each body's pre-attach
/// `IntegOrigin` so the cross-body kernel arithmetic — `omega × r`,
/// `T_inertial_struct.transpose()` shifts — operates on a single
/// inertial frame) and lowered through the parent's integ origin
/// for the writeback into the parent's `TranslationalStateC`
/// storage.
///
/// The expected merged state is computed by calling
/// `stage_attach_combine` directly with both bodies lifted to root
/// inertial through their integ-frame positions, then comparing the
/// kernel's output (lowered through the parent's integ origin)
/// component-wise against the parent's post-attach
/// `TranslationalStateC`. This is the same kernel-parity contract
/// `bevy_parity_attach_detach_momentum_bevy_attach_conserves_linear_and_angular_momentum` enforces for
/// the same-integ-frame case, generalised across non-zero integ
/// origins.
///
/// The frame-tree reparent is verified independently: post-attach,
/// the child's body-frame entity must have its `ChildOf` parent
/// equal to source A's frame entity (the parent's integ-frame
/// entity, JEOD's `set_integ_frame` semantics) — not source B's
/// frame entity (the child's pre-attach integ frame).
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_cross_integ_frame_runs_combine_and_reparents_child_frame(
) {
    let parent_mass = MassProperties::with_inertia(
        1000.0,
        DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        500.0,
        DMat3::from_diagonal(DVec3::new(200.0, 200.0, 200.0)),
        DVec3::ZERO,
    );
    // Per-frame integ-frame-relative initial states. Velocities are
    // distinct so the merge induces a mass-weighted velocity in the
    // composite (a soft co-mover would obscure whether the integ
    // origin was applied to the lift).
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(0.0, 7e6, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7600.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    // Two distinct gravity-source entities at distinct inertial
    // positions so each integ frame is structurally and numerically
    // separate. The fixture exercises the realistic asymmetric case
    // where the integ-origin shifts on the kernel inputs are large
    // and distinct — the same shape as a body in `Earth.inertial`
    // attaching to a body in `Moon.inertial`.
    let mu = 3.986004415e14_f64;
    let source_a_pos = DVec3::new(1.0e8, 0.0, 0.0);
    let source_b_pos = DVec3::new(0.0, 2.5e8, 0.0);
    let source_a = app
        .world_mut()
        .spawn((
            Name::new("SourceA"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                source_a_pos,
            )),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: source_a_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();
    let source_b = app
        .world_mut()
        .spawn((
            Name::new("SourceB"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                source_b_pos,
            )),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: source_b_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
            IntegSourceC(Some(source_a)),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
            IntegSourceC(Some(source_b)),
        ))
        .id();

    // Run startup so register_source_frames + register_body_frames
    // fire and the bodies' frame entities get parented under their
    // respective sources' integ frames before the attach event is
    // processed.
    app.world_mut().run_schedule(Startup);
    step(&mut app, 1, 1.0);

    // Resolve the body-frame entities + each source's frame entity.
    // The reparent assertion below checks the child's body-frame
    // entity moves from `source_b.frame.inertial` to
    // `source_a.frame.inertial` post-attach.
    let source_a_frame = app
        .world()
        .get::<FrameEntityC>(source_a)
        .expect("source A registered FrameEntityC")
        .0;
    let source_b_frame = app
        .world()
        .get::<FrameEntityC>(source_b)
        .expect("source B registered FrameEntityC")
        .0;
    let child_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("child registered FrameEntityC")
        .0;
    assert_eq!(
        app.world()
            .get::<ChildOf>(child_frame_entity)
            .expect("child body-frame has ChildOf")
            .parent(),
        source_b_frame,
        "fixture sanity: child body-frame entity must be ChildOf source B's frame entity \
         pre-attach (live integ-frame source of truth)",
    );

    // Fire the attach. The cross-integ-frame branch runs the
    // combine kernel with both bodies lifted to root inertial via
    // their pre-attach integ origins, then lowers the merged
    // composite through the parent's integ origin for the writeback
    // into the parent's `TranslationalStateC`.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // Verify the child's body-frame entity is now ChildOf the
    // parent's integ-frame entity (source A's frame entity), not
    // source B's. JEOD's `set_integ_frame` reparent-only contract.
    assert_eq!(
        app.world()
            .get::<ChildOf>(child_frame_entity)
            .expect("child body-frame has ChildOf post-attach")
            .parent(),
        source_a_frame,
        "post-attach: child body-frame entity must be reparented under source A's frame \
         entity (the parent's integ-frame entity, JEOD `set_integ_frame` semantics) — \
         not under source B's frame entity (its pre-attach integ frame)",
    );
    assert_ne!(
        app.world()
            .get::<ChildOf>(child_frame_entity)
            .unwrap()
            .parent(),
        source_b_frame,
        "post-attach: reparent must not leave the child under its pre-attach integ frame",
    );

    // Independent expected merged state: lift both bodies to root
    // inertial through their integ origins, run the combine kernel,
    // then lower through the parent's integ origin so the result
    // can be compared against the parent's `TranslationalStateC`
    // (which stores the body in integ-frame coordinates). Mirrors
    // the same lift/lower the runner's `mass_tree::attach_inner`
    // does at the kernel boundary.
    let combined_mass = app
        .world()
        .resource::<MassTreeR>()
        .0
        .get(id_a)
        .composite_properties;
    let q = JeodQuat::identity();
    let parent_position_root = parent_trans.position + source_a_pos;
    let parent_velocity_root = parent_trans.velocity;
    let child_position_root = child_trans.position + source_b_pos;
    let child_velocity_root = child_trans.velocity;
    let expected_root = astrodyn::stage_attach_combine(StageAttachInputs {
        parent_position: parent_position_root,
        parent_velocity: parent_velocity_root,
        parent_quaternion: q,
        parent_ang_vel_body: DVec3::ZERO,
        parent_mass,
        orig_parent_cm_struct: parent_mass.position,
        parent_t_inertial_struct: DMat3::IDENTITY,
        child_position: child_position_root,
        child_velocity: child_velocity_root,
        child_quaternion: q,
        child_ang_vel_body: DVec3::ZERO,
        child_mass,
        combined_mass,
    });
    let expected_position_in_a = expected_root.position - source_a_pos;
    let expected_velocity_in_a = expected_root.velocity;

    let pos = read_position(app.world(), parent_entity);
    let vel = read_velocity(app.world(), parent_entity);
    let omega = read_ang_vel(app.world(), parent_entity);
    assert!(
        (pos - expected_position_in_a).length() < 1e-9,
        "post-attach position (in source A's integ frame): bevy {pos:?} vs expected \
         {expected_position_in_a:?}",
    );
    assert!(
        (vel - expected_velocity_in_a).length() < 1e-9,
        "post-attach velocity (root-inertial = source A's frame velocity since source A \
         is stationary): bevy {vel:?} vs expected {expected_velocity_in_a:?}",
    );
    assert!(
        (omega - expected_root.ang_vel_body).length() < 1e-9,
        "post-attach ang_vel: bevy {omega:?} vs expected {:?}",
        expected_root.ang_vel_body,
    );

    // Verify that reading the parent's body-frame entity through
    // `RelativeFrameState::position(root, parent_frame)` gives the
    // root-inertial absolute position of the merged composite —
    // i.e. the same value `expected_root.position` produced by the
    // kernel. This pins that the integ-origin lift on the writeback
    // is consistent with the frame-tree's interpretation of the
    // parent's `TranslationalStateC` as integ-frame-relative.
    //
    // The frame-tree walk reads `FrameTransC` on the parent's
    // body-frame entity. `staging_system` writes the merged state
    // into `TranslationalStateC`; `sync_body_to_frame_system`
    // mirrors that into `FrameTransC` later in the same tick (in
    // `AstrodynSet::Integration`). The first `step()` call above
    // already covered both, so the walk below sees the post-merge
    // value. A regression that mismatches the lift / lower (e.g.
    // forgets to lower the result through the parent's integ
    // origin) would produce an `abs_pos` off by `source_a_pos` —
    // ~1e8 m, well clear of the f64-rounding tolerance below.
    let root_e = app.world().resource::<RootFrameEntityR>().0;
    let parent_frame_entity = app
        .world()
        .get::<FrameEntityC>(parent_entity)
        .expect("parent registered FrameEntityC")
        .0;
    // The parent in this fixture has no `GravityControlsC`, which
    // is a required (non-optional) component on
    // `integration_system`'s body query — so the parent does not
    // match the integrator and is *not* advanced at all. At t=dt the
    // parent's `TranslationalStateC` therefore still holds the
    // merged-composite seed `staging_system` wrote (no `vel · dt`
    // advance). The drift bound below is kept generous — `2 · vel ·
    // dt` — so the assertion would still hold under a future
    // refactor that moved the parent into the integrator query
    // (which would then advance it under no-force kinematics from
    // the same seed); the parity check only needs the integ-origin
    // shift to be applied correctly on the writeback. A regression
    // that mismatches the lift / lower (e.g. forgets to lower the
    // result through the parent's integ origin) would produce an
    // `abs_pos` off by `source_a_pos` ~1e8 m, well clear of the
    // `2 · vel_max · dt` ~ 1.5e4 m tolerance below.
    let abs_pos = app
        .world_mut()
        .run_system_cached_with(
            move |In((root, frame)): In<(Entity, Entity)>, rel: RelativeFrameState| {
                rel.position(root, frame)
            },
            (root_e, parent_frame_entity),
        )
        .expect("RelativeFrameState position lookup");
    let expected_abs_pos = expected_position_in_a + source_a_pos;
    let dt = 1.0;
    let drift_bound = 2.0 * expected_velocity_in_a.length() * dt;
    assert!(
        (abs_pos - expected_abs_pos).length() < drift_bound + 1e-3,
        "post-attach absolute position via RelativeFrameState: bevy {abs_pos:?} \
         vs expected {expected_abs_pos:?} (allowed drift {drift_bound:.3} m) — the \
         integ-origin lift on the writeback must be consistent with the \
         frame-tree's interpretation of the body's TranslationalStateC as \
         integ-frame-relative; a mismatched lift / lower would produce a \
         ~{:.3e} m discrepancy ≫ tolerance",
        (source_b_pos - source_a_pos).length(),
    );

    // Sanity: the mass-tree attach landed (the `unwrap` proves it).
    let tree = &app.world().resource::<MassTreeR>().0;
    assert_eq!(
        tree.parent(id_b),
        Some(id_a),
        "post-attach: child's mass-tree parent must be the parent body",
    );
    // Sanity: combined mass on the parent matches the tree.
    assert!(
        (read_mass(app.world(), parent_entity) - combined_mass.mass).abs() < 1e-12,
        "composite mass on parent should match the tree's post-attach value",
    );
}

/// **Cross-integ-frame attach: child's stored coordinates land in
/// the new integ frame within the staging tick.**
///
/// `register_body_frames_system`'s docstring fixes the storage
/// contract: a body's `TranslationalStateC` is interpreted as already
/// in integ-frame coordinates, where "integ frame" is the body-frame
/// entity's current `ChildOf` parent. The cross-integ-frame attach
/// branch in `staging_system` reparents the child's body-frame
/// entity under the parent's integ-frame entity — that flips the
/// "integ frame" interpretation from old to new — so the stored
/// numerics must shift by `(old_origin - new_origin)` in
/// root-inertial coordinates to remain consistent. Without that
/// shift, every consumer that reads the child's stored state
/// between `staging_system` and the next
/// `propagate_state_from_root_system` (the entire `Interaction` set
/// — drag, gravity-torque, SRP — plus `force_collection_system` at
/// the top of `ForceCollection`) reads pre-attach numerics through
/// post-attach topology and silently mixes coordinates across
/// distinct integration frames.
///
/// This test pins that the rewrite happens by reading the child's
/// stored state through TWO independent channels after a single
/// step that drains the attach event:
///
/// * **`TranslationalStateC` as integ-frame coords**: post-attach
///   the child's body-frame entity is reparented under source A's
///   frame entity (the parent's integ frame), so the child's stored
///   `TranslationalStateC` is interpreted as source-A-relative.
///   `child.position + source_a_origin == merged_root_position +
///   link_offset_in_root` post-step (the `RootFrameEntityR` itself
///   is unchanged — it remains the inertial root). A regression that
///   skipped the numerical rewrite would produce
///   `child.position + source_a_origin == old_root_position +
///   link_offset_in_root - (source_a - source_b)` ≠ the merged
///   value, off by `~|source_a - source_b|` ≈ 2.5e8 m, four orders of
///   magnitude wider than the propagation tolerance below.
///
/// * **`FrameTransC` on the child's body-frame entity**: the
///   reparent + state rewrite go through the same `Commands::insert`
///   batch in `staging_system`, so post-flush
///   `FrameTransC.position` reflects the child's pre-attach state
///   shifted into the new parent frame's coordinates. The frame
///   tree's `RelativeFrameState` walk reads `FrameTransC` directly,
///   so the child's `RelativeFrameState::position(root,
///   child_frame_entity)` must equal the absolute root-inertial
///   position of the child's body. The same regression would put
///   the child's body-frame `FrameTransC` at its pre-attach value
///   in the wrong parent's coordinates, producing the same large
///   discrepancy under the walk.
///
/// The test deliberately uses sources with non-zero, distinct
/// inertial positions — both shifts are large and asymmetric, so
/// the regression scale dominates any f64-rounding noise.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_cross_integ_frame_rewrites_child_state_into_new_integ_frame(
) {
    let parent_mass = MassProperties::with_inertia(
        1000.0,
        DMat3::from_diagonal(DVec3::new(500.0, 500.0, 500.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        500.0,
        DMat3::from_diagonal(DVec3::new(200.0, 200.0, 200.0)),
        DVec3::ZERO,
    );
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(0.0, 7e6, 0.0),
        velocity: DVec3::new(0.0, 0.0, 7600.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let mu = 3.986004415e14_f64;
    let source_a_pos = DVec3::new(1.0e8, 0.0, 0.0);
    let source_b_pos = DVec3::new(0.0, 2.5e8, 0.0);
    let source_a = app
        .world_mut()
        .spawn((
            Name::new("SourceA"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                source_a_pos,
            )),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: source_a_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();
    let source_b = app
        .world_mut()
        .spawn((
            Name::new("SourceB"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                source_b_pos,
            )),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: source_b_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
            IntegSourceC(Some(source_a)),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
            IntegSourceC(Some(source_b)),
        ))
        .id();

    app.world_mut().run_schedule(Startup);
    step(&mut app, 1, 1.0);

    // The child's pre-attach absolute root-inertial position. The
    // attach event itself is a topology change at a frozen instant
    // — the child's physical location in space does not move when
    // the integ-frame interpretation flips from source B to source
    // A. So the post-attach root-inertial position of the child's
    // body is its pre-attach absolute value, plus at most one
    // tick's velocity drift from the integrator advancing the
    // (merged) parent under no-force kinematics. The staging-time
    // rewrite must produce stored coords whose interpretation
    // through the new integ frame round-trips to this same
    // root-inertial value.
    let child_pre_attach_root_position = child_trans.position + source_b_pos;
    let child_pre_attach_root_velocity = child_trans.velocity;

    // Fire the cross-integ-frame attach.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // ── Channel 1: child's TranslationalStateC, interpreted as
    //    new-integ-frame-relative per `register_body_frames_system`. ──
    //
    // Root-inertial absolute position = `stored + new_integ_origin`.
    // After the staging-time rewrite, the stored value is the
    // pre-attach value shifted by `(old_origin - new_origin)`, so
    // this round-trips to the child's pre-attach absolute root
    // position. On the SAME tick as the attach, the wrench system
    // hasn't yet inserted `KinematicChildC` (it runs *after* the
    // first propagate pass), so propagate skips the marker-gated
    // writeback and the child's `TranslationalStateC` keeps the
    // staging-time-rewritten value through the rest of the tick.
    // A regression that skipped the staging-time rewrite would
    // leave `stored == old-frame value`, so reading via
    // `stored + new_origin` would land at
    // `old-frame value + new_origin = pre-attach absolute -
    // (old_origin - new_origin)` — off by ~|source_a - source_b|
    // ≈ 2.7e8 m, four orders larger than the f64-rounding
    // tolerance below.
    let child_post_position_in_new_frame = read_position(app.world(), child_entity);
    let child_abs_position = child_post_position_in_new_frame + source_a_pos;
    let pos_err_root = (child_abs_position - child_pre_attach_root_position).length();
    assert!(
        pos_err_root < 1e-6,
        "post-attach child position (interpreted as new-integ-frame coords): got root \
         absolute {child_abs_position:?} vs expected {expected:?}. A regression that skipped \
         the staging-time numerical rewrite would land off by ~{regression_scale:.3e} m.",
        expected = child_pre_attach_root_position,
        regression_scale = (source_a_pos - source_b_pos).length(),
    );

    let child_post_velocity_in_new_frame = read_velocity(app.world(), child_entity);
    let vel_err_root = (child_post_velocity_in_new_frame - child_pre_attach_root_velocity).length();
    assert!(
        vel_err_root < 1e-9,
        "post-attach child velocity (interpreted as new-integ-frame coords, both source \
         frames stationary so the lift contributes zero velocity offset): got \
         {child_post_velocity_in_new_frame:?} vs expected {expected:?}. A regression that \
         skipped the staging-time rewrite would leave the velocity in the old frame's \
         numerical convention.",
        expected = child_pre_attach_root_velocity,
    );

    // ── Channel 2: child's body-frame entity FrameTransC under
    //    the new parent (the reparent target). Read via
    //    `RelativeFrameState::position(root, child_frame_entity)`
    //    so the walk goes through the post-reparent topology. ──
    //
    // This consumer doesn't use `TranslationalStateC` at all; it
    // walks the frame tree directly. Without the staging-time
    // `FrameTransC` rewrite the child's frame entity would still
    // hold its pre-attach value (in source B's coordinates) but
    // be parented under source A's frame entity, producing a
    // discontinuity ≈ |source_a - source_b| in the absolute
    // position. After the rewrite, the FrameTransC has been
    // shifted into source A's coordinates so the walk reproduces
    // the post-step body position. `sync_body_to_frame_system`
    // (in `Integration`, after staging) re-syncs `FrameTransC`
    // from the (still-rewritten) `TranslationalStateC` later in
    // the same tick, so the round-trip stays consistent.
    let root_e = app.world().resource::<RootFrameEntityR>().0;
    let child_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("child registered FrameEntityC")
        .0;
    let child_abs_via_walk = app
        .world_mut()
        .run_system_cached_with(
            move |In((root, frame)): In<(Entity, Entity)>, rel: RelativeFrameState| {
                rel.position(root, frame)
            },
            (root_e, child_frame_entity),
        )
        .expect("RelativeFrameState position lookup");
    let walk_err = (child_abs_via_walk - child_pre_attach_root_position).length();
    assert!(
        walk_err < 1e-6,
        "post-attach child absolute position via RelativeFrameState walk: got \
         {child_abs_via_walk:?} vs expected {expected:?}. A regression that skipped the \
         staging-time FrameTransC rewrite would walk through a ChildOf-mismatched \
         FrameTransC, off by ~{regression_scale:.3e} m.",
        expected = child_pre_attach_root_position,
        regression_scale = (source_a_pos - source_b_pos).length(),
    );

    // Sanity: the body-frame entity is structurally under the new
    // parent (the actual cross-integ-frame reparent target), not
    // its pre-attach integ frame.
    let source_a_frame = app
        .world()
        .get::<FrameEntityC>(source_a)
        .expect("source A registered FrameEntityC")
        .0;
    assert_eq!(
        app.world()
            .get::<ChildOf>(child_frame_entity)
            .expect("child body-frame has ChildOf post-attach")
            .parent(),
        source_a_frame,
        "post-attach: child body-frame entity must be reparented under source A's frame entity \
         (the parent's integ frame, JEOD set_integ_frame semantics)",
    );
}

/// **Same-integ-frame attach after a frame switch: must NOT panic.**
///
/// `frame_switch_system` mutates a body's `ChildOf` parent on every
/// switch but intentionally leaves `IntegSourceC` (the config-time
/// intent) stale. The cross-integ-frame guard in `staging_system`
/// must therefore consult the body-frame entity's `ChildOf` parent
/// — the live integ-frame source of truth — rather than the body's
/// `IntegSourceC` component.
///
/// This test pins that policy: spawn parent and child with
/// **different** `IntegSourceC` values (parent: `Some(source)`,
/// child: `None`), then simulate a post-frame-switch state by
/// reparenting the child's body-frame entity under the same
/// integration-frame entity that the parent's body-frame entity
/// lives under (and rewriting its stored `TranslationalStateC` /
/// `FrameTransC` into that frame, exactly like
/// `frame_switch_system` does on a real switch). Both bodies are
/// now in the same integ frame structurally; the attach must
/// proceed.
///
/// A guard that compares `IntegSourceC` directly (the previous
/// implementation) would falsely reject this attach — the
/// `IntegSourceC` values differ. A guard that compares the
/// body-frame entities' `ChildOf` parents — what the live integ
/// frame *actually is* — accepts it.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_post_frame_switch_same_integ_frame_succeeds() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    // Child starts root-coordinate-relative; we'll rewrite it into
    // source-relative coordinates below to mirror what
    // `frame_switch_system` does on a real switch.
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    // Single gravity source at a non-zero position so the
    // "child starts root-relative, gets rewritten into source-relative"
    // step actually changes coordinate values (not a no-op).
    let mu = 3.986004415e14_f64;
    let source_pos = DVec3::new(1.0e8, 0.0, 0.0);
    let source = app
        .world_mut()
        .spawn((
            Name::new("Source"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                source_pos,
            )),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: source_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();

    // Parent: configured to live in the source's integ frame from
    // the start (`IntegSourceC(Some(source))`). Its body-frame entity
    // gets parented under `source`'s frame entity by
    // `register_body_frames_system` at startup.
    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
            IntegSourceC(Some(source)),
        ))
        .id();
    // Child: configured root-integrated (`IntegSourceC(None)`) to
    // start. Its body-frame entity gets parented under the root
    // frame entity at startup.
    let child_root_relative_pos = DVec3::new(7e6 + source_pos.x, 1.0, 0.0);
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: child_root_relative_pos,
                velocity: DVec3::new(0.0, 7600.0, 0.0),
            }),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
            IntegSourceC(None),
        ))
        .id();

    // Run startup so registration parents each body-frame entity
    // under the appropriate integ frame entity.
    app.world_mut().run_schedule(Startup);
    step(&mut app, 1, 1.0);

    // Resolve the source's frame entity (the live integ frame for
    // the parent post-startup) and the child's body-frame entity.
    let source_frame_entity = app
        .world()
        .get::<FrameEntityC>(source)
        .expect("source registered FrameEntityC")
        .0;
    let child_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("child registered FrameEntityC")
        .0;

    // Simulate a post-frame-switch state on the child: reparent its
    // body-frame entity under the source's frame entity and rewrite
    // both `TranslationalStateC` and `FrameTransC` into source-
    // relative coordinates. This is the exact mutation
    // `frame_switch_system` performs on a real switch — and crucially
    // it leaves `IntegSourceC` unchanged.
    let child_source_relative_pos = child_root_relative_pos - source_pos;
    {
        let world = app.world_mut();
        world
            .entity_mut(child_frame_entity)
            .insert(ChildOf(source_frame_entity))
            .insert(FrameTransC {
                position: child_source_relative_pos,
                velocity: DVec3::new(0.0, 7600.0, 0.0),
            });
        let mut child_trans = world
            .get_mut::<TranslationalStateC<astrodyn::Earth>>(child_entity)
            .expect("child still has TranslationalStateC");
        child_trans.0.position =
            astrodyn::Position::<astrodyn::PlanetInertial<astrodyn::Earth>>::from_raw_si(
                child_source_relative_pos,
            );
    }

    // Sanity-check the asymmetric setup: `IntegSourceC` values differ
    // (parent: Some(source); child: None — stale post-"switch") but
    // the body-frame entities now share the same `ChildOf` parent.
    assert_ne!(
        app.world().get::<IntegSourceC>(parent_entity).unwrap().0,
        app.world().get::<IntegSourceC>(child_entity).unwrap().0,
        "fixture sanity: IntegSourceC values must differ — that is the whole point of this test"
    );
    let parent_frame_entity = app
        .world()
        .get::<FrameEntityC>(parent_entity)
        .expect("parent registered FrameEntityC")
        .0;
    assert_eq!(
        app.world()
            .get::<ChildOf>(parent_frame_entity)
            .expect("parent body-frame has ChildOf parent")
            .parent(),
        source_frame_entity,
        "fixture sanity: parent body-frame entity must be ChildOf the source frame entity"
    );
    assert_eq!(
        app.world()
            .get::<ChildOf>(child_frame_entity)
            .expect("child body-frame has ChildOf parent")
            .parent(),
        source_frame_entity,
        "fixture sanity: child body-frame entity must be ChildOf the source frame entity \
         (post-simulated-switch)"
    );

    // Fire the attach. The new `ChildOf`-based check sees both bodies
    // in the same integ frame and lets the merge proceed; the old
    // `IntegSourceC` check would have falsely rejected this attach.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // Verify the attach actually happened: the mass tree now has
    // `id_b` as a child of `id_a`, and the child carries no
    // `DetachedSubtreeStateC` (it was attached, not freed).
    let tree = &app.world().resource::<MassTreeR>().0;
    assert_eq!(
        tree.parent(id_b),
        Some(id_a),
        "post-attach: child's mass-tree parent must be the parent body — the attach proceeded"
    );
    assert!(
        app.world()
            .get::<DetachedSubtreeStateC>(child_entity)
            .is_none(),
        "post-attach: child must not carry DetachedSubtreeStateC"
    );
}

/// Detached subtrees coast ballistically (no force, no torque). The
/// runner's split between `Simulation::bodies` and
/// `Simulation::detached_subtrees` only evaluates gravity, drag, SRP,
/// gravity torque, and force collection on the integrated set —
/// detached entries are not part of any wrench-aggregation walk.
///
/// `gravity_computation_system`, `aero_drag_system`,
/// `gravity_torque_system`, the SRP systems, and `force_collection_system`
/// must therefore skip detached bodies. Otherwise
/// `GravityAccelerationC`, `AerodynamicForceC`, `RadiationForceC`, and
/// `TotalForceC` populate with values no integrator consumes, exposing
/// stale or misleading readings to diagnostics / logging consumers.
///
/// This test pins the per-step components to zero on a detached body
/// even when a gravity source is in range that would otherwise produce
/// a non-trivial acceleration / total force.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_detached_body_skips_force_pipeline() {
    // Build a minimal world with a gravity source at the origin so a
    // free-flying body at 7e6 m would otherwise see ~µ/r² gravity.
    let mu = 3.986004415e14_f64;
    let body_mass = MassProperties::new(1000.0);
    let initial_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    // Mass tree (required for attach/detach).
    let mut tree = MassTree::new();
    let id_body = tree.add_body("Body".into(), body_mass);
    app.insert_resource(MassTreeR(tree));

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Planet"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    let body_entity = app
        .world_mut()
        .spawn((
            Name::new("Body"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(initial_trans),
            RotationalStateC::default(),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(body_mass)),
            ),
            MassBodyIdC(id_body),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, GravityGradient::Skip)],
            }),
        ))
        .id();

    // Step once *without* detaching: confirm the gravity pipeline does
    // populate `GravityAccelerationC` and `TotalForceC` with non-trivial
    // values when the body is integrated. This pins the precondition —
    // if the gravity pipeline was a no-op for some other reason the
    // detached-body assertions below would be trivially satisfied.
    step(&mut app, 1, 1.0);

    let attached_grav = app
        .world()
        .get::<GravityAccelerationC>(body_entity)
        .unwrap()
        .grav_accel
        .raw_si();
    let attached_trans_accel = app
        .world()
        .get::<FrameDerivativesC>(body_entity)
        .unwrap()
        .0
        .trans_accel
        .raw_si();
    assert!(
        attached_grav.length() > 1.0,
        "precondition: integrated body should see non-trivial gravity \
         (attached_grav={attached_grav:?})"
    );
    assert!(
        attached_trans_accel.length() > 1.0,
        "precondition: integrated body should have non-trivial translational \
         acceleration in FrameDerivativesC \
         (attached_trans_accel={attached_trans_accel:?})"
    );

    // Promote the body into a detached subtree by inserting
    // `DetachedSubtreeStateC` directly. Going through `DetachEvent`
    // would require a parent attach first; the direct insert
    // exercises the same downstream filter the detach handler
    // ultimately produces.
    use astrodyn::{BodyAttitude, DetachedSubtreeState, RootInertial, SelfRef, Vec3Ext};
    let detached_state = DetachedSubtreeState {
        composite_position: initial_trans.position.m_at::<RootInertial>(),
        composite_velocity: initial_trans.velocity.m_per_s_at::<RootInertial>(),
        composite_attitude: BodyAttitude::<SelfRef>::identity(),
        composite_ang_vel_body: DVec3::ZERO,
    };
    app.world_mut()
        .entity_mut(body_entity)
        .insert(DetachedSubtreeStateC(detached_state));

    // Zero the per-step force outputs by hand so we can observe
    // whether the next step writes anything new.
    {
        let mut grav = app
            .world_mut()
            .get_mut::<GravityAccelerationC>(body_entity)
            .unwrap();
        *grav = GravityAccelerationC::default();
    }
    {
        let mut derivs = app
            .world_mut()
            .get_mut::<FrameDerivativesC>(body_entity)
            .unwrap();
        *derivs = FrameDerivativesC::default();
    }

    // Step again. With the detached filter in place, none of
    // `gravity_computation_system`, `force_collection_system`, or
    // the interaction systems should touch the detached body's
    // force-pipeline components.
    step(&mut app, 1, 1.0);

    let detached_grav = app
        .world()
        .get::<GravityAccelerationC>(body_entity)
        .unwrap()
        .grav_accel
        .raw_si();
    let detached_trans_accel = app
        .world()
        .get::<FrameDerivativesC>(body_entity)
        .unwrap()
        .0
        .trans_accel
        .raw_si();
    assert_eq!(
        detached_grav,
        DVec3::ZERO,
        "GravityAccelerationC must stay zero on a detached body \
         (gravity_computation_system must filter Without<DetachedSubtreeStateC>)"
    );
    assert_eq!(
        detached_trans_accel,
        DVec3::ZERO,
        "FrameDerivativesC.trans_accel must stay zero on a detached body \
         (force_collection_system must filter Without<DetachedSubtreeStateC>)"
    );
}

/// Regression: the detach handler must read the parent's
/// pre-detach composite-CoM from the live `MassTreeR` arena, not
/// from `MassPropertiesC`. The ECS-tree fast path in
/// `composite_mass_system` runs **before** `staging_system` in the
/// same FixedUpdate tick (and on every subsequent tick the parent
/// has no `MassChildOf` edge), and reverts the parent's
/// `MassPropertiesC` to its `CoreMassPropertiesC` cache. Reading
/// `parent_pre_composite_props` from `MassPropertiesC` after that
/// revert pulls the parent's *core* mass props (specifically
/// `position` = core CoM, typically zero) instead of the live
/// post-attach composite CoM, corrupting the CoM-shift formula
/// `parent_pre.position − parent_post.position` and leaving the
/// parent's post-detach inertial position equal to its pre-detach
/// inertial position (zero shift) instead of the JEOD-faithful
/// `−Δ_composite_struct` shift.
///
/// This test exercises that exact race with a non-trivial CoM
/// offset between attach and detach: `parent.core.position = 0`,
/// `child.position` offset by `(3, 0, 0)` along the structure
/// frame. The merged composite CoM shifts to
/// `(m_c · 3) / (m_p + m_c) = 80 · 3 / 500 = 0.48 m`. After detach,
/// parent's composite CoM returns to zero. The parent's post-detach
/// inertial position must therefore shift by
/// `(0 − 0.48) m = −0.48 m` along x relative to the merged
/// (post-attach) composite-body inertial position — JEOD's
/// composite-CoM-tracks-struct invariant. Without the live-arena
/// read in the detach handler, the shift is computed as
/// `(0 − 0) m = 0` and the parent's post-detach position is wrong
/// by `0.48 m`.
///
/// One step suffices: the bug surfaces in tick 2 — the first tick
/// after attach, when `composite_mass_system` first reverts the
/// parent's `MassPropertiesC` to core. No multi-step propagation
/// needed.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_detach_reads_live_composite_through_mass_property_revert(
) {
    let parent_mass = MassProperties::with_inertia(
        420.0,
        DMat3::from_diagonal(DVec3::new(150.0, 200.0, 250.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        80.0,
        DMat3::from_diagonal(DVec3::new(40.0, 50.0, 60.0)),
        DVec3::ZERO,
    );
    // Co-moving: identical velocity → no induced spin in the merge.
    // Pure CoM-tracking case so the only thing the post-detach shift
    // depends on is the composite-CoM delta in struct frame.
    let v0 = DVec3::new(0.0, 7600.0, 0.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: v0,
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6 + 3.0, 0.0, 0.0),
        velocity: v0,
    };
    let parent_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    let child_rot = parent_rot;
    let offset = DVec3::new(3.0, 0.0, 0.0);
    let dt = 1.0;

    let (mut app, parent_entity, child_entity, _, _) = build_two_body_world(
        dt,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );

    // Attach + step. The attach branch writes the merged composite
    // into the parent's `TranslationalStateC` (post-attach inertial
    // position = parent_pre.position + cm_delta_inertial).
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(offset),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, dt);

    let parent_pos_post_attach = read_position(app.world(), parent_entity);
    // Soft co-moving merge: cm_delta_struct = combined.position − 0
    //   = (m_c · offset) / total = 80 · 3 / 500 = 0.48 m along x.
    // With identity attitude, struct == body == inertial here.
    let cm_delta_attach = DVec3::new(80.0 * 3.0 / 500.0, 0.0, 0.0);
    let expected_parent_pos_post_attach = parent_trans.position + cm_delta_attach;
    assert!(
        (parent_pos_post_attach - expected_parent_pos_post_attach).length() < 1e-9,
        "precondition: post-attach parent position must follow combined-CoM \
         shift: bevy={parent_pos_post_attach:?} expected={expected_parent_pos_post_attach:?}"
    );

    // Detach + step. This is the tick where `composite_mass_system`
    // reverts the parent's `MassPropertiesC` to core BEFORE
    // `staging_system` runs. The detach handler must still see the
    // live composite (cm = 0.48 m struct) — read from the arena —
    // not the reverted core (cm = 0). The post-detach inertial
    // position shifts by `−cm_delta_attach` from the post-attach
    // value. There is no integrator state on the parent in this
    // minimal harness, so `integration_system` does not advance
    // `TranslationalStateC` — the CoM-shift is the only mutation.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent {
            child: child_entity,
        });
    step(&mut app, 1, dt);

    let parent_pos_post_detach = read_position(app.world(), parent_entity);
    // The parent has no integrator state in this minimal harness so
    // `integration_system` does not advance `TranslationalStateC` —
    // the only mutation between attach and detach is the staging
    // handler's CoM-shift writeback. Expected value: post-attach
    // position - cm_delta_attach == original pre-attach position.
    let expected_parent_pos_post_detach = parent_trans.position;
    assert!(
        (parent_pos_post_detach - expected_parent_pos_post_detach).length() < 1e-9,
        "post-detach parent position must shift by −cm_delta_struct \
         (computed against the live arena composite, not the reverted \
         MassPropertiesC core): bevy={parent_pos_post_detach:?} expected={expected_parent_pos_post_detach:?}\n\
         If this fails by ~{}m along x, `staging_system`'s detach handler is \
         reading `parent_pre_composite_props` from `MassPropertiesC` (which \
         `composite_mass_system` reverted to core in this tick) instead of \
         from `tree.get(tree_root_id).composite_properties`.",
        cm_delta_attach.x,
    );
}

// ════════════════════════════════════════════════════════════════════
// Cross-runtime parity (sub-issue #297)
// ════════════════════════════════════════════════════════════════════
//
// The Bevy adapter's `staging_system` and `astrodyn_runner::Simulation`'s
// `attach` / `detach` are both thin orchestrators around the
// `astrodyn_dynamics` kernel. Both adapter paths must produce
// bit-identical post-attach / post-detach state on the parent and
// child for the same input. Any drift indicates a snapshot/writeback
// asymmetry between the two adapters.

/// Parent + child attach + detach via the runner's
/// `Simulation::attach` / `Simulation::detach` produces the same
/// composite-body state as the Bevy adapter's `staging_system` for
/// the same scenario. Bit-identical to `to_bits()`.
///
/// Scope guard: this test pins the *value the kernel writes* in each
/// adapter. It does NOT compare states after subsequent
/// integration / `step_detached_system` advance — those use
/// different per-tick advancement formulas (Bevy's
/// `BodyAttitude::advance_under_body_rate` exact rotation vs the
/// runner's RK4 quaternion stage), so a tick-by-tick attitude
/// comparison would diverge by integrator floor even when both
/// adapters' kernel output is bit-identical.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_runner_parity_attach_detach_momentum() {
    use astrodyn::{
        GravityControl as RunnerGravityControl, GravityControls as RunnerGravityControls,
        GravityModel as RunnerGravityModel, GravitySource as RunnerGravitySource,
        GravitySourceEntry as RunnerGravitySourceEntry, IntegratorType as RunnerIntegratorType,
        SimulationTime as RunnerSimulationTime, VehicleConfig as RunnerVehicleConfig,
    };
    use astrodyn_runner::Simulation;

    // Identical scenario for both runtimes — same masses, same
    // positions, same velocities, same offset / rotation. The bodies
    // are co-moving (child.velocity == parent.velocity) so the
    // post-attach kernel produces zero ω, which lets the Bevy side's
    // schedule (which runs `staging_system`, then
    // `step_detached_system`, then the integration_system within one
    // FixedUpdate) avoid drifting the parent's quaternion / the
    // detached child's quaternion away from identity. The runner does
    // *not* `step()` after attach in this test (we are comparing the
    // value the kernel writes, not the integrator's tick), so any
    // non-zero post-attach ω would leave Bevy and runner attitudes
    // diverged by exactly one integration step — a different
    // arithmetic floor than the kernel's writeback.
    let parent_mass = MassProperties::with_inertia(
        420.0,
        DMat3::from_diagonal(DVec3::new(150.0, 200.0, 250.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        80.0,
        DMat3::from_diagonal(DVec3::new(40.0, 50.0, 60.0)),
        DVec3::ZERO,
    );
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let parent_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6 + 3.0, 0.0, 0.0),
        // Co-moving: identical velocity to the parent → no induced
        // spin. The merge is a soft co-mover, but the CoM still
        // shifts because the child is structurally offset from the
        // parent — so the parent's post-attach inertial position
        // moves by `cm_delta_inertial` even though no momentum is
        // exchanged. That's the orchestration we want to pin against
        // the Bevy adapter byte-for-byte.
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    let offset = DVec3::new(3.0, 0.0, 0.0);
    let t_parent_child = DMat3::IDENTITY;
    let dt = 1.0;

    // ── Bevy path ──────────────────────────────────────────────────
    let (mut app, parent_entity, child_entity, _, _) = build_two_body_world(
        dt,
        parent_mass,
        parent_trans,
        parent_rot,
        child_mass,
        child_trans,
        child_rot,
    );
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(offset),
            t_parent_child: astrodyn::FrameTransform::from_matrix(t_parent_child),
        });
    step(&mut app, 1, dt);

    let bevy_parent_pos = read_position(app.world(), parent_entity);
    let bevy_parent_vel = read_velocity(app.world(), parent_entity);
    let bevy_parent_q = astrodyn::typed_bridge::rot_typed_to_raw(
        &app.world()
            .get::<RotationalStateC>(parent_entity)
            .unwrap()
            .0,
    )
    .quaternion;
    let bevy_parent_w = read_ang_vel(app.world(), parent_entity);

    // ── Runner path ────────────────────────────────────────────────
    // Inertial-only environment (mu = 0) so the integrator's force
    // evaluation between snapshot and writeback contributes nothing.
    let time = RunnerSimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    let inertial = sim.add_source(
        "InertialAnchor",
        RunnerGravitySourceEntry {
            source: RunnerGravitySource {
                mu: 0.0,
                model: RunnerGravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
    let parent_idx = sim.add_body(RunnerVehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&parent_trans),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(parent_rot))),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass))),
        integrator: RunnerIntegratorType::Rk4,
        gravity_controls: RunnerGravityControls {
            controls: vec![RunnerGravityControl::new_spherical(
                inertial,
                GravityGradient::Skip,
            )],
        },
        ..Default::default()
    });
    let child_idx = sim.add_body(RunnerVehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&child_trans),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(child_rot))),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass))),
        integrator: RunnerIntegratorType::Rk4,
        gravity_controls: RunnerGravityControls {
            controls: vec![RunnerGravityControl::new_spherical(
                inertial,
                GravityGradient::Skip,
            )],
        },
        ..Default::default()
    });
    sim.add_body_to_tree(parent_idx, "Parent");
    sim.add_body_to_tree(child_idx, "Child");
    sim.validate().unwrap();

    sim.attach(child_idx, parent_idx, offset, t_parent_child);
    let runner_parent = sim.body(parent_idx);
    let runner_pos = runner_parent.trans.position.raw_si();
    let runner_vel = runner_parent.trans.velocity.raw_si();
    let runner_rot = runner_parent
        .rot
        .expect("6-DOF runner parent must keep rot");
    let runner_q = runner_rot.q_inertial_body.to_jeod_quat();
    let runner_w = runner_rot.ang_vel_body.raw_si();

    // Bit-identical post-attach state across the two adapters.
    assert_eq!(
        bevy_parent_pos.to_array().map(f64::to_bits),
        runner_pos.to_array().map(f64::to_bits),
        "post-attach parent position differs across Bevy / runner: bevy={bevy_parent_pos:?} runner={runner_pos:?}"
    );
    assert_eq!(
        bevy_parent_vel.to_array().map(f64::to_bits),
        runner_vel.to_array().map(f64::to_bits),
        "post-attach parent velocity differs: bevy={bevy_parent_vel:?} runner={runner_vel:?}"
    );
    assert_eq!(
        [
            bevy_parent_q.scalar().to_bits(),
            bevy_parent_q.vector().x.to_bits(),
            bevy_parent_q.vector().y.to_bits(),
            bevy_parent_q.vector().z.to_bits(),
        ],
        [
            runner_q.scalar().to_bits(),
            runner_q.vector().x.to_bits(),
            runner_q.vector().y.to_bits(),
            runner_q.vector().z.to_bits(),
        ],
        "post-attach parent quaternion differs across Bevy / runner"
    );
    assert_eq!(
        bevy_parent_w.to_array().map(f64::to_bits),
        runner_w.to_array().map(f64::to_bits),
        "post-attach parent ang_vel differs: bevy={bevy_parent_w:?} runner={runner_w:?}"
    );

    // ── Detach: same parity assertion. ─────────────────────────────
    // Bevy detach inserts `DetachedSubtreeStateC` on the child during
    // `staging_system` (capturing the at-detach-instant state) and
    // then `step_detached_system` advances it by `dt` in the same
    // FixedUpdate tick. To recover the captured value, subtract
    // `vel * dt` from the position; with zero ang_vel the attitude
    // and ang_vel are identical to the captured ones.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent {
            child: child_entity,
        });
    step(&mut app, 1, dt);

    let bevy_child_state = app
        .world()
        .get::<DetachedSubtreeStateC>(child_entity)
        .expect("DetachEvent must insert DetachedSubtreeStateC")
        .0;

    sim.detach(child_idx);
    let runner_child = sim.body(child_idx);
    let runner_parent_post = sim.body(parent_idx);

    // Reverse Bevy's one-tick `step_ballistic` advance: the captured
    // state at the detach instant is `pos = bevy.pos − vel·dt` (vel
    // unchanged, attitude unchanged because ang_vel = 0).
    let bevy_pos_si = bevy_child_state.composite_position.raw_si();
    let bevy_vel_si = bevy_child_state.composite_velocity.raw_si();
    let bevy_child_pos_at_detach = bevy_pos_si - bevy_vel_si * dt;
    let bevy_child_vel_at_detach = bevy_vel_si;
    let bevy_child_q_at_detach = bevy_child_state.composite_attitude.to_jeod_quat();
    let bevy_child_w_at_detach = bevy_child_state.composite_ang_vel_body;

    let runner_child_pos = runner_child.trans.position.raw_si();
    let runner_child_vel = runner_child.trans.velocity.raw_si();
    assert_eq!(
        bevy_child_pos_at_detach.to_array().map(f64::to_bits),
        runner_child_pos.to_array().map(f64::to_bits),
        "post-detach child position differs: bevy(captured)={bevy_child_pos_at_detach:?} runner(body.trans)={runner_child_pos:?}"
    );
    assert_eq!(
        bevy_child_vel_at_detach.to_array().map(f64::to_bits),
        runner_child_vel.to_array().map(f64::to_bits),
        "post-detach child velocity differs"
    );
    let runner_child_rot = runner_child.rot.expect("6-DOF child must keep rot");
    let runner_child_q = runner_child_rot.q_inertial_body.to_jeod_quat();
    let runner_child_w = runner_child_rot.ang_vel_body.raw_si();
    assert_eq!(
        [
            bevy_child_q_at_detach.scalar().to_bits(),
            bevy_child_q_at_detach.vector().x.to_bits(),
            bevy_child_q_at_detach.vector().y.to_bits(),
            bevy_child_q_at_detach.vector().z.to_bits(),
        ],
        [
            runner_child_q.scalar().to_bits(),
            runner_child_q.vector().x.to_bits(),
            runner_child_q.vector().y.to_bits(),
            runner_child_q.vector().z.to_bits(),
        ],
        "post-detach child quaternion differs"
    );
    assert_eq!(
        bevy_child_w_at_detach.to_array().map(f64::to_bits),
        runner_child_w.to_array().map(f64::to_bits),
        "post-detach child ang_vel differs"
    );

    // Parent-side post-detach CoM-shift parity. The runner's
    // `Simulation::detach` writes the post-detach parent CoM-shift
    // directly into `body.trans` (the runner's mass tree is the
    // single source of truth). The Bevy adapter writes the same
    // shift into `TranslationalStateC` during `staging_system` —
    // reading `parent_pre_composite_props` from the live
    // `MassTreeR` arena rather than the entity's `MassPropertiesC`
    // (which the ECS-tree fast path in `composite_mass_system`
    // reverts to its `CoreMassPropertiesC` cache when no
    // `MassChildOf` edge is present). With both adapters keying off
    // the same arena composite, the parent-side CoM-shift is
    // bit-identical across runtimes.
    let bevy_parent_pos_post_detach = read_position(app.world(), parent_entity);
    let bevy_parent_vel_post_detach = read_velocity(app.world(), parent_entity);
    let bevy_parent_q_post_detach = astrodyn::typed_bridge::rot_typed_to_raw(
        &app.world()
            .get::<RotationalStateC>(parent_entity)
            .unwrap()
            .0,
    )
    .quaternion;
    let bevy_parent_w_post_detach = read_ang_vel(app.world(), parent_entity);

    let runner_parent_post_pos = runner_parent_post.trans.position.raw_si();
    let runner_parent_post_vel = runner_parent_post.trans.velocity.raw_si();
    assert_eq!(
        bevy_parent_pos_post_detach.to_array().map(f64::to_bits),
        runner_parent_post_pos.to_array().map(f64::to_bits),
        "post-detach parent position differs across Bevy / runner: bevy={bevy_parent_pos_post_detach:?} runner={runner_parent_post_pos:?}"
    );
    assert_eq!(
        bevy_parent_vel_post_detach.to_array().map(f64::to_bits),
        runner_parent_post_vel.to_array().map(f64::to_bits),
        "post-detach parent velocity differs: bevy={bevy_parent_vel_post_detach:?} runner={runner_parent_post_vel:?}"
    );
    let runner_parent_post_rot = runner_parent_post
        .rot
        .expect("6-DOF runner parent must keep rot post-detach");
    let runner_parent_post_q = runner_parent_post_rot.q_inertial_body.to_jeod_quat();
    let runner_parent_post_w = runner_parent_post_rot.ang_vel_body.raw_si();
    assert_eq!(
        [
            bevy_parent_q_post_detach.scalar().to_bits(),
            bevy_parent_q_post_detach.vector().x.to_bits(),
            bevy_parent_q_post_detach.vector().y.to_bits(),
            bevy_parent_q_post_detach.vector().z.to_bits(),
        ],
        [
            runner_parent_post_q.scalar().to_bits(),
            runner_parent_post_q.vector().x.to_bits(),
            runner_parent_post_q.vector().y.to_bits(),
            runner_parent_post_q.vector().z.to_bits(),
        ],
        "post-detach parent quaternion differs"
    );
    assert_eq!(
        bevy_parent_w_post_detach.to_array().map(f64::to_bits),
        runner_parent_post_w.to_array().map(f64::to_bits),
        "post-detach parent ang_vel differs: bevy={bevy_parent_w_post_detach:?} runner={runner_parent_post_w:?}"
    );
}

/// **Cross-integration-frame attach: Bevy / runner parity.**
///
/// Both adapters lift each body to root inertial via its pre-attach
/// `IntegOrigin` before calling `combine_states_at_attach`, then
/// lower the merged composite through the integrated tree root's
/// integ origin for the writeback. With identical inputs the two
/// adapters must produce bit-identical post-attach state on the
/// integrated tree root (the parent), end-to-end across the
/// kernel-shift-site pair. Any drift indicates the lift / lower
/// pair has diverged between the two adapters.
///
/// The scenario uses a non-zero parent integ origin (parent
/// integrates in `Earth.inertial`, child integrates in
/// `Moon.inertial`-style placeholder source at a distinct position)
/// so the lift / lower contributions are non-trivial; the
/// same-integ-frame path collapses them to zero and is already
/// covered by `bevy_parity_attach_detach_momentum_bevy_runner_parity_attach_detach_momentum`.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_runner_parity_cross_integ_frame_attach() {
    use astrodyn::{
        GravityControl as RunnerGravityControl, GravityControls as RunnerGravityControls,
        GravityModel as RunnerGravityModel, GravitySource as RunnerGravitySource,
        GravitySourceEntry as RunnerGravitySourceEntry, IntegratorType as RunnerIntegratorType,
        SimulationTime as RunnerSimulationTime, VehicleConfig as RunnerVehicleConfig,
    };
    use astrodyn_runner::Simulation;

    // Co-mover scenario keeps post-attach ω = 0, so any
    // integrator-tick attitude drift between the adapters cancels
    // out and the parity assertion is on the kernel writeback alone
    // (same scope guard as the same-integ-frame parity test above).
    let parent_mass = MassProperties::with_inertia(
        420.0,
        DMat3::from_diagonal(DVec3::new(150.0, 200.0, 250.0)),
        DVec3::ZERO,
    );
    let child_mass = MassProperties::with_inertia(
        80.0,
        DMat3::from_diagonal(DVec3::new(40.0, 50.0, 60.0)),
        DVec3::ZERO,
    );

    // Two non-central sources at distinct root-inertial positions —
    // the same shape as the runner's
    // `runner_detach_lifts_through_integ_origin` test (Earth at
    // 1.5e11 m off the SSB-rooted frame) generalised so that *both*
    // bodies integrate in non-root frames and the two integ origins
    // differ. Source A holds the parent; source B holds the child.
    let source_a_pos = DVec3::new(1.5e11, 0.0, 0.0);
    let source_b_pos = DVec3::new(1.5e11 + 4.0e8, 0.0, 0.0);

    // Per-frame integ-frame-relative initial states. Co-mover
    // velocity (so the merge is "soft"); offset is non-trivial in
    // the parent's struct frame.
    let common_velocity = DVec3::new(0.0, 7600.0, 0.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: common_velocity,
    };
    let child_trans = TranslationalState {
        // Place the child at root-inertial position (~7e6 m relative
        // to source B) so once both bodies are lifted to root
        // inertial, their absolute separation is consistent with
        // the offset below (`combine_states_at_attach` is offset-
        // agnostic on the kinematic side; the offset only enters
        // the orientation-of-link maths). The relative separation
        // between the two absolute positions is
        //   |(source_a + parent.pos) - (source_b + child.pos)|
        //  = |(1.5e11+7e6) - (1.5e11+4e8+7e6)| = 4e8 m
        // mirroring "parent in low Earth orbit, child near Moon".
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: common_velocity,
    };
    let parent_rot = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    let child_rot = parent_rot;
    let offset = DVec3::new(3.0, 0.0, 0.0);
    let t_parent_child = DMat3::IDENTITY;
    let dt = 1.0;

    // ── Bevy path ──────────────────────────────────────────────────
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(dt));
    app.insert_resource(IntegrationDtR(dt));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let mu = 0.0_f64; // gravity-free environment so the integrator's
                      // force evaluation between snapshot and writeback
                      // contributes nothing.
    let source_a = app
        .world_mut()
        .spawn((
            Name::new("SourceA"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                source_a_pos,
            )),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: source_a_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();
    let source_b = app
        .world_mut()
        .spawn((
            Name::new("SourceB"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                source_b_pos,
            )),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: source_b_pos,
                velocity: DVec3::ZERO,
            }),
        ))
        .id();
    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(parent_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
            IntegSourceC(Some(source_a)),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(child_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
            IntegSourceC(Some(source_b)),
        ))
        .id();
    app.world_mut().run_schedule(Startup);

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(offset),
            t_parent_child: astrodyn::FrameTransform::from_matrix(t_parent_child),
        });
    step(&mut app, 1, dt);

    let bevy_parent_pos = read_position(app.world(), parent_entity);
    let bevy_parent_vel = read_velocity(app.world(), parent_entity);
    let bevy_parent_q = astrodyn::typed_bridge::rot_typed_to_raw(
        &app.world()
            .get::<RotationalStateC>(parent_entity)
            .unwrap()
            .0,
    )
    .quaternion;
    let bevy_parent_w = read_ang_vel(app.world(), parent_entity);

    // ── Runner path ────────────────────────────────────────────────
    // Build an SSB root + two non-central sources at the same
    // inertial positions as the Bevy fixture. Parent integrates in
    // source A, child in source B — the runner's `integ_source` is
    // the equivalent of `IntegSourceC` here.
    let time = RunnerSimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    let _ssb = sim.add_source(
        "SSB",
        RunnerGravitySourceEntry {
            source: RunnerGravitySource {
                mu: 0.0,
                model: RunnerGravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
    let runner_source_a = sim.add_source(
        "SourceA",
        RunnerGravitySourceEntry {
            source: RunnerGravitySource {
                mu: 0.0,
                model: RunnerGravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(source_a_pos),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    let runner_source_b = sim.add_source(
        "SourceB",
        RunnerGravitySourceEntry {
            source: RunnerGravitySource {
                mu: 0.0,
                model: RunnerGravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(source_b_pos),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    let parent_idx = sim.add_body(RunnerVehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&parent_trans),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(parent_rot))),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass))),
        integrator: RunnerIntegratorType::Rk4,
        gravity_controls: RunnerGravityControls {
            controls: vec![RunnerGravityControl::new_spherical(
                runner_source_a,
                GravityGradient::Skip,
            )],
        },
        integ_source: Some(runner_source_a),
        ..Default::default()
    });
    let child_idx = sim.add_body(RunnerVehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&child_trans),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(child_rot))),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass))),
        integrator: RunnerIntegratorType::Rk4,
        gravity_controls: RunnerGravityControls {
            controls: vec![RunnerGravityControl::new_spherical(
                runner_source_b,
                GravityGradient::Skip,
            )],
        },
        integ_source: Some(runner_source_b),
        ..Default::default()
    });
    sim.add_body_to_tree(parent_idx, "Parent");
    sim.add_body_to_tree(child_idx, "Child");
    sim.validate().unwrap();

    sim.attach(child_idx, parent_idx, offset, t_parent_child);
    let runner_parent = sim.body(parent_idx);
    let runner_pos = runner_parent.trans.position.raw_si();
    let runner_vel = runner_parent.trans.velocity.raw_si();
    let runner_rot = runner_parent
        .rot
        .expect("6-DOF runner parent must keep rot");
    let runner_q = runner_rot.q_inertial_body.to_jeod_quat();
    let runner_w = runner_rot.ang_vel_body.raw_si();

    // Bit-identical post-attach state across the two adapters. Any
    // mismatch between the lift / lower shifts in the two adapter
    // paths produces a discrepancy bounded below by the integ-origin
    // separation (`source_b - source_a` ≈ 4e8 m), well beyond f64
    // rounding — so a single bit difference here means the lift /
    // lower pair has diverged, not just numerical noise.
    assert_eq!(
        bevy_parent_pos.to_array().map(f64::to_bits),
        runner_pos.to_array().map(f64::to_bits),
        "post-attach parent position differs across Bevy / runner: \
         bevy={bevy_parent_pos:?} runner={runner_pos:?}",
    );
    assert_eq!(
        bevy_parent_vel.to_array().map(f64::to_bits),
        runner_vel.to_array().map(f64::to_bits),
        "post-attach parent velocity differs across Bevy / runner: \
         bevy={bevy_parent_vel:?} runner={runner_vel:?}",
    );
    assert_eq!(
        [
            bevy_parent_q.scalar().to_bits(),
            bevy_parent_q.vector().x.to_bits(),
            bevy_parent_q.vector().y.to_bits(),
            bevy_parent_q.vector().z.to_bits(),
        ],
        [
            runner_q.scalar().to_bits(),
            runner_q.vector().x.to_bits(),
            runner_q.vector().y.to_bits(),
            runner_q.vector().z.to_bits(),
        ],
        "post-attach parent quaternion differs across Bevy / runner",
    );
    assert_eq!(
        bevy_parent_w.to_array().map(f64::to_bits),
        runner_w.to_array().map(f64::to_bits),
        "post-attach parent ang_vel differs across Bevy / runner: \
         bevy={bevy_parent_w:?} runner={runner_w:?}",
    );

    // Frame-tree side check: the Bevy adapter additionally reparents
    // the child's body-frame entity under the parent's integ-frame
    // entity (mirroring JEOD's `set_integ_frame`). The runner does
    // not have a parallel structural component since `integ_frame_id`
    // is set at body-spawn time and never mutated by `attach` — the
    // runner relies purely on the integ-origin lift, while the Bevy
    // adapter must keep both the lift *and* the frame-tree node in
    // sync. Verify the Bevy reparent landed.
    let source_a_frame = app.world().get::<FrameEntityC>(source_a).unwrap().0;
    let child_frame_entity = app.world().get::<FrameEntityC>(child_entity).unwrap().0;
    assert_eq!(
        app.world()
            .get::<ChildOf>(child_frame_entity)
            .unwrap()
            .parent(),
        source_a_frame,
        "post-attach: Bevy adapter must reparent child body-frame entity under \
         the parent's integ-frame entity (source A's frame entity), matching JEOD's \
         `set_integ_frame` semantics",
    );
}

/// **Root-equivalent topology must NOT panic.**
///
/// `astrodyn_runner` collapses the central body's inertial frame onto the
/// root frame, so a body with `IntegSourceC(Some(earth))` and a body
/// with `IntegSourceC(None)` integrate in identical coordinates. The
/// Bevy adapter splits them topologically — `Earth.inertial` lives one
/// level below the generic root with identity state — but they remain
/// numerically root-equivalent.
///
/// The cross-integ-frame fence in `staging_system` therefore must
/// fold root-equivalent parents back onto root before comparing.
/// Otherwise the canonical Earth-centered-as-central-body setup (the
/// shape `tests/spawn_bevy_integ_source_and_frame_switches.rs` covers)
/// would falsely panic any time a sibling body left its `IntegSourceC`
/// at the implicit-root default.
///
/// This test pins that policy: parent has `IntegSourceC(Some(source))`,
/// child has `IntegSourceC(None)`. Their body-frame entities have
/// distinct `ChildOf` parents (`source.inertial` vs root), but the
/// helper resolves both to root, the fence accepts the attach, and the
/// merge proceeds.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_root_equivalent_parents_succeed() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    // Source at the origin with identity state — the central-body
    // case where `Earth.inertial` is structurally root-equivalent.
    // Stored `TranslationalStateC` and `SourceInertialPositionC` are
    // both zero so `register_source_frames_system` produces a frame
    // entity whose `FrameTransC` / `FrameRotC` / `FrameAngVelC` are
    // all identity — i.e. root-equivalent.
    let mu = 3.986004415e14_f64;
    let source = app
        .world_mut()
        .spawn((
            Name::new("Source"),
            GravitySourceC(GravitySource {
                mu,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    // Parent integrates in the source's inertial frame. After
    // `register_body_frames_system` runs at startup its body-frame
    // entity is `ChildOf(source.inertial)`.
    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
            IntegSourceC(Some(source)),
        ))
        .id();
    // Child integrates root-relative. After registration its body-frame
    // entity is `ChildOf(root)`. The `ChildOf` parents differ, but
    // both are root-equivalent.
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
            IntegSourceC(None),
        ))
        .id();

    app.world_mut().run_schedule(Startup);
    step(&mut app, 1, 1.0);

    // Sanity-check the asymmetric setup: the body-frame entities have
    // different `ChildOf` parents (one under the source, one under
    // root), so a raw equality fence would reject this attach.
    let parent_frame_entity = app
        .world()
        .get::<FrameEntityC>(parent_entity)
        .expect("parent registered FrameEntityC")
        .0;
    let child_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("child registered FrameEntityC")
        .0;
    let parent_parent = app
        .world()
        .get::<ChildOf>(parent_frame_entity)
        .expect("parent body-frame has ChildOf")
        .parent();
    let child_parent = app
        .world()
        .get::<ChildOf>(child_frame_entity)
        .expect("child body-frame has ChildOf")
        .parent();
    assert_ne!(
        parent_parent, child_parent,
        "fixture sanity: parent and child body-frame entities must have \
         different ChildOf parents — that is the whole point of this test"
    );

    // Fire the attach. The fence must fold both parents onto root and
    // proceed.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // Verify the attach actually happened.
    let tree = &app.world().resource::<MassTreeR>().0;
    assert_eq!(
        tree.parent(id_b),
        Some(id_a),
        "post-attach: child's mass-tree parent must be the parent body — \
         the root-equivalent topology must let the attach proceed"
    );
}

/// **Malformed frame node must panic.**
///
/// If a body still carries `FrameEntityC` but its frame entity has
/// lost its `ChildOf` parent, the live integ-frame source of truth is
/// gone. `frame_switch_system` already hard-fails the same invariant
/// at `src/systems.rs:765-781`; the staging fence must match that
/// behavior so the same misconfig is rejected at attach time rather
/// than silently bypassed.
///
/// Per the Fail Loudly rule (CLAUDE.md), the fence panics with a
/// diagnostic naming the missing invariant — `ChildOf` on the
/// body-frame entity — and points to
/// `register_body_frames_system` as the canonical source of that
/// `ChildOf` insertion.
#[test]
#[should_panic(expected = "has no ChildOf parent")]
fn bevy_parity_attach_detach_momentum_bevy_attach_malformed_frame_node_panics() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    // Run Startup so register_body_frames_system inserts the bodies'
    // FrameEntityC + the body-frame entities' ChildOf parents.
    app.world_mut().run_schedule(Startup);
    step(&mut app, 1, 1.0);

    // Sanity-check the registration ran.
    let child_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("child registered FrameEntityC")
        .0;
    assert!(
        app.world().get::<ChildOf>(child_frame_entity).is_some(),
        "fixture sanity: child body-frame entity must initially carry ChildOf"
    );

    // Corrupt the frame tree: remove the child's body-frame entity's
    // `ChildOf` parent while leaving the body's `FrameEntityC` intact.
    // This is the exact malformed-frame-node shape the new fence must
    // panic on.
    app.world_mut()
        .entity_mut(child_frame_entity)
        .remove::<ChildOf>();

    // Fire the attach. The fence must panic before the merge runs.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);
}

/// **Equal-but-illegal parents must panic.**
///
/// A fence that only checks "do both bodies' `ChildOf` parents
/// match?" silently accepts a configuration where both body-frame
/// entities have been reparented under some arbitrary frame entity
/// (e.g. another body's frame entity, or a stray frame entity created
/// by a buggy mission script). `frame_switch_system` already rejects
/// the same misconfig at `src/systems.rs:765-781`: the body's integ
/// frame must be either the root frame entity or a registered gravity
/// source's frame entity.
///
/// This test pins the legality check: spawn parent + child, then
/// reparent both body-frame entities under a third frame entity that
/// is not registered as a gravity source. The fence must panic with
/// the "registered gravity source" diagnostic.
#[test]
#[should_panic(expected = "is neither the root frame entity")]
fn bevy_parity_attach_detach_momentum_bevy_attach_equal_but_illegal_parents_panic() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    app.world_mut().run_schedule(Startup);
    step(&mut app, 1, 1.0);

    let parent_frame_entity = app
        .world()
        .get::<FrameEntityC>(parent_entity)
        .expect("parent registered FrameEntityC")
        .0;
    let child_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("child registered FrameEntityC")
        .0;

    // Spawn a stray frame entity under root — looks like a frame node
    // but is not registered as a gravity source. Then reparent both
    // body-frame entities under it. The `ChildOf` parents now match
    // (so a naive equality check passes) but the parent is illegal:
    // it is not the root frame entity and not a registered source's
    // frame entity.
    //
    // The stray frame's stored state is *non-identity* so it is not
    // root-equivalent — otherwise the helper would fold it back onto
    // root and the legality check would let it through. Distinguishing
    // a registered-source frame from a stray frame is the entire job
    // of the legality check; the test must hit that branch directly.
    let root_e = app.world().resource::<RootFrameEntityR>().0;
    let stray_frame = app
        .world_mut()
        .spawn((
            Name::new("StrayFrame"),
            FrameTransC {
                position: DVec3::new(1.0e8, 0.0, 0.0),
                velocity: DVec3::ZERO,
            },
            FrameRotC::default(),
            FrameAngVelC::default(),
            ChildOf(root_e),
        ))
        .id();
    app.world_mut()
        .entity_mut(parent_frame_entity)
        .insert(ChildOf(stray_frame));
    app.world_mut()
        .entity_mut(child_frame_entity)
        .insert(ChildOf(stray_frame));

    // Fire the attach. The legality check in the fence must reject
    // the equal-but-illegal parents.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);
}

/// **Root-equivalent stray parent must still be rejected.**
///
/// The legality check in the cross-integ-frame fence must run on the
/// *original* `ChildOf` parent of each body's frame entity, not on
/// the root-equivalent fold of that parent. Otherwise a stray frame
/// entity that happens to be a direct child of root with identity
/// state would silently fold to the root frame entity and pass
/// legality — even though `frame_switch_system` would reject the
/// same parent on the next tick because it is not in the registered
/// source-frame set.
///
/// This test pins the soundness gap: spawn an unregistered stray
/// frame entity that *does* satisfy root-equivalence (direct child
/// of root with identity `FrameTransC` / `FrameRotC` / `FrameAngVelC`),
/// reparent both bodies under it, and verify the fence panics with
/// the legality diagnostic before the equality check (which would
/// pass after folding) gets a chance to let the attach through.
#[test]
#[should_panic(expected = "is neither the root frame entity")]
fn bevy_parity_attach_detach_momentum_bevy_attach_root_equivalent_stray_parent_panics() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    app.world_mut().run_schedule(Startup);
    step(&mut app, 1, 1.0);

    let parent_frame_entity = app
        .world()
        .get::<FrameEntityC>(parent_entity)
        .expect("parent registered FrameEntityC")
        .0;
    let child_frame_entity = app
        .world()
        .get::<FrameEntityC>(child_entity)
        .expect("child registered FrameEntityC")
        .0;

    // Spawn a stray frame entity directly under root with identity
    // state — it satisfies the root-equivalent topology rule but is
    // NOT a registered gravity source. A fence that folds before
    // checking legality would see `root_e == root_e` and accept the
    // attach; the corrected ordering rejects on the un-folded parent.
    let root_e = app.world().resource::<RootFrameEntityR>().0;
    let stray_root_equivalent_frame = app
        .world_mut()
        .spawn((
            Name::new("StrayRootEquivalentFrame"),
            FrameTransC::default(),
            FrameRotC::default(),
            FrameAngVelC::default(),
            ChildOf(root_e),
        ))
        .id();
    app.world_mut()
        .entity_mut(parent_frame_entity)
        .insert(ChildOf(stray_root_equivalent_frame));
    app.world_mut()
        .entity_mut(child_frame_entity)
        .insert(ChildOf(stray_root_equivalent_frame));

    // Fire the attach. The legality check must run on the original
    // (un-folded) parent and reject the stray frame even though it
    // would fold to root for the equality comparison.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);
}

/// **Mass-only attach (no `FrameEntityC` on either body) must succeed.**
///
/// `AttachEvent`'s contract requires both entities to carry
/// `MassBodyIdC` — frame-side components are explicitly optional
/// (see `staging_system`'s `bodies` query, which holds `Option<&mut
/// TranslationalStateC>` / `Option<&mut RotationalStateC>`). This
/// matches JEOD's `MassBody`-without-`DynBody` configuration: a
/// passive structural body that lives in the mass tree but has no
/// kinematic state of its own.
///
/// `register_body_frames_system` only inserts `FrameEntityC` for
/// entities filtered by `With<TranslationalStateC>` +
/// `With<DynamicsConfigC>`, so a mass-only body has no
/// `FrameEntityC` and therefore no node in the frame tree. The
/// cross-integ-frame fence has nothing to protect for such a body
/// (no frame-tree state can be corrupted), and its assertions must
/// be skipped — otherwise a legitimate mass-only attach panics
/// where it used to succeed.
///
/// This test pins that contract: spawn parent and child as pure
/// mass-tree nodes (no `DynamicsConfigC`, no `TranslationalStateC`,
/// no `RotationalStateC` — therefore no `FrameEntityC` after
/// registration), fire `AttachEvent`, and verify the fence is
/// bypassed and the mass tree composes successfully.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_mass_only_no_frame_entity_succeeds() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    // Mass-only spawn: only MassBodyIdC + MassPropertiesC. No
    // DynamicsConfigC, no TranslationalStateC, no RotationalStateC.
    // register_body_frames_system will skip these entities (its
    // filter is `With<TranslationalStateC>` + `With<DynamicsConfigC>`)
    // so neither carries `FrameEntityC` after Startup runs.
    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    // Sanity-check the fixture: neither entity carries
    // `FrameEntityC`. If this fails, the registration filter
    // changed and this regression no longer exercises the
    // mass-only carve-out — update the spawn above so neither
    // entity matches `register_body_frames_system`'s filter.
    assert!(
        app.world().get::<FrameEntityC>(parent_entity).is_none(),
        "fixture broken: mass-only parent unexpectedly has FrameEntityC"
    );
    assert!(
        app.world().get::<FrameEntityC>(child_entity).is_none(),
        "fixture broken: mass-only child unexpectedly has FrameEntityC"
    );

    // Fire the attach event. The fence's mass-only carve-out must
    // skip the legality / equality assertions when either body
    // lacks `FrameEntityC`. The mass-tree composite recompute
    // still runs; without the carve-out the previous code panicked
    // here on `body_frames.get(body)` returning Err.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);

    // Composite mass on the parent must reflect both bodies
    // post-attach. Reading parent's MassPropertiesC after a step
    // returns the composite (parent + child) per the mass-tree's
    // post-order recompute.
    let composite_mass = read_mass(app.world(), parent_entity);
    let expected = parent_mass.mass + child_mass.mass;
    assert!(
        (composite_mass - expected).abs() < 1e-12,
        "mass-only attach: parent composite mass {composite_mass} != \
         expected {expected} — the mass-tree composite recompute did \
         not run, indicating the attach was rejected by the fence \
         despite the mass-only carve-out."
    );
}

/// **Half-broken frame tree (`FrameEntityC` present but `ChildOf`
/// missing) must still panic.**
///
/// The mass-only carve-out relaxes the fence only for entities with
/// no `FrameEntityC`. An entity that *does* carry `FrameEntityC`
/// has a node in the frame tree, and that node is required to be
/// parented under its integration-frame entity (root or a
/// registered source). If the `ChildOf` is missing, the frame tree
/// itself is corrupt and the attach cannot be safely processed —
/// the fence must surface this per the Fail Loudly rule rather
/// than silently bypassing.
///
/// This test pins the boundary: spawn a normal body (with
/// `FrameEntityC` after registration), then strip the `ChildOf`
/// off its body-frame entity, and verify the attach panics with
/// the "no ChildOf parent" diagnostic.
#[test]
#[should_panic(expected = "has no ChildOf parent")]
fn bevy_parity_attach_detach_momentum_bevy_attach_frame_entity_without_child_of_panics() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    // Strip ChildOf from the parent's body-frame entity. The fence
    // must now panic on the missing parent rather than silently
    // bypass — `FrameEntityC` is still there, so the mass-only
    // carve-out doesn't apply.
    let parent_frame_entity = app
        .world()
        .get::<FrameEntityC>(parent_entity)
        .expect("parent registered FrameEntityC")
        .0;
    app.world_mut()
        .entity_mut(parent_frame_entity)
        .remove::<ChildOf>();

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);
}

/// **Registration race: dynamic body without `FrameEntityC` must
/// panic.**
///
/// The mass-only carve-out in `staging_system`'s cross-integ-frame
/// fence skips the legality / equality assertions when an attach
/// participant has no `FrameEntityC`. That carve-out is intentionally
/// narrow: it covers the legitimate `MassBody`-without-`DynBody`
/// configuration (entity carries `MassBodyIdC` + `MassPropertiesC` but
/// is missing at least one of the eligibility components for
/// `register_body_frames_system` — `DynamicsConfigC` and
/// `TranslationalStateC`). For such a body, registration will *never*
/// insert `FrameEntityC` and the entity has nothing in the frame tree
/// to corrupt.
///
/// The opposite case — an entity carrying *both* eligibility
/// components but lacking `FrameEntityC` — is a registration race,
/// not a mass-only configuration. `register_body_frames_system` runs
/// before `AstrodynSet::EphemerisUpdate`; `staging_system` runs later
/// in the same `FixedUpdate` (after `Environment`, before
/// `Interaction`). A body spawned mid-tick after the registration
/// pass already ran will not yet carry `FrameEntityC` even though its
/// component set qualifies for one. Treating that as carve-out would
/// silently corrupt the frame tree on the next register pass; per
/// Fail Loudly the fence must surface the misconfiguration.
///
/// This test pins the boundary: spawn a fully-eligible dynamic body
/// (`DynamicsConfigC` + `TranslationalStateC` + `RotationalStateC` +
/// `MassPropertiesC` + `MassBodyIdC`), strip the `FrameEntityC` that
/// `register_body_frames_system` inserts at Startup, fire
/// `AttachEvent`, and verify the fence panics with the
/// registration-race diagnostic instead of silently bypassing.
#[test]
#[should_panic(expected = "registration race")]
fn bevy_parity_attach_detach_momentum_bevy_attach_dynamic_body_with_no_frame_entity_panics() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    // Sanity-check the fixture: after Startup, register_body_frames_system
    // has registered the child. Strip its `FrameEntityC` mid-tick and
    // run staging directly. We bypass `FixedUpdate` because that
    // schedule re-runs `register_body_frames_system` first, which
    // would re-insert `FrameEntityC` and mask the race we're trying
    // to pin.
    assert!(
        app.world().get::<FrameEntityC>(child_entity).is_some(),
        "fixture broken: dynamic child unexpectedly has no FrameEntityC \
         after Startup; the registration filter likely changed"
    );
    app.world_mut()
        .entity_mut(child_entity)
        .remove::<FrameEntityC>();

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    // Invoke `staging_system` directly so the registration-race
    // condition (eligibility components present, FrameEntityC
    // absent) is observed by the fence — running `FixedUpdate`
    // would re-register the child first.
    app.world_mut()
        .run_system_cached(astrodyn_bevy::staging_system::<astrodyn::Earth>)
        .expect("run staging_system");
}

/// **Dynamic child attached to mass-only parent must panic.**
///
/// Asymmetric carve-out boundary: a mass-only child attached to a
/// dynamic parent matches JEOD's `add_mass_body` path — the dynamic
/// parent carries the composite state and the mass-tree composite
/// recompute folds the mass-only child's mass into the parent's
/// composite. That direction is allowed.
///
/// The reverse — a *dynamic* child attached to a *mass-only* parent
/// — is rejected. JEOD's `dyn_body_attach.cc::attach_validate_parent`
/// rejects this with "Dynamic attachments can only be made to valid
/// DynBodies"; in our pipeline the combine-back-write only writes
/// the merged composite into the parent's `TranslationalStateC` /
/// `RotationalStateC`, which a mass-only parent does not carry. With
/// no place to receive the merged state, allowing the attach
/// silently drops the result. Per Fail Loudly the fence must surface
/// this.
///
/// This test pins that boundary: spawn a mass-only parent (no
/// `DynamicsConfigC`, no `TranslationalStateC`) and a dynamic child
/// (full eligibility), fire `AttachEvent`, and verify the fence
/// panics with the dynamic-child-on-mass-only-parent diagnostic.
#[test]
#[should_panic(expected = "Dynamic attachments can only be made to valid DynBodies")]
fn bevy_parity_attach_detach_momentum_bevy_attach_dynamic_child_on_mass_only_parent_panics() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    // Mass-only parent: no DynamicsConfigC / TranslationalStateC /
    // RotationalStateC, so register_body_frames_system will skip it
    // and it never acquires FrameEntityC. JEOD calls this a
    // MassBody-without-DynBody configuration.
    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    // Dynamic child: full eligibility set, so register_body_frames_system
    // inserts FrameEntityC at Startup.
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    assert!(
        app.world().get::<FrameEntityC>(parent_entity).is_none(),
        "fixture broken: mass-only parent unexpectedly has FrameEntityC"
    );
    assert!(
        app.world().get::<FrameEntityC>(child_entity).is_some(),
        "fixture broken: dynamic child has no FrameEntityC after Startup"
    );

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step(&mut app, 1, 1.0);
}

/// **State-completeness fail-loud (FrameEntityC present, state
/// component(s) stripped).**
///
/// `register_body_frames_system` is one-time per body — it inserts
/// `FrameEntityC` once and never cleans it up if the eligibility
/// components are removed afterward. A body that ends up with
/// `FrameEntityC` but no `TranslationalStateC` (or `RotationalStateC`)
/// reaches `staging_system` in a miscomputing-attach state: the
/// kernel reads `position` / `velocity` / `quaternion` / `ang_vel`
/// from the absent components and silently substitutes zero / identity,
/// then writes the merged composite back conditionally on the same
/// components — so the merged result is silently dropped.
///
/// Step 1.5 of the cross-integ-frame fence catches this and panics
/// with the state-completeness diagnostic, naming the missing
/// component(s). JEOD's `dyn_body_attach.cc::attach_validate_child`
/// (lines 121-180) rejects the analog with "Child body has an
/// incomplete state" / "Root body has an incomplete state".
///
/// This test pins the boundary: spawn a fully-eligible dynamic body
/// (so registration inserts `FrameEntityC` at Startup), strip its
/// `TranslationalStateC` mid-tick, then fire `AttachEvent` and verify
/// the fence panics with the new state-completeness diagnostic
/// instead of silently dropping the merge.
#[test]
#[should_panic(expected = "missing required state component")]
fn bevy_parity_attach_detach_momentum_bevy_attach_frame_entity_without_translational_state_panics()
{
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let parent_trans = TranslationalState {
        position: DVec3::new(7e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    app.add_plugins(AstrodynPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(parent_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    // Sanity-check the fixture: after Startup, the dynamic child has
    // FrameEntityC. Strip its TranslationalStateC mid-tick to drive
    // the partially-stripped state the fence now rejects. Bypass
    // FixedUpdate's register pass by invoking staging_system directly
    // (the same pattern as the registration-race test above) — that
    // pass would not re-insert TranslationalStateC because
    // register_body_frames_system reads it but never writes it.
    assert!(
        app.world().get::<FrameEntityC>(child_entity).is_some(),
        "fixture broken: dynamic child unexpectedly has no FrameEntityC \
         after Startup; the registration filter likely changed"
    );
    app.world_mut()
        .entity_mut(child_entity)
        .remove::<TranslationalStateC<astrodyn::Earth>>();

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    app.world_mut()
        .run_system_cached(astrodyn_bevy::staging_system::<astrodyn::Earth>)
        .expect("run staging_system");
}

/// **Fence runs without `AstrodynPlugin`: dynamic-child-on-mass-only-parent
/// still panics.**
///
/// `RootFrameEntityR` is inserted by `AstrodynPlugin::build` only — a
/// low-level test (or a partial app) that runs `staging_system`
/// directly without `AstrodynPlugin` does not have the resource. The
/// fence's *root-equivalence equality fold* needs the resource (it
/// folds `Earth.inertial`-style direct-child-of-root frames onto
/// root before comparing parent/child integ frames), but the
/// *structural* fail-loud checks — mass-only carve-out, registration
/// race detection, dynamic-child-on-mass-only-parent rejection,
/// state-completeness, legality against `known_source_frames` — must
/// run regardless: they protect invariants that hold without any
/// reference to the root entity.
///
/// This test pins that contract: stand up a mass-tree world *without*
/// `AstrodynPlugin` (so `RootFrameEntityR` is absent), forge the
/// FrameEntityC presence pattern of "dynamic child on mass-only
/// parent" (which the fence rejects with JEOD's `attach_validate_parent`
/// diagnostic), invoke `staging_system` directly, and verify the same
/// panic that fires with `AstrodynPlugin` still fires here.
#[test]
#[should_panic(expected = "Dynamic attachments can only be made to valid DynBodies")]
fn bevy_parity_attach_detach_momentum_bevy_attach_dynamic_child_on_mass_only_parent_panics_without_jeod_plugin(
) {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);
    let child_trans = TranslationalState {
        position: DVec3::new(7e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7600.0, 0.0),
    };
    let initial_rot = RotationalState::default();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    // Crucially: NO `app.add_plugins(AstrodynPlugin);` — this regression
    // pins the fence's behaviour when `RootFrameEntityR` is absent.
    // Register the AttachEvent / DetachEvent message resources by
    // hand so `staging_system` can read its event reader without
    // panicking on "Requested resource does not exist". `AstrodynPlugin`
    // does this in `build`; this regression deliberately bypasses it.
    app.add_message::<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>();
    app.add_message::<DetachEvent>();

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    // Mass-only parent: no DynamicsConfigC / TranslationalStateC /
    // RotationalStateC. Without `register_body_frames_system` running
    // (no plugin) it could never carry FrameEntityC anyway — the
    // structural shape we're pinning is the same.
    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    // Dynamic child carrying full eligibility components AND a
    // pre-built body-frame entity that the fence will resolve to
    // through `body_frames` — without `register_body_frames_system`
    // available we set up the `FrameEntityC` link by hand to mimic
    // the post-registration shape the fence sees in production. The
    // body-frame entity needs a `ChildOf` parent or step 1 panics
    // first with "has no ChildOf parent"; we create a stand-in
    // fake-root entity so the resolver succeeds and the fence
    // reaches the dynamic-child-on-mass-only-parent check (the
    // mismatch we're actually pinning here).
    let fake_root_entity = app
        .world_mut()
        .spawn((
            FrameTransC::default(),
            FrameRotC::default(),
            FrameAngVelC::default(),
        ))
        .id();
    let child_frame_entity = app
        .world_mut()
        .spawn((
            FrameTransC::default(),
            FrameRotC::default(),
            FrameAngVelC::default(),
            ChildOf(fake_root_entity),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(child_trans),
            RotationalStateC::from(
                astrodyn::typed_bridge::rot_raw_to_typed::<astrodyn::SelfRef>(&(initial_rot)),
            ),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
            FrameEntityC(child_frame_entity),
        ))
        .id();

    // Verify our hand-built fixture matches the rejection shape: the
    // mass-only parent has no FrameEntityC; the dynamic child does.
    assert!(
        app.world().get::<FrameEntityC>(parent_entity).is_none(),
        "fixture broken: mass-only parent unexpectedly has FrameEntityC"
    );
    assert!(
        app.world().get::<FrameEntityC>(child_entity).is_some(),
        "fixture broken: dynamic child should carry FrameEntityC"
    );
    assert!(
        !app.world().contains_resource::<RootFrameEntityR>(),
        "fixture broken: RootFrameEntityR is unexpectedly present — \
         this regression pins fence behaviour without AstrodynPlugin"
    );

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    app.world_mut()
        .run_system_cached(astrodyn_bevy::staging_system::<astrodyn::Earth>)
        .expect("run staging_system");
}

/// **Fence runs without `AstrodynPlugin`: legitimate mass-only attach
/// still succeeds.**
///
/// Companion to the negative regression above: the structural
/// fail-loud checks must reject misconfigurations regardless of
/// `RootFrameEntityR`'s presence, but they must *not* reject
/// legitimate mass-only attaches in the same low-level setup.
/// `AstrodynPlugin`-less callers running pure mass-tree composition
/// (no frame tree) must still see the mass-tree composite recompute
/// and integrator reset run as expected.
///
/// This test stands up two pure mass-tree nodes (no `FrameEntityC`,
/// no eligibility components, no `RootFrameEntityR`) and verifies
/// `AttachEvent` composes their masses without panicking.
#[test]
fn bevy_parity_attach_detach_momentum_bevy_attach_mass_only_succeeds_without_jeod_plugin() {
    let parent_mass = MassProperties::new(1000.0);
    let child_mass = MassProperties::new(500.0);

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(1.0));
    app.insert_resource(IntegrationDtR(1.0));
    // No `add_plugins(AstrodynPlugin)` — the fence must not depend on
    // `RootFrameEntityR` for the mass-only carve-out path. Register
    // the AttachEvent / DetachEvent message resources by hand
    // (normally done by `AstrodynPlugin::build`) so `staging_system`'s
    // event reader can run.
    app.add_message::<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>();
    app.add_message::<DetachEvent>();

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(parent_mass)),
            ),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            MassPropertiesC::from(
                astrodyn::typed_bridge::mass_raw_to_typed::<astrodyn::SelfRef>(&(child_mass)),
            ),
            MassBodyIdC(id_b),
        ))
        .id();

    assert!(
        !app.world().contains_resource::<RootFrameEntityR>(),
        "fixture broken: RootFrameEntityR is unexpectedly present — \
         this regression pins fence behaviour without AstrodynPlugin"
    );

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    app.world_mut()
        .run_system_cached(astrodyn_bevy::staging_system::<astrodyn::Earth>)
        .expect("run staging_system");

    let composite_mass = read_mass(app.world(), parent_entity);
    let expected = parent_mass.mass + child_mass.mass;
    assert!(
        (composite_mass - expected).abs() < 1e-12,
        "mass-only attach without AstrodynPlugin: parent composite mass \
         {composite_mass} != expected {expected} — the mass-tree \
         composite recompute did not run, indicating the attach was \
         rejected by the fence despite the mass-only carve-out."
    );
}
