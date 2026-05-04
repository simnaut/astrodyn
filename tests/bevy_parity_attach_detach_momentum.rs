//! Bevy adapter parity for the JEOD attach/detach
//! momentum-conservation port + detached-subtree ballistic tracking.
//! Mirrors `models/dynamics/dyn_body/src/dyn_body_attach.cc` and
//! `dyn_body_detach.cc` in JEOD.
//!
//! Three layers of coverage:
//!
//! 1. **Attach momentum conservation.** Spawn two free-flying bodies
//!    with non-trivial relative state, fire `AttachEvent`, verify the
//!    parent's `TranslationalStateC` / `RotationalStateC` post-event
//!    matches `jeod_sim::stage_attach_combine` byte-for-byte. Linear
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

use bevy::prelude::*;
use bevy_jeod::{
    AttachEvent, DetachEvent, DetachedSubtreeStateC, DynamicsConfigC, FrameDerivativesC,
    FrameEntityC, FrameTransC, GravityAccelerationC, GravityControlsC, GravitySourceC, JeodPlugin,
    MassBodyIdC, MassPropertiesC, MassTreeR, RotationalStateC, SourceInertialPositionC,
    TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, JeodQuat, MassProperties,
    MassTree, RotationalState, StageAttachInputs, TranslationalState,
};
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
    jeod_sim::MassBodyId,
    jeod_sim::MassBodyId,
) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.add_plugins(JeodPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("Parent".into(), parent_mass);
    let id_b = tree.add_body("Child".into(), child_mass);
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC::default(),
            TranslationalStateC::from(parent_trans),
            RotationalStateC::from(parent_rot),
            MassPropertiesC::from(parent_mass),
            MassBodyIdC(id_a),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC::default(),
            TranslationalStateC::from(child_trans),
            RotationalStateC::from(child_rot),
            MassPropertiesC::from(child_mass),
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
        .get::<TranslationalStateC>(entity)
        .expect("entity has TranslationalStateC")
        .0
        .position
        .raw_si()
}

fn read_velocity(world: &World, entity: Entity) -> DVec3 {
    world
        .get::<TranslationalStateC>(entity)
        .expect("entity has TranslationalStateC")
        .0
        .velocity
        .raw_si()
}

fn read_ang_vel(world: &World, entity: Entity) -> DVec3 {
    world
        .get::<RotationalStateC>(entity)
        .expect("entity has RotationalStateC")
        .0
        .to_untyped()
        .ang_vel_body
}

fn read_mass(world: &World, entity: Entity) -> f64 {
    world
        .get::<MassPropertiesC>(entity)
        .expect("entity has MassPropertiesC")
        .0
        .to_untyped()
        .mass
}

/// Attach with relative translational velocity at non-zero offset
/// induces angular momentum (the JEOD "magical merge" — see
/// `dyn_body_attach.cc` / `combine_states_at_attach`). Verify the
/// parent's post-attach `TranslationalStateC` / `RotationalStateC`
/// match the kernel's output byte-for-byte.
#[test]
fn bevy_attach_conserves_linear_and_angular_momentum() {
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
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: DVec3::new(2.0, 0.0, 0.0),
            t_parent_child: DMat3::IDENTITY,
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
    let expected = jeod_sim::stage_attach_combine(StageAttachInputs {
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
fn bevy_attach_no_relative_motion_preserves_parent_state() {
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
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
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
fn bevy_detach_captures_subtree_state() {
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
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
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
    assert!(
        (detached.0.composite_position - expected_pos).length() < 1e-9,
        "detached pos: {:?} expected {:?} (= parent_composite_at_detach + vel·dt)",
        detached.0.composite_position,
        expected_pos
    );
    assert!(
        (detached.0.composite_velocity - parent_vel_at_detach).length() < 1e-12,
        "detached velocity should match parent composite: {:?} vs {:?}",
        detached.0.composite_velocity,
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
fn bevy_detach_derives_child_state_via_rigid_body_composition() {
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
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: attach_offset,
            t_parent_child: DMat3::IDENTITY,
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

    assert!(
        (detached.0.composite_position - expected_pos_after_step).length() < 1e-6,
        "detached pos via rigid-body composition: got {:?}, expected {:?}",
        detached.0.composite_position,
        expected_pos_after_step
    );
    assert!(
        (detached.0.composite_velocity - expected_child_vel).length() < 1e-6,
        "detached velocity via rigid-body composition: got {:?}, expected {:?}",
        detached.0.composite_velocity,
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
fn bevy_detached_subtree_propagates_ballistically() {
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
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
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
        (detached.0.composite_position - post_pos).length() < 1e-12,
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
fn bevy_re_attach_consumes_detached_state() {
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
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
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
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
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
fn stage_attach_combine_parity_smoke() {
    use jeod_sim::stage_attach_combine;
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
fn bevy_step_detached_runs_before_frame_tree_sync() {
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
    // instant — same as `bevy_detached_subtree_propagates_ballistically`).
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
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
fn bevy_detached_body_skips_force_pipeline() {
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
    app.add_plugins(JeodPlugin);

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
            TranslationalStateC::default(),
        ))
        .id();

    let body_entity = app
        .world_mut()
        .spawn((
            Name::new("Body"),
            DynamicsConfigC::default(),
            TranslationalStateC::from(initial_trans),
            RotationalStateC::default(),
            MassPropertiesC::from(body_mass),
            MassBodyIdC(id_body),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
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
    use jeod_sim::{BodyAttitude, DetachedSubtreeState, SelfRef};
    let detached_state = DetachedSubtreeState {
        composite_position: initial_trans.position,
        composite_velocity: initial_trans.velocity,
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
