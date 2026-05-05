//! Bevy ECS attach parity for bodies whose `IntegSourceC` is a
//! non-root planet (lunar-orbit integ frame). Pins the
//! integ→root-inertial lift the staging system applies before feeding
//! `stage_attach_combine` and the matching root→integ-frame lower
//! applied at the writeback into the parent's `TranslationalStateC`.
//!
//! `stage_attach_combine` does cross-body composition (mass-weighted
//! velocity, inertial-frame CoM shift, ω×r over offsets) which is
//! only arithmetic-valid when both sides live in the same inertial
//! frame. `TranslationalStateC` is `IntegrationFrame` storage —
//! planet-relative for a non-root-integrated body — so feeding the
//! kernel raw integ-frame coords for a parent (and child) integrating
//! in `PlanetInertial<Moon>` would silently mix coordinates across
//! distinct origins. The fix lifts each body's snapshot through its
//! own `IntegOrigin` at the read site and lowers the merged result
//! through the parent's `IntegOrigin` at the writeback site. For
//! root-integrated bodies the origin is identically zero so the
//! lift/lower pair is a numerical no-op; for non-root integ_source
//! the lift is load-bearing.
//!
//! What this test pins:
//!
//! 1. **Attach instant**: the merged composite written into the
//!    parent's `TranslationalStateC` matches `stage_attach_combine`'s
//!    output computed with **lifted** root-inertial inputs, then
//!    **lowered** through the parent's `IntegOrigin`. Without the
//!    lift the kernel would be fed integ-frame coords; without the
//!    lower the writeback would land root-inertial coords in
//!    integ-frame storage. Either bug would corrupt the merged
//!    velocity by `MOON_VELOCITY` (~1 km/s) — orders of magnitude
//!    above the 1e-9 tolerance.
//!
//! 2. **`parent_was_detached` writeback**: when the parent itself was
//!    a free-flying detached subtree before the attach, the merged
//!    `DetachedSubtreeStateC` re-stamped onto it must carry
//!    root-inertial coordinates (the typed phantom of
//!    `DetachedSubtreeState.composite_*` is `RootInertial` by
//!    witness). With the upstream lift the kernel produces the
//!    merged composite in root-inertial directly, so the re-stamp is
//!    a direct relabel — no per-branch fixup needed. This test pins
//!    that contract end-to-end.
//!
//! Mirrors the runner's lift/lower pattern in
//! `crates/jeod_runner/src/simulation/mass_tree.rs` (`attach`'s
//! seed-time lift at lines 282-307 and writeback lower at 399-407).

use bevy::prelude::*;
use bevy_jeod::{
    AttachEvent, DetachedSubtreeStateC, DynamicsConfigC, FrameDerivativesC, GravityControlsC,
    IntegSourceC, JeodPlugin, MassBodyIdC, MassPropertiesC, MassTreeR, PlanetBundle,
    RotationalStateC, SourceInertialVelocityC, SourceMutator, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_sim::{
    DynamicsConfig, GravityControls, JeodQuat, MassProperties, MassTree, RotationalState,
    StageAttachInputs, TranslationalState, EARTH, MOON,
};

const DT: f64 = 60.0;
const MOON_OFFSET: DVec3 = DVec3::new(3.844e8, 0.0, 0.0);
/// Non-zero Moon inertial velocity. Picked along y so it doesn't
/// align with `MOON_OFFSET` (along x) — a bug that drops the
/// velocity term of the lift would carry through as a ~1 km/s
/// offset in the merged velocity, while a bug that drops the
/// position term would carry as a ~3.8e8 m offset in the merged
/// position. Both are orders of magnitude above tolerances.
const MOON_VELOCITY: DVec3 = DVec3::new(0.0, 1_000.0, 0.0);

fn parent_mass() -> MassProperties {
    MassProperties::with_inertia(
        1_000.0,
        DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0)),
        DVec3::ZERO,
    )
}

fn child_mass() -> MassProperties {
    MassProperties::with_inertia(
        500.0,
        DMat3::from_diagonal(DVec3::new(50.0, 50.0, 50.0)),
        DVec3::ZERO,
    )
}

/// 100 km lunar circular orbit, in Moon-centered (integ-frame) coords.
fn parent_initial_trans() -> TranslationalState {
    let r = 1_837_400.0;
    let v = (MOON.shape.mu / r).sqrt();
    TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
    }
}

/// Child sits at the same planet-relative position, but with a small
/// translational delta-v so the merge isn't a degenerate soft merge —
/// the kernel must compute a non-trivial mass-weighted velocity. The
/// delta is in the same integ frame as the parent, so feeding raw
/// integ-frame coords to the kernel would still produce a "valid"
/// arithmetic result internally; the bug surface is *what frame the
/// merged output is in* + the writeback to integ-frame storage.
fn child_initial_trans() -> TranslationalState {
    let parent = parent_initial_trans();
    TranslationalState {
        position: parent.position,
        velocity: parent.velocity + DVec3::new(0.0, 0.0, 1.5),
    }
}

fn initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    }
}

fn six_dof_config() -> DynamicsConfig {
    DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    }
}

/// Build a Bevy world with two bodies that integrate in Moon's
/// inertial frame, plus a Moon parked at `MOON_OFFSET` with
/// `MOON_VELOCITY`. Returns the app + parent/child entities + Moon
/// entity + parent's `MassBodyId` (caller can fire attach/detach
/// events afterward and look up the merged composite mass).
fn build_lunar_app() -> (App, Entity, Entity, Entity, jeod_sim::MassBodyId) {
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
        .insert(SourceInertialVelocityC::default())
        .id();

    let mut tree = MassTree::new();
    let id_parent = tree.add_body("Parent".into(), parent_mass());
    let id_child = tree.add_body("Child".into(), child_mass());
    app.insert_resource(MassTreeR(tree));

    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC(six_dof_config()),
            MassPropertiesC::from(parent_mass()),
            MassBodyIdC(id_parent),
            TranslationalStateC::from(parent_initial_trans()),
            RotationalStateC::from(initial_rot()),
            FrameDerivativesC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
            IntegSourceC(Some(moon)),
        ))
        .id();
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC(six_dof_config()),
            MassPropertiesC::from(child_mass()),
            MassBodyIdC(id_child),
            TranslationalStateC::from(child_initial_trans()),
            RotationalStateC::from(initial_rot()),
            FrameDerivativesC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
            IntegSourceC(Some(moon)),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator| {
            m.set_source_state(moon, MOON_OFFSET, MOON_VELOCITY);
        });
    app.world_mut().run_system(sys).unwrap();

    (app, parent_entity, child_entity, moon, id_parent)
}

/// Fresh attach (neither side previously detached): both parent and
/// child integrate in `PlanetInertial<Moon>`. The merged composite
/// written into the parent's `TranslationalStateC` must equal the
/// kernel's output computed with **lifted** root-inertial inputs,
/// then lowered back through the parent's `IntegOrigin`.
///
/// A regression that drops the read-side lift would feed integ-frame
/// coords to a kernel that composes with `parent_t_inertial_struct`
/// — silently producing a merged composite in some hybrid frame
/// that's neither integ nor root-inertial. A regression that drops
/// the writeback lower would land a root-inertial value in
/// integ-frame storage, off by `MOON_OFFSET` in position and
/// `MOON_VELOCITY` in velocity.
#[test]
fn bevy_parity_attach_non_root_integ_source_lift_and_lower() {
    let (mut app, parent_entity, child_entity, _moon, id_parent) = build_lunar_app();

    // Snapshot pre-attach integ-frame state for the kernel comparison.
    let parent_pre_pos_integ = parent_initial_trans().position;
    let parent_pre_vel_integ = parent_initial_trans().velocity;
    let child_pre_pos_integ = child_initial_trans().position;
    let child_pre_vel_integ = child_initial_trans().velocity;

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: jeod_sim::Vec3Ext::m_at::<jeod_sim::StructuralFrame<jeod_sim::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: DMat3::IDENTITY,
        });

    // Run the staging system once via FixedUpdate.
    app.world_mut().run_schedule(FixedUpdate);

    // Independent kernel run: feed the kernel root-inertial coords
    // (= integ-frame + IntegOrigin), then lower the result through
    // the parent's IntegOrigin to compare against the integ-frame
    // storage on the parent entity.
    let combined_mass = app
        .world()
        .resource::<MassTreeR>()
        .0
        .get(id_parent)
        .composite_properties;

    let q = JeodQuat::identity();
    let parent_mass_props = parent_mass();
    let expected = jeod_sim::stage_attach_combine(StageAttachInputs {
        parent_position: parent_pre_pos_integ + MOON_OFFSET,
        parent_velocity: parent_pre_vel_integ + MOON_VELOCITY,
        parent_quaternion: q,
        parent_ang_vel_body: DVec3::ZERO,
        parent_mass: parent_mass_props,
        orig_parent_cm_struct: parent_mass_props.position,
        parent_t_inertial_struct: DMat3::IDENTITY,
        child_position: child_pre_pos_integ + MOON_OFFSET,
        child_velocity: child_pre_vel_integ + MOON_VELOCITY,
        child_quaternion: q,
        child_ang_vel_body: DVec3::ZERO,
        child_mass: child_mass(),
        combined_mass,
    });

    // Lower the kernel's root-inertial output back through the
    // parent's IntegOrigin to produce the expected integ-frame
    // storage value.
    let expected_pos_integ = expected.position - MOON_OFFSET;
    let expected_vel_integ = expected.velocity - MOON_VELOCITY;

    let post_pos_integ = app
        .world()
        .get::<TranslationalStateC>(parent_entity)
        .unwrap()
        .0
        .position
        .raw_si();
    let post_vel_integ = app
        .world()
        .get::<TranslationalStateC>(parent_entity)
        .unwrap()
        .0
        .velocity
        .raw_si();

    assert!(
        (post_pos_integ - expected_pos_integ).length() < 1e-6,
        "post-attach TranslationalStateC.position mismatch (lift/lower):\n  \
         got {:?}\n  expected {:?}\n  delta {:?}",
        post_pos_integ,
        expected_pos_integ,
        post_pos_integ - expected_pos_integ,
    );
    assert!(
        (post_vel_integ - expected_vel_integ).length() < 1e-9,
        "post-attach TranslationalStateC.velocity mismatch (lift/lower):\n  \
         got {:?}\n  expected {:?}\n  delta {:?}",
        post_vel_integ,
        expected_vel_integ,
        post_vel_integ - expected_vel_integ,
    );

    // A regression where the read-side lift is dropped while the
    // writeback lower is kept would produce a stored velocity off
    // by exactly `-MOON_VELOCITY` from the correct expected value:
    // the kernel's mass-weighted average would carry the
    // integ-frame velocity directly, and the lower would then
    // subtract MOON_VELOCITY one more time. The position-branch
    // analogue is off by `-MOON_OFFSET`. Pin both to ensure the
    // assertions above aren't satisfied by a coincidental symmetry.
    let lift_dropped_alias_vel = expected_vel_integ - MOON_VELOCITY;
    let lift_dropped_alias_pos = expected_pos_integ - MOON_OFFSET;
    assert!(
        (post_vel_integ - lift_dropped_alias_vel).length() > 1.0,
        "post-attach velocity matches the 'lift dropped' bug-mode alias \
         (off by -MOON_VELOCITY ≈ 1 km/s) — the assertion above passed \
         only because the soft-merge happened to be symmetric. \
         got {post_vel_integ:?}, alias {lift_dropped_alias_vel:?}",
    );
    assert!(
        (post_pos_integ - lift_dropped_alias_pos).length() > 1.0,
        "post-attach position matches the 'lift dropped' bug-mode alias \
         (off by -MOON_OFFSET ≈ 3.8e8 m). \
         got {post_pos_integ:?}, alias {lift_dropped_alias_pos:?}",
    );
}

/// Re-attach: parent was previously detached (carries
/// `DetachedSubtreeStateC`), then a new child attaches onto it.
/// Pins the `parent_was_detached` writeback re-stamp on
/// `DetachedSubtreeStateC` (typed `RootInertial`).
///
/// With the upstream lift, `merged` is already in root-inertial, so
/// the re-stamp is a direct relabel. A regression that re-introduces
/// a redundant `+IntegOrigin` would double-shift the merged value
/// and land `composite_position` off by `+MOON_OFFSET` from the true
/// root-inertial value (≈3.8e8 m, far above tolerance).
#[test]
fn bevy_parity_attach_non_root_integ_source_parent_was_detached() {
    let (mut app, parent_entity, child_entity, _moon, id_parent) = build_lunar_app();

    // Land the parent in the `parent_was_detached == true` branch by
    // installing a `DetachedSubtreeStateC` directly. The branch only
    // checks the component's presence on the parent at attach time,
    // so this is the minimal in-test reproduction of "parent is a
    // free-flying detached-subtree root" (otherwise reachable by
    // attach-then-detach against a transient grandparent). The
    // captured state mirrors what `stage_detach_capture` would have
    // produced from the parent's pre-attach `TranslationalStateC`
    // lifted through the parent's `IntegOrigin`.
    let parent_pos_integ = parent_initial_trans().position;
    let parent_vel_integ = parent_initial_trans().velocity;
    let captured = jeod_sim::stage_detach_capture(
        parent_pos_integ + MOON_OFFSET,
        parent_vel_integ + MOON_VELOCITY,
        JeodQuat::identity(),
        DVec3::ZERO,
    );
    app.world_mut()
        .entity_mut(parent_entity)
        .insert(DetachedSubtreeStateC(captured));

    // Now fire the attach. The staging system's `parent_was_detached`
    // branch must re-stamp the parent's `DetachedSubtreeStateC` with
    // the merged composite in root-inertial.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: jeod_sim::Vec3Ext::m_at::<jeod_sim::StructuralFrame<jeod_sim::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: DMat3::IDENTITY,
        });
    app.world_mut().run_schedule(FixedUpdate);

    // Independent kernel run with lifted inputs.
    let combined_mass = app
        .world()
        .resource::<MassTreeR>()
        .0
        .get(id_parent)
        .composite_properties;

    let q = JeodQuat::identity();
    let parent_mass_props = parent_mass();
    let expected = jeod_sim::stage_attach_combine(StageAttachInputs {
        parent_position: parent_initial_trans().position + MOON_OFFSET,
        parent_velocity: parent_initial_trans().velocity + MOON_VELOCITY,
        parent_quaternion: q,
        parent_ang_vel_body: DVec3::ZERO,
        parent_mass: parent_mass_props,
        orig_parent_cm_struct: parent_mass_props.position,
        parent_t_inertial_struct: DMat3::IDENTITY,
        child_position: child_initial_trans().position + MOON_OFFSET,
        child_velocity: child_initial_trans().velocity + MOON_VELOCITY,
        child_quaternion: q,
        child_ang_vel_body: DVec3::ZERO,
        child_mass: child_mass(),
        combined_mass,
    });

    let detached = app
        .world()
        .get::<DetachedSubtreeStateC>(parent_entity)
        .expect(
            "parent_was_detached branch must keep DetachedSubtreeStateC on parent after attach",
        );
    let detached_pos_root = detached.0.composite_position.raw_si();
    let detached_vel_root = detached.0.composite_velocity.raw_si();

    assert!(
        (detached_pos_root - expected.position).length() < 1e-6,
        "parent_was_detached: DetachedSubtreeStateC.composite_position must equal kernel \
         output in root-inertial:\n  got {:?}\n  expected {:?}\n  delta {:?}",
        detached_pos_root,
        expected.position,
        detached_pos_root - expected.position,
    );
    assert!(
        (detached_vel_root - expected.velocity).length() < 1e-9,
        "parent_was_detached: DetachedSubtreeStateC.composite_velocity must equal kernel \
         output in root-inertial:\n  got {:?}\n  expected {:?}\n  delta {:?}",
        detached_vel_root,
        expected.velocity,
        detached_vel_root - expected.velocity,
    );

    // A regression that re-introduces the redundant `+IntegOrigin`
    // shift in this branch would land the value at
    // `expected + MOON_OFFSET` (and `expected.velocity +
    // MOON_VELOCITY` for velocity) — far above tolerance.
    let double_shift_alias_pos = expected.position + MOON_OFFSET;
    let double_shift_alias_vel = expected.velocity + MOON_VELOCITY;
    let dist_pos = (detached_pos_root - double_shift_alias_pos).length();
    let dist_vel = (detached_vel_root - double_shift_alias_vel).length();
    assert!(
        dist_pos > 1.0,
        "parent_was_detached: composite_position matches the 'double-shift' \
         bug-mode alias (off by +MOON_OFFSET) — got {detached_pos_root:?}, \
         alias {double_shift_alias_pos:?}",
    );
    assert!(
        dist_vel > 1.0,
        "parent_was_detached: composite_velocity matches the 'double-shift' \
         bug-mode alias (off by +MOON_VELOCITY) — got {detached_vel_root:?}, \
         alias {double_shift_alias_vel:?}",
    );
}

/// Cross-source attach: the parent integrates in `PlanetInertial<Moon>`
/// (non-zero `IntegOrigin == (MOON_OFFSET, MOON_VELOCITY)`) while the
/// child integrates at root (`IntegSourceC` absent, `IntegOrigin == 0`).
///
/// Cross-integration-frame attaches are gated by the staging fence:
/// `staging_system` rejects an `AttachEvent` whose two bodies' live
/// integ-frame entities differ, because the corresponding frame-tree
/// reparent + coordinate-rewrite is not yet implemented and allowing
/// the merge would silently corrupt downstream `RelativeFrameState`
/// walks. JEOD's `dyn_body_attach.cc::attach_establish_links` calls
/// `set_integ_frame` to perform that reparent recursively over the
/// child's frame entities; until our `staging_system` ports that
/// step, the cross-source path must fail loud.
///
/// This test pins the gated panic for the cross-source scenario so a
/// future change that quietly relaxes the fence without also wiring
/// in the frame-tree reparent is caught immediately. The companion
/// guard `bevy_attach_cross_integ_frame_panics_with_fail_loud_diagnostic`
/// in `bevy_parity_attach_detach_momentum.rs` covers the general
/// cross-frame case with two distinct gravity sources; this variant
/// pins the asymmetric `Some(moon)` vs `None` configuration where one
/// side resolves to the root frame entity and the other to a planet's
/// inertial frame.
///
/// Once the frame-tree reparent lands, this test is replaced with
/// positive coverage that the per-body lift uses each body's *own*
/// integ-origin (parent gets `+MOON_OFFSET / +MOON_VELOCITY`, child
/// gets `+0 / +0`) and the merged composite lands within the 1e-6 m
/// / 1e-9 m·s⁻¹ tolerances of the bug-mode aliases.
#[test]
#[should_panic(expected = "AttachEvent: parent")]
fn bevy_parity_attach_non_root_integ_source_per_body_lift_distinct_sources() {
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
        .insert(SourceInertialVelocityC::default())
        .id();

    let mut tree = MassTree::new();
    let id_parent = tree.add_body("Parent".into(), parent_mass());
    let id_child = tree.add_body("Child".into(), child_mass());
    app.insert_resource(MassTreeR(tree));

    // Parent: Moon-integrated, planet-relative state.
    let parent_pos_integ = parent_initial_trans().position;
    let parent_vel_integ = parent_initial_trans().velocity;
    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC(six_dof_config()),
            MassPropertiesC::from(parent_mass()),
            MassBodyIdC(id_parent),
            TranslationalStateC::from(parent_initial_trans()),
            RotationalStateC::from(initial_rot()),
            FrameDerivativesC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
            IntegSourceC(Some(moon)),
        ))
        .id();

    // Child: root-integrated (no `IntegSourceC`). Author the child's
    // root-inertial state to coincide with the parent's *lifted*
    // root-inertial state plus the same `+1.5 m/s ẑ` delta the
    // same-source test uses — keeps the merge non-degenerate (mass-
    // weighted velocity does real work) and produces the same kernel
    // expected output if and only if the per-body lift uses each
    // body's own origin.
    let child_root_pos = parent_pos_integ + MOON_OFFSET;
    let child_root_vel = parent_vel_integ + MOON_VELOCITY + DVec3::new(0.0, 0.0, 1.5);
    let child_entity = app
        .world_mut()
        .spawn((
            Name::new("Child"),
            DynamicsConfigC(six_dof_config()),
            MassPropertiesC::from(child_mass()),
            MassBodyIdC(id_child),
            TranslationalStateC::from(TranslationalState {
                position: child_root_pos,
                velocity: child_root_vel,
            }),
            RotationalStateC::from(initial_rot()),
            FrameDerivativesC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
            // No IntegSourceC: defaults to root.
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    // Park the Moon at a non-zero inertial state so the parent's
    // integ-origin is non-zero in root-inertial.
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator| {
            m.set_source_state(moon, MOON_OFFSET, MOON_VELOCITY);
        });
    app.world_mut().run_system(sys).unwrap();

    // Fire the attach.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: child_entity,
            parent: parent_entity,
            offset: jeod_sim::Vec3Ext::m_at::<jeod_sim::StructuralFrame<jeod_sim::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: DMat3::IDENTITY,
        });
    app.world_mut().run_schedule(FixedUpdate);

    // Independent kernel run with each body lifted through *its own*
    // integ-origin: parent gets `+MOON_*`, child stays at its
    // root-inertial state.
    let combined_mass = app
        .world()
        .resource::<MassTreeR>()
        .0
        .get(id_parent)
        .composite_properties;

    let q = JeodQuat::identity();
    let parent_mass_props = parent_mass();
    let expected = jeod_sim::stage_attach_combine(StageAttachInputs {
        parent_position: parent_pos_integ + MOON_OFFSET,
        parent_velocity: parent_vel_integ + MOON_VELOCITY,
        parent_quaternion: q,
        parent_ang_vel_body: DVec3::ZERO,
        parent_mass: parent_mass_props,
        orig_parent_cm_struct: parent_mass_props.position,
        parent_t_inertial_struct: DMat3::IDENTITY,
        // Child is root-integrated: lifted state == raw stored state.
        child_position: child_root_pos,
        child_velocity: child_root_vel,
        child_quaternion: q,
        child_ang_vel_body: DVec3::ZERO,
        child_mass: child_mass(),
        combined_mass,
    });

    // Lower the kernel's root-inertial output through the *parent's*
    // integ-origin (the writeback uses the parent's origin only — the
    // post-attach composite lives in the parent's storage frame).
    let expected_pos_integ = expected.position - MOON_OFFSET;
    let expected_vel_integ = expected.velocity - MOON_VELOCITY;

    let post_pos_integ = app
        .world()
        .get::<TranslationalStateC>(parent_entity)
        .unwrap()
        .0
        .position
        .raw_si();
    let post_vel_integ = app
        .world()
        .get::<TranslationalStateC>(parent_entity)
        .unwrap()
        .0
        .velocity
        .raw_si();

    assert!(
        (post_pos_integ - expected_pos_integ).length() < 1e-6,
        "cross-source attach: TranslationalStateC.position must equal kernel output \
         lowered through parent's IntegOrigin:\n  got {post_pos_integ:?}\n  \
         expected {expected_pos_integ:?}\n  delta {:?}",
        post_pos_integ - expected_pos_integ,
    );
    assert!(
        (post_vel_integ - expected_vel_integ).length() < 1e-9,
        "cross-source attach: TranslationalStateC.velocity must equal kernel output \
         lowered through parent's IntegOrigin:\n  got {post_vel_integ:?}\n  \
         expected {expected_vel_integ:?}\n  delta {:?}",
        post_vel_integ - expected_vel_integ,
    );

    // Bug-mode alias 1: the per-body lift accidentally reused the
    // parent's origin for the child (so the child's lifted position
    // would be `child_root + MOON_OFFSET` instead of `child_root`).
    // The kernel's mass-weighted merge would then differ from the
    // correct expected by the child-mass fraction times `MOON_OFFSET`.
    let m_p = parent_mass_props.mass;
    let m_c = child_mass().mass;
    let m_total = m_p + m_c;
    let child_share = m_c / m_total;
    let alias_swap_child_pos = (expected.position + child_share * MOON_OFFSET) - MOON_OFFSET;
    let alias_swap_child_vel = (expected.velocity + child_share * MOON_VELOCITY) - MOON_VELOCITY;
    assert!(
        (post_pos_integ - alias_swap_child_pos).length() > 1.0,
        "cross-source attach: TranslationalStateC.position matches the \
         'parent's origin reused for child' bug-mode alias (off by \
         child_share * MOON_OFFSET ≈ {:.0} m). got {post_pos_integ:?}, \
         alias {alias_swap_child_pos:?}",
        (child_share * MOON_OFFSET).length(),
    );
    assert!(
        (post_vel_integ - alias_swap_child_vel).length() > 1.0,
        "cross-source attach: TranslationalStateC.velocity matches the \
         'parent's origin reused for child' bug-mode alias. \
         got {post_vel_integ:?}, alias {alias_swap_child_vel:?}",
    );

    // Bug-mode alias 2: the per-body lift accidentally reused the
    // child's origin (zero) for the parent. The kernel's mass-
    // weighted merge would be off by the parent-mass fraction times
    // `MOON_OFFSET` in the opposite direction.
    let parent_share = m_p / m_total;
    let alias_swap_parent_pos = (expected.position - parent_share * MOON_OFFSET) - MOON_OFFSET;
    let alias_swap_parent_vel = (expected.velocity - parent_share * MOON_VELOCITY) - MOON_VELOCITY;
    assert!(
        (post_pos_integ - alias_swap_parent_pos).length() > 1.0,
        "cross-source attach: TranslationalStateC.position matches the \
         'child's origin reused for parent' bug-mode alias. \
         got {post_pos_integ:?}, alias {alias_swap_parent_pos:?}",
    );
    assert!(
        (post_vel_integ - alias_swap_parent_vel).length() > 1.0,
        "cross-source attach: TranslationalStateC.velocity matches the \
         'child's origin reused for parent' bug-mode alias. \
         got {post_vel_integ:?}, alias {alias_swap_parent_vel:?}",
    );
}
