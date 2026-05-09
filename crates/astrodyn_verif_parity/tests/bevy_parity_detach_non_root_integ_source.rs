// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy ECS detach parity for bodies whose `IntegSourceC` is a
//! non-root planet (lunar-orbit integ frame). Pins the
//! integ→root-inertial lift the staging system applies before
//! capturing `DetachedSubtreeStateC` and the matching root→integ-frame
//! lower the step system applies before writing back to
//! `TranslationalStateC`.
//!
//! With #316's typing change `DetachedSubtreeState.composite_*` is
//! `Position/Velocity<RootInertial>` by witness. For a body that
//! integrates in a non-root `PlanetInertial<P>` frame, the
//! `TranslationalStateC` storage is planet-relative (integration-frame
//! coords); a detach handler that copies those raw values into a
//! `RootInertial`-typed slot would silently mislabel the frame. The
//! fix lifts through the body's `IntegOrigin` at the read site and
//! lowers through it at every writeback site (subtree-state →
//! `TranslationalStateC` ballistic-step writeback, parent_was_detached
//! re-stamp on attach, parent-side post-detach CoM-shift's detached
//! re-stamp). For root-integrated bodies the origin is identically
//! zero so the lift/lower pair is a numerical no-op; for non-root
//! integ_source it is load-bearing.
//!
//! What this test pins:
//!
//! 1. **Detach instant**: the captured `DetachedSubtreeStateC.composite_
//!    position` equals the body's pre-detach root-inertial position
//!    (i.e. integ-frame position + Moon's offset). Without the lift,
//!    the captured value would be raw integ-frame coords, off by
//!    `MOON_OFFSET` (~3.84e8 m).
//!
//! 2. **TranslationalStateC after writeback**: the synced
//!    `TranslationalStateC` still lives in the body's integration
//!    frame (the canonical storage convention for a non-root-
//!    integrated body). Without the writeback lower, the synced
//!    value would carry root-inertial coords, off by `MOON_OFFSET`
//!    in the opposite direction.
//!
//! Mirrors the runner's lift/lower pattern in
//! `crates/astrodyn_runner/src/simulation/mass_tree.rs:567-585` (lift
//! before the rigid-body chain walk) and `:681-688` (lower at the
//! writeback to the body's typed storage).
//!
//! The runner's `Simulation::detach` resumes integrated dynamics on
//! the detached child, while the Bevy adapter switches the child to
//! ballistic propagation via `DetachedSubtreeStateC`. The two are
//! deliberately asymmetric for non-zero force fields, so this test
//! pins only the **detach instant** and the **first ballistic-step
//! writeback** — points where both runtimes' captured / synced state
//! must agree on coordinates and frame label.

use astrodyn::{
    DynamicsConfig, GravityControls, JeodQuat, MassProperties, MassTree, RotationalState,
    TranslationalState, EARTH, MOON,
};
use astrodyn_bevy::{
    AstrodynPlugin, AttachEvent, DetachEvent, DetachedSubtreeStateC, DynamicsConfigC,
    FrameDerivativesC, GravityControlsC, IntegSourceC, MassBodyIdC, MassPropertiesC, MassTreeR,
    PlanetBundle, RotationalStateC, SourceInertialVelocityC, SourceMutator, TranslationalStateC,
};
use bevy::prelude::*;
use glam::{DMat3, DVec3};
use std::time::Duration;

const DT: f64 = 60.0;
const NUM_STEPS_BEFORE_DETACH: usize = 5;
const MOON_OFFSET: DVec3 = DVec3::new(3.844e8, 0.0, 0.0);
/// Non-zero Moon inertial velocity for the velocity-shift variant.
/// Picked along y so it is orthogonal to `MOON_OFFSET` (along x) —
/// orthogonal so a bug that mixes the position and velocity branches
/// of the lift/lower pair leaves a measurable residual in the
/// component the failing branch would not have touched. Magnitude
/// ~1 km/s, well above any conceivable f64 round-off in the lift/
/// lower chain. A bug that drops the velocity term of the lift would
/// mislabel an integ-frame velocity as root-inertial and leak the
/// full 1 km/s into `composite_velocity` — orders of magnitude above
/// the 1e-9 tolerance.
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

/// 100 km lunar circular orbit (in moon-centered coordinates).
fn lunar_initial_trans() -> TranslationalState {
    let r = 1_837_400.0;
    let v = (MOON.shape.mu / r).sqrt();
    TranslationalState {
        position: DVec3::new(r, 0.0, 0.0),
        velocity: DVec3::new(0.0, v, 0.0),
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

/// Shared body of both lift/lower variants.
///
/// `moon_velocity` is the Moon's inertial velocity in the simulation's
/// root-inertial frame. The zero-velocity variant exercises the position
/// shift only; the non-zero variant pins the **velocity** branch of the
/// lift/lower pair. A bug that drops the velocity term of the read-side
/// lift would let the captured `DetachedSubtreeState.composite_velocity`
/// inherit an integ-frame velocity (off by `MOON_VELOCITY`); a bug that
/// drops the velocity term of the writeback lower would let the synced
/// `TranslationalStateC.velocity` carry root-inertial coords (off by
/// `MOON_VELOCITY` in the opposite direction). Both branches must pass.
fn run_lift_and_lower(moon_velocity: DVec3) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Moon", &MOON))
        .insert(SourceInertialVelocityC::default())
        .id();

    // Mass-tree with two attachable bodies.
    let mut tree = MassTree::new();
    let id_parent = tree.add_body("Parent".into(), parent_mass());
    let id_child = tree.add_body("Child".into(), child_mass());
    app.insert_resource(MassTreeR(tree));

    // Both bodies share lunar-orbit state and integrate in Moon's
    // inertial frame. Soft-merge invariant: the rigid composite
    // recovers the shared input at the detach instant.
    let parent_entity = app
        .world_mut()
        .spawn((
            Name::new("Parent"),
            DynamicsConfigC(six_dof_config()),
            MassPropertiesC::from(astrodyn_bevy::typed_bridge::mass_raw_to_self_ref(
                &(parent_mass()),
            )),
            MassBodyIdC(id_parent),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(lunar_initial_trans()),
            RotationalStateC::from(astrodyn_bevy::typed_bridge::rot_raw_to_self_ref(
                &(initial_rot()),
            )),
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
            MassPropertiesC::from(astrodyn_bevy::typed_bridge::mass_raw_to_self_ref(
                &(child_mass()),
            )),
            MassBodyIdC(id_child),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(lunar_initial_trans()),
            RotationalStateC::from(astrodyn_bevy::typed_bridge::rot_raw_to_self_ref(
                &(initial_rot()),
            )),
            FrameDerivativesC::default(),
            GravityControlsC(GravityControls { controls: vec![] }),
            IntegSourceC(Some(moon)),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    // Park the Moon at MOON_OFFSET with the supplied inertial velocity.
    // `set_source_state` writes both `SourceInertialPositionC` and
    // `SourceInertialVelocityC`, so the staging-system lift and the
    // step_detached-system lower both pick up the non-zero velocity
    // term via `body_integ_origin_in_root` (which reads through the
    // frame entity's `FrameTransC`).
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator<astrodyn::Earth>| {
            m.set_source_state(moon, MOON_OFFSET, moon_velocity);
        });
    app.world_mut().run_system(sys).unwrap();

    // Soft-merge: identical pre-state on both bodies, zero structural
    // offset. Pre-detach `TranslationalStateC` of the integrated tree
    // root equals `lunar_initial_trans()` (integ-frame coords).
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

    // Step a few ticks so the integrated parent advances under lunar
    // gravity (point-mass earth — but with empty gravity_controls,
    // bodies coast). The exact propagated state isn't load-bearing
    // for this test; what matters is that the staging-system detach
    // path runs against a non-trivial integ-frame `TranslationalStateC`.
    for _ in 0..NUM_STEPS_BEFORE_DETACH {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    // Snapshot the parent's `TranslationalStateC` *before* the detach
    // event fires. This is the integ-frame composite state the detach
    // handler will derive the child from.
    let parent_pre_detach_pos_integ = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(parent_entity)
        .unwrap()
        .0
        .position
        .raw_si();
    let parent_pre_detach_vel_integ = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(parent_entity)
        .unwrap()
        .0
        .velocity
        .raw_si();

    // Fire detach + step once. The `staging_system` consumes the
    // event, captures the child's composite-body state via
    // `propagate_forward` from the parent's pre-detach composite, and
    // inserts `DetachedSubtreeStateC`. Then `step_detached_system`
    // advances ballistically by `DT` and syncs `TranslationalStateC`.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent {
            child: child_entity,
        });
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);

    let detached = app
        .world()
        .get::<DetachedSubtreeStateC>(child_entity)
        .expect("DetachEvent should have inserted DetachedSubtreeStateC on the child");

    let detached_pos_root = detached.0.composite_position.raw_si();
    let detached_vel_root = detached.0.composite_velocity.raw_si();

    // Expected detached-state values:
    //
    // - At the detach instant the child's instantaneous composite-body
    //   inertial state equals the parent's composite (soft merge,
    //   zero offset → `propagate_forward` is a no-op step).
    // - The lift converts integ-frame to root-inertial:
    //     root_pos = integ_pos + MOON_OFFSET
    //     root_vel = integ_vel + moon_velocity
    // - `step_detached_system` then advances ballistically by DT:
    //     root_pos_after_step = root_pos_at_detach + root_vel * DT
    //     root_vel_after_step = root_vel  (ballistic, no force)
    let expected_root_pos_at_detach = parent_pre_detach_pos_integ + MOON_OFFSET;
    let expected_root_vel = parent_pre_detach_vel_integ + moon_velocity;
    let expected_detached_pos_root_after_step =
        expected_root_pos_at_detach + expected_root_vel * DT;

    // f64 round-off across the chain walk + ballistic step is ~1e-9
    // on a ~7e6 m-magnitude position; tolerate a few ULP per
    // coordinate but reject any error of order MOON_OFFSET (3.8e8) in
    // position or MOON_VELOCITY (1e3 m/s) in velocity.
    assert!(
        (detached_pos_root - expected_detached_pos_root_after_step).length() < 1e-6,
        "DetachedSubtreeStateC.composite_position not lifted to root-inertial:\n  \
         got {:?}\n  expected {:?} (= integ_pos_at_detach + MOON_OFFSET + vel·DT)\n  \
         delta {:?}",
        detached_pos_root,
        expected_detached_pos_root_after_step,
        detached_pos_root - expected_detached_pos_root_after_step,
    );
    assert!(
        (detached_vel_root - expected_root_vel).length() < 1e-9,
        "DetachedSubtreeStateC.composite_velocity not lifted by Moon velocity:\n  \
         got {:?}\n  expected {:?} (= integ_vel + moon_velocity)\n  \
         delta {:?}",
        detached_vel_root,
        expected_root_vel,
        detached_vel_root - expected_root_vel,
    );

    // Writeback side: the synced `TranslationalStateC` must carry
    // integration-frame coordinates after the lower. Position equals
    // `expected_root_pos_at_detach + vel·DT - MOON_OFFSET`; velocity
    // equals `expected_root_vel - moon_velocity`.
    let synced_pos_integ = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(child_entity)
        .unwrap()
        .0
        .position
        .raw_si();
    let synced_vel_integ = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(child_entity)
        .unwrap()
        .0
        .velocity
        .raw_si();

    let expected_synced_pos_integ = expected_detached_pos_root_after_step - MOON_OFFSET;
    let expected_synced_vel_integ = expected_root_vel - moon_velocity;
    assert!(
        (synced_pos_integ - expected_synced_pos_integ).length() < 1e-6,
        "TranslationalStateC.position not lowered to integ-frame after step_detached_system:\n  \
         got {:?}\n  expected {:?}\n  delta {:?}",
        synced_pos_integ,
        expected_synced_pos_integ,
        synced_pos_integ - expected_synced_pos_integ,
    );
    assert!(
        (synced_vel_integ - expected_synced_vel_integ).length() < 1e-9,
        "TranslationalStateC.velocity not lowered to integ-frame:\n  \
         got {:?}\n  expected {:?}\n  delta {:?}",
        synced_vel_integ,
        expected_synced_vel_integ,
        synced_vel_integ - expected_synced_vel_integ,
    );

    // The two stored values must differ by exactly MOON_OFFSET in
    // position and exactly `moon_velocity` in velocity — this is the
    // structural pin that catches a regression where one of the
    // lift/lower pair drops out:
    //
    // - if the read-side lift is dropped (in either component),
    //   `detached - synced` would carry only the ballistic-step delta
    //   (`-vel·DT` in position, `0` in velocity) instead of the
    //   integ-origin offset;
    // - if the writeback lower is dropped, `detached - synced` would
    //   carry `+vel·DT` in position (no MOON_OFFSET separation) and
    //   `0` in velocity (no `moon_velocity` separation).
    let pos_offset = detached_pos_root - synced_pos_integ;
    let vel_offset = detached_vel_root - synced_vel_integ;
    assert!(
        (pos_offset - MOON_OFFSET).length() < 1e-6,
        "DetachedSubtreeStateC vs TranslationalStateC position offset must equal MOON_OFFSET \
         (lift/lower symmetry):\n  got {:?}\n  expected {:?}",
        pos_offset,
        MOON_OFFSET,
    );
    assert!(
        (vel_offset - moon_velocity).length() < 1e-9,
        "DetachedSubtreeStateC vs TranslationalStateC velocity offset must equal Moon velocity \
         (lift/lower symmetry):\n  got {:?}\n  expected {:?}",
        vel_offset,
        moon_velocity,
    );
}

/// Detach a body whose `IntegSourceC` points at a non-root planet,
/// with the Moon at rest in root-inertial. Pins the **position**
/// branch of the lift/lower pair: a bug in the velocity branch would
/// pass silently because both terms reduce to zero.
#[test]
fn bevy_parity_detach_non_root_integ_source_lift_and_lower() {
    run_lift_and_lower(DVec3::ZERO);
}

/// Same scenario as `…_lift_and_lower`, but with the Moon moving at
/// `MOON_VELOCITY` in root-inertial. Pins the **velocity** branch of
/// the lift/lower pair: the captured `DetachedSubtreeStateC.composite_
/// velocity` must include `moon_velocity` (lift), and the synced
/// `TranslationalStateC.velocity` must subtract it back out at
/// writeback (lower). A regression that drops either velocity shift
/// would fail by ~1 km/s — orders of magnitude above the 1e-9
/// tolerance — while the zero-velocity sibling test would still pass.
#[test]
fn bevy_parity_detach_non_root_integ_source_lift_and_lower_with_source_velocity() {
    run_lift_and_lower(MOON_VELOCITY);
}
