//! Regression tests pinning the panic-on-detection contract of
//! `validate_jeod_invariants` (closes #495).
//!
//! Before this conversion the validator emitted `bevy::log::warn!()` for
//! the two `is_warning()`-classified `ValidationError` variants
//! (`UninitializedState`, `NonRootFrameWithRootDependentFeatures`) and
//! the entity continued through the integration pipeline. Per the
//! project's "Fail Loudly" non-negotiable (`CLAUDE.md`), validation is
//! now a hard gate: every detected `ValidationError` panics with a
//! diagnostic that names the offending entity, the specific failure,
//! and how to fix the call site.
//!
//! Coverage:
//!
//! 1. `validate_jeod_invariants_panics_on_uninitialized_translational_state`
//!    pins the panic for a body spawned with zero `TranslationalStateC`
//!    (the `is_likely_uninitialized` path in `astrodyn::validate_body`).
//! 2. `validate_jeod_invariants_panics_on_non_root_integ_with_drag`
//!    pins the panic for a body integrating in a non-root inertial frame
//!    with a root-dependent feature (`DragConfigC` here — a non-shift
//!    site per RF.10) attached. The current Bevy adapter previously
//!    emitted a `FAIL_LOUD_EXEMPT` warning here; the panic-on-detection
//!    contract supersedes that deviation.
//! 3. `validate_jeod_invariants_panics_on_rotational_without_mass` pins
//!    a representative fatal-class panic so we verify the unified
//!    `"fails component validation"` diagnostic prefix is shared by both
//!    the fatal-class and warning-class paths.
//!
//! Each test drives a single `FixedUpdate` tick inside
//! `std::panic::catch_unwind` and asserts the panic payload contains
//! both the "fails component validation" prefix and a substring
//! distinguishing the specific failure. The Bevy scheduler runs
//! systems in parallel, so other systems may also panic in worker
//! threads downstream of the validation panic — `catch_unwind` lets
//! us inspect the actual validator panic instead of whichever payload
//! Bevy's parallel executor happens to surface to the main thread last.

use std::panic::AssertUnwindSafe;
use std::time::Duration;

use astrodyn::{
    DynamicsConfig, Earth, GravityControl, GravityControls, GravityGradient, MassProperties,
    PlanetInertial, TranslationalStateTyped, Vec3Ext, EARTH,
};
use astrodyn_bevy::{
    AstrodynPlugin, DynamicsConfigC, GravityAccelerationC, GravityControlsC, IntegrationDtR,
    MassPropertiesC, PlanetBundle, TranslationalStateC,
};
use bevy::ecs::schedule::ExecutorKind;
use bevy::prelude::*;
use glam::DVec3;

const DT: f64 = 10.0;

fn build_minimal_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(Time::<Fixed>::from_seconds(DT))
        .insert_resource(IntegrationDtR(DT))
        .add_plugins(AstrodynPlugin);
    // Force single-threaded execution for the schedules we drive in
    // these tests so the first panic short-circuits the rest of the
    // tick. Bevy's default multi-threaded executor lets queued
    // worker-thread systems finish (and panic themselves) before the
    // schedule aborts, which makes `catch_unwind` surface whichever
    // payload happens to be reported last on the main thread instead
    // of the validator's panic. Single-threaded execution is
    // semantically equivalent here (every test panics in the same
    // schedule pass) and gives deterministic catch_unwind payloads.
    app.edit_schedule(Startup, |s| {
        s.set_executor_kind(ExecutorKind::SingleThreaded);
    });
    app.edit_schedule(FixedUpdate, |s| {
        s.set_executor_kind(ExecutorKind::SingleThreaded);
    });
    app
}

/// Drive one `FixedUpdate` tick inside `catch_unwind`, returning the
/// panic payload as a `String`. Bevy's parallel scheduler may surface
/// multiple panics from worker threads downstream of the validation
/// panic; the `build_minimal_app` helper pins both `Startup` and
/// `FixedUpdate` to the single-threaded executor so the first panic
/// short-circuits the rest of the tick and gives a deterministic
/// `catch_unwind` payload. The caller must have already advanced past
/// `Startup` (either via `app.update()` or
/// `app.world_mut().run_schedule(Startup)`) before invoking this
/// helper.
fn collect_first_panic(app: &mut App) -> String {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }));
    let payload = result.expect_err(
        "expected validation to panic on the misconfigured body, but the FixedUpdate \
         schedule completed without error — validate_jeod_invariants is no longer \
         enforcing the failure or the body was filtered out of its query",
    );
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&'static str>()
                .map(|s| (*s).to_string())
        })
        .unwrap_or_else(|| "<non-string panic payload>".to_string())
}

/// A body with `TranslationalStateC<Earth>` left at all-zero
/// position/velocity is misconfigured — the integrator would treat the
/// origin as a valid initial condition and silently propagate from
/// (0, 0, 0). `astrodyn::validate_body` reports this as
/// `ValidationError::UninitializedState`, which is the canonical
/// warning-class error the previous `warn!()` path swallowed.
#[test]
fn validate_jeod_invariants_panics_on_uninitialized_translational_state() {
    let mut app = build_minimal_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<Earth>::point_mass("Earth", &EARTH))
        .id();

    // Spawn a body whose `TranslationalStateC<Earth>` is zero — that's
    // the trip wire `is_likely_uninitialized` catches inside
    // `astrodyn::validate_body`.
    app.world_mut().spawn((
        Name::new("Bogus"),
        TranslationalStateC::<Earth>(TranslationalStateTyped::<PlanetInertial<Earth>> {
            position: DVec3::ZERO.m_at::<PlanetInertial<Earth>>(),
            velocity: DVec3::ZERO.m_per_s_at::<PlanetInertial<Earth>>(),
        }),
        MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
            &MassProperties::with_inertia(
                1_000.0,
                glam::DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0)),
                DVec3::ZERO,
            ),
        )),
        DynamicsConfigC(DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        }),
        GravityAccelerationC::default(),
        GravityControlsC(GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        }),
    ));

    // Run Startup so frame-tree registration completes before
    // `Added<GravityControlsC>` fires the validator on `FixedUpdate`.
    app.world_mut().run_schedule(Startup);

    let msg = collect_first_panic(&mut app);
    assert!(
        msg.contains("fails component validation"),
        "panic message did not name the validation surface: {msg}"
    );
    assert!(
        msg.contains("uninitialized") || msg.contains("Translational state"),
        "panic message did not name the UninitializedState diagnostic: {msg}"
    );
}

/// A body integrating in a non-root inertial frame (here: the Moon's
/// inertial frame) that also carries a root-dependent feature
/// component is the configuration the RF.10 mismatch warning catches.
/// `DragConfigC` is a NON-SHIFT site in the RF.10 taxonomy — the body's
/// `Position<PlanetInertial<P>>` is consumed directly by the drag
/// kernel, so mixing it with a non-root integration frame leaves the
/// caller responsible for the RF.10 shift discipline. The panic stops
/// the misconfiguration at the validation tick.
#[test]
fn validate_jeod_invariants_panics_on_non_root_integ_with_drag() {
    use astrodyn::{DragConfig, RootInertial, RotationalState, SourceHandle, VehicleBuilder, MOON};
    use astrodyn_bevy::{SourceInertialVelocityC, SourceMutator, VehicleConfigBevyExt};

    let mut app = build_minimal_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<Earth>::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::<Earth>::point_mass("Moon", &MOON))
        .insert(SourceInertialVelocityC::default())
        .id();

    // Run Startup so `register_source_frames_system` materializes the
    // Moon's frame entity and parents it under the root frame, then
    // offset the Moon to 3.84e8 m so the frame check classifies its
    // inertial frame as genuinely non-root (an identity-state child of
    // root would otherwise be folded back onto "root-equivalent" by
    // `is_root_equivalent_entity`).
    app.world_mut().run_schedule(Startup);
    let mutator_sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator<Earth>| {
            m.set_source_position(moon, DVec3::new(3.84e8, 0.0, 0.0));
        });
    app.world_mut().run_system(mutator_sys).unwrap();

    // Build a vehicle through the public typed surface so it lands
    // with the full set of frame-tree components (`FrameEntityC`,
    // `ChildOf` wiring, etc.) the non-root check inspects. The
    // misconfiguration is the pairing of `integ_source(moon)` —
    // which lifts the body into the Moon's (non-root) inertial frame
    // — with a `DragConfigC`, a root-dependent non-shift consumer
    // per RF.10.
    let cfg = VehicleBuilder::new()
        .with_translational(astrodyn::TranslationalStateTyped::<RootInertial> {
            position: DVec3::new(1_837_400.0, 0.0, 0.0).m_at::<RootInertial>(),
            velocity: DVec3::new(0.0, 1_600.0, 0.0).m_per_s_at::<RootInertial>(),
        })
        .sixdof(
            RotationalState {
                quaternion: astrodyn::JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            },
            MassProperties::with_inertia(
                1_000.0,
                glam::DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0)),
                DVec3::ZERO,
            ),
        )
        .rk4()
        .gravity(GravityControl::new_spherical(
            SourceHandle::index(1),
            GravityGradient::Skip,
        ))
        .integ_source(SourceHandle::index(1))
        .drag(DragConfig {
            cd: 2.2,
            area: 10.0,
            constant_density: None,
        })
        .build();

    {
        let mut commands_queue = app.world_mut().commands();
        cfg.spawn_bevy::<Earth>(&mut commands_queue, &[earth, moon]);
    }
    app.world_mut().flush();

    let msg = collect_first_panic(&mut app);
    assert!(
        msg.contains("fails component validation"),
        "panic message did not name the validation surface: {msg}"
    );
    assert!(
        msg.contains("non-root") || msg.contains("RF.10") || msg.contains("drag="),
        "panic message did not name the RF.10 non-root + root-dependent diagnostic: {msg}"
    );
}

/// `rotational_dynamics=true` without a `MassPropertiesC` is a
/// fatal-class `ValidationError::RotationalWithoutMass`. The same
/// panic-on-detection contract that catches the warning-class cases
/// should also fire here — pinning the unified `"fails component
/// validation"` diagnostic prefix proves both fatal and warning-class
/// errors share the same caller-facing message.
#[test]
fn validate_jeod_invariants_panics_on_rotational_without_mass() {
    let mut app = build_minimal_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<Earth>::point_mass("Earth", &EARTH))
        .id();

    // 6-DOF body without `MassPropertiesC` — `validate_body` reports
    // `RotationalWithoutMass` (a fatal-class error).
    app.world_mut().spawn((
        Name::new("MisconfiguredSixDof"),
        TranslationalStateC::<Earth>(TranslationalStateTyped::<PlanetInertial<Earth>> {
            position: DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<PlanetInertial<Earth>>(),
            velocity: DVec3::new(0.0, 7_000.0, 0.0).m_per_s_at::<PlanetInertial<Earth>>(),
        }),
        DynamicsConfigC(DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        }),
        GravityAccelerationC::default(),
        GravityControlsC(GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        }),
    ));

    // Run Startup so frame-tree registration completes before
    // `Added<GravityControlsC>` fires the validator on `FixedUpdate`.
    app.world_mut().run_schedule(Startup);

    let msg = collect_first_panic(&mut app);
    assert!(
        msg.contains("fails component validation"),
        "panic message did not name the validation surface: {msg}"
    );
    assert!(
        msg.contains("rotational_dynamics") || msg.contains("RotationalState"),
        "panic message did not name the rotational-without-mass diagnostic: {msg}"
    );
}
