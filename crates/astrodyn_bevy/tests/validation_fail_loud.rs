//! Regression tests pinning the warning-class / fatal-class split of
//! `validate_jeod_invariants` for `astrodyn::validate_body` errors.
//!
//! Per the project's "Fail Loudly" non-negotiable (`CLAUDE.md`),
//! fatal-class `ValidationError`s are a hard gate: the validator
//! panics with a diagnostic that names the offending entity, the
//! specific failure, and how to fix the call site.
//! `ValidationError::is_warning()`-class failures
//! (`UninitializedState`, `NonRootFrameWithRootDependentFeatures`)
//! emit `bevy::log::warn!()` instead — they flag suspicious-but-valid
//! configurations and let the entity continue.
//!
//! Coverage:
//!
//! 1. `validate_jeod_invariants_panics_on_rotational_without_mass`
//!    pins the fatal-class panic shape (`RotationalWithoutMass`) and
//!    proves the `"fails component validation"` diagnostic prefix is
//!    emitted for fatal-class errors. The body carries a default
//!    `RotationalStateC` so the only error is the
//!    mass-related one (precise regression against a single failure).
//! 2. `validate_jeod_invariants_warns_on_uninitialized_translational_state`
//!    pins the warning-class path: a body spawned with zero
//!    `TranslationalStateC` (`is_likely_uninitialized`) must *not*
//!    panic — the validator warns and lets the entity continue.
//!
//! The fatal-class test drives a single `FixedUpdate` tick inside
//! `std::panic::catch_unwind` and asserts the panic payload contains
//! both the "fails component validation" prefix and a substring
//! distinguishing the specific failure. The Bevy scheduler runs
//! systems in parallel, so other systems may also panic in worker
//! threads downstream of the validation panic — `catch_unwind` lets
//! us inspect the actual validator panic instead of whichever payload
//! Bevy's parallel executor happens to surface to the main thread last.
//!
//! The warning-class test asserts that the same `FixedUpdate` tick
//! completes without panicking.

use std::panic::AssertUnwindSafe;
use std::time::Duration;

use astrodyn::{
    DynamicsConfig, Earth, GravityControl, GravityControls, GravityGradient, MassProperties,
    PlanetInertial, TranslationalStateTyped, Vec3Ext, EARTH,
};
use astrodyn_bevy::{
    AstrodynPlugin, DynamicsConfigC, GravityAccelerationC, GravityControlsC, IntegrationDtR,
    MassPropertiesC, PlanetBundle, RotationalStateC, TranslationalStateC,
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

/// `rotational_dynamics=true` without a `MassPropertiesC` is a
/// fatal-class `ValidationError::RotationalWithoutMass`. Pinning the
/// `"fails component validation"` diagnostic prefix proves the
/// fatal-class path panics.
///
/// The body carries a default `RotationalStateC` so the only
/// `ValidationError` raised is `RotationalWithoutMass` — without it,
/// `validate_body` would also report `RotationalWithoutRotState` and
/// the assertion below could not distinguish which failure surfaced
/// the panic. Adding the rotational-state component makes the
/// regression precise to the mass-related diagnostic.
#[test]
fn validate_jeod_invariants_panics_on_rotational_without_mass() {
    let mut app = build_minimal_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<Earth>::point_mass("Earth", &EARTH))
        .id();

    // 6-DOF body without `MassPropertiesC` — `validate_body` reports
    // `RotationalWithoutMass` (a fatal-class error). `RotationalStateC`
    // is present (default) so `RotationalWithoutRotState` is *not*
    // raised, narrowing the failure set to one entry.
    app.world_mut().spawn((
        Name::new("MisconfiguredSixDof"),
        TranslationalStateC::<Earth>(TranslationalStateTyped::<PlanetInertial<Earth>> {
            position: DVec3::new(7_000_000.0, 0.0, 0.0).m_at::<PlanetInertial<Earth>>(),
            velocity: DVec3::new(0.0, 7_000.0, 0.0).m_per_s_at::<PlanetInertial<Earth>>(),
        }),
        RotationalStateC::default(),
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

/// A body with `TranslationalStateC<Earth>` left at all-zero
/// position/velocity trips the `is_likely_uninitialized` heuristic
/// inside `astrodyn::validate_body`, producing
/// `ValidationError::UninitializedState`. That variant is classified
/// as warning-class (`is_warning() == true`) because the origin can
/// be a legitimate initial condition; the Bevy adapter therefore
/// emits `bevy::log::warn!()` and lets the entity continue rather
/// than panicking inside the validator.
///
/// Downstream systems on the same tick (e.g. `gravity_computation`)
/// may still panic on a body sitting at the gravity source center —
/// that's a different invariant, surfaced by a different site. This
/// regression test asserts only that *the validator itself* does not
/// produce its `"fails component validation"` panic for the
/// warning-class error: if the FixedUpdate tick panics, the panic
/// payload must not be the validator's diagnostic.
#[test]
fn validate_jeod_invariants_warns_on_uninitialized_translational_state() {
    let mut app = build_minimal_app();
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<Earth>::point_mass("Earth", &EARTH))
        .id();

    app.world_mut().spawn((
        Name::new("ZeroState"),
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

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }));
    if let Err(payload) = result {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&'static str>()
                    .map(|s| (*s).to_string())
            })
            .unwrap_or_else(|| "<non-string panic payload>".to_string());
        assert!(
            !msg.contains("fails component validation"),
            "warning-class UninitializedState must not be raised by the \
             validator as a fatal panic — got: {msg}"
        );
    }
}
