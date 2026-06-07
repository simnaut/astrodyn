//! Negative test for IN.09: a body carrying `SolarBetaC` must not
//! survive into the per-tick pipeline without a matching `SunMarker`
//! entity in the world.
//!
//! Catalogued in `docs/JEOD_invariants.md` row IN.09. Two enforcement
//! sites cover this invariant:
//! - The startup-pass `validate_jeod_invariants::<P>` panics with
//!   `"Entity {entity:?}: SolarBetaC present but no SunMarker entity
//!   exists."` for bodies that pass through the validator
//!   (`crates/astrodyn_bevy/src/validation.rs`).
//! - The per-step `solar_beta_system` panics with
//!   `"{entity:?} solar beta: no SunMarker entity exists in the
//!   World"` as a fallback for bodies that bypass validation (e.g. a
//!   body with `SolarBetaC` but no `GravityControlsC`, or a
//!   `SolarBetaC` inserted after the body's spawn tick)
//!   (`crates/astrodyn_bevy/src/systems/derived_state.rs`).
//!
//! This integration test drives the validator site by going through
//! the public `VehicleBuilder` API + the full `AstrodynPlugin`
//! schedule — mirroring the AT.03 negative test's shape. The
//! in-module unit test
//! `derived_state::tests::solar_beta_missing_sun_marker_panics_with_caller_fix`
//! drives the per-step site in isolation. Both messages share the
//! `"no SunMarker entity exists"` substring so this test is robust
//! to a future refactor that reorders the two sites.
//!
//! The pre-#535 silent fallback was `SolarBeta::default() = 0.0` —
//! the geometrically-plausible "perfectly noon" value that downstream
//! thermal / power / pointing budgets cannot distinguish from a real
//! computation.

use std::time::Duration;

use astrodyn::{
    F64Ext, GravityControl, GravityGradient, RootInertial, TranslationalStateTyped, Vec3Ext,
    VehicleBuilder, EARTH,
};
use astrodyn_bevy::{AstrodynPlugin, IntegrationDtR, PlanetBundle, VehicleConfigBevyExt};
use bevy::prelude::*;
use glam::DVec3;

const DT: f64 = 60.0;

fn iss_trans() -> TranslationalStateTyped<RootInertial> {
    TranslationalStateTyped::<RootInertial> {
        position: DVec3::new(6_778_137.0, 0.0, 0.0).m_at::<RootInertial>(),
        velocity: DVec3::new(0.0, 7668.56, 0.0).m_per_s_at::<RootInertial>(),
    }
}

fn iss_mass_kg() -> uom::si::f64::Mass {
    400_000.0.kg()
}

fn step(app: &mut App) {
    app.world_mut().run_schedule(Startup);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(DT));
    app.world_mut().run_schedule(FixedUpdate);
}

#[test]
#[should_panic(expected = "no SunMarker entity exists")]
fn in_09_panics_on_solar_beta_without_sun_marker() {
    // JEOD_INV: IN.09 — a body requesting solar-beta derived state
    // (SolarBetaC) in a world that has zero SunMarker entities must
    // fail loudly rather than silently writing SolarBeta = 0.0. The
    // validator catches the misconfiguration on the first FixedUpdate
    // tick; the per-step solar_beta_system is the second-line safety
    // net for bodies that bypass validation (covered by the in-module
    // unit test in derived_state::tests).
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(Time::<Fixed>::from_seconds(DT))
        .insert_resource(IntegrationDtR(DT))
        .add_plugins(AstrodynPlugin);

    // Spawn an Earth gravity source. Deliberately no `SunBundle` /
    // `SunMarker` in the world — that is the misconfiguration this
    // test drives.
    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass_only(
            "Earth",
            astrodyn::GravitySource {
                mu: EARTH.shape.mu,
                model: astrodyn::GravityModel::PointMass,
            },
        ))
        .id();

    // Build a vehicle that requests solar-beta derived state. The
    // builder's `.solar_beta()` flips `derived.solar_beta = true`;
    // `spawn_bevy` then inserts `SolarBetaC::default()` on the body
    // (per `crates/astrodyn_bevy/src/lib.rs::spawn_bevy_inner` —
    // see the `derived.solar_beta` row in the rustdoc enumeration).
    let cfg = VehicleBuilder::new()
        .vehicle_named("in-09-solar-beta-requires-sun-marker-0")
        .with_translational(iss_trans())
        .three_dof_point_mass(iss_mass_kg())
        .rk4()
        .gravity(GravityControl::new_spherical(
            astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
            GravityGradient::Skip,
        ))
        .solar_beta()
        .build();

    {
        let mut commands = app.world_mut().commands();
        let _vehicle = cfg.spawn_bevy::<astrodyn::Earth>(&mut commands);
    }
    app.world_mut().flush();

    // Sanity precondition: the misconfiguration is actually wired —
    // the world has a body carrying SolarBetaC and zero SunMarker
    // entities. If either assertion fails the test is no longer
    // driving the IN.09 enforcement sites.
    {
        let world = app.world_mut();
        let mut solar_beta_q = world.query::<&astrodyn_bevy::SolarBetaC>();
        let solar_beta_count = solar_beta_q.iter(world).count();
        assert_eq!(
            solar_beta_count, 1,
            "test precondition: exactly one body must carry SolarBetaC for IN.09 to fire"
        );
        let mut sun_q = world.query::<&astrodyn_bevy::SunMarker>();
        let sun_count = sun_q.iter(world).count();
        assert_eq!(
            sun_count, 0,
            "test precondition: zero SunMarker entities must exist for IN.09 to fire"
        );
    }

    step(&mut app);
}
