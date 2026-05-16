// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Negative test for AT.03: `geodetic_system` must panic when
//! `GeodeticConfigC.planet` references an entity that does not carry
//! `PlanetFixedRotationC<P>`.
//!
//! Catalogued in `docs/JEOD_invariants.md` row AT.03 and enforced in
//! `crates/astrodyn_bevy/src/systems/derived_state.rs::geodetic_system`.
//! The pre-#523 fallback was a silent `GeodeticState::default()` write
//! (`(lat, lon, alt) = (0, 0, 0)`, the Gulf of Guinea) that downstream
//! consumers could not distinguish from a real fix. The post-#523
//! `assert!` names the offending body, the failed planet-entity lookup,
//! and the two-line remediation (spawn the planet source with a
//! non-`None` `rotation_model`, or hand-insert `PlanetFixedRotationC<P>`
//! on the existing planet entity).
//!
//! `PlanetBundle::point_mass_only` is the minimum-component bundle that
//! deliberately omits `PlanetFixedRotationC` (the bundle's docstring
//! enumerates the omission). Pointing `GeodeticConfigC.planet` at an
//! entity spawned from that bundle is the canonical misconfiguration
//! this test drives.

use std::time::Duration;

use astrodyn::{
    F64Ext, GravityControl, GravityGradient, RootInertial, TranslationalStateTyped, Vec3Ext,
    VehicleBuilder, EARTH,
};
use astrodyn_bevy::{
    AstrodynPlugin, GeodeticConfigC, IntegrationDtR, PlanetBundle, VehicleConfigBevyExt,
};
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

// JEOD_INV: AT.03 — drives the `geodetic_system` panic when
// `GeodeticConfigC.planet` resolves to an entity without
// `PlanetFixedRotationC<P>`. Spawning the planet via
// `PlanetBundle::point_mass_only` is the canonical misconfiguration:
// the bundle omits `PlanetFixedRotationC` precisely so callers that
// want point-mass-only physics don't pull in rotation-dependent
// per-step systems. Wiring `GeodeticConfigC` against that entity
// crosses the rotation boundary and the per-cause panic fires on
// the first `DerivedState` tick.
#[test]
#[should_panic(expected = "does not resolve to PlanetFixedRotationC")]
fn at_03_panics_on_geodetic_planet_without_planet_fixed_rotation() {
    // JEOD_INV: AT.03 — geodetic_system must panic when the planet
    // entity lacks PlanetFixedRotationC (in-body tag so the
    // invariant-coverage scanner picks it up regardless of how long
    // the doc-comment block above the #[should_panic] attribute grows).
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(Time::<Fixed>::from_seconds(DT))
        .insert_resource(IntegrationDtR(DT))
        .add_plugins(AstrodynPlugin);

    // `point_mass_only` deliberately omits `PlanetFixedRotationC`. Any
    // entity spawned from this bundle cannot satisfy `geodetic_system`'s
    // planet-query precondition.
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass_only(
            "Earth",
            astrodyn::GravitySource {
                mu: EARTH.shape.mu,
                model: astrodyn::GravityModel::PointMass,
            },
        ))
        .id();

    // Build a body that requests geodetic state pointing at the
    // rotation-less Earth entity. Gravity has to be wired so the
    // builder validates; we use the standard point-mass spherical
    // recipe. The `geodetic(0, &EARTH)` step copies the ellipsoid
    // radii into the resulting `GeodeticConfigC` and points its
    // `planet` field at the first source entity passed to
    // `spawn_bevy`.
    let cfg = VehicleBuilder::new()
        .with_translational(iss_trans())
        .three_dof_point_mass(iss_mass_kg())
        .rk4()
        .gravity(GravityControl::new_spherical(
            0_usize,
            GravityGradient::Skip,
        ))
        .geodetic(0, &EARTH)
        .build();

    {
        let mut commands = app.world_mut().commands();
        let vehicle = cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth]);
        // Force the deferred spawn so the GeodeticConfigC is present
        // before the first FixedUpdate runs; otherwise the body lands
        // mid-tick and the per-step system would see an empty query
        // on the failing tick.
        let _ = vehicle;
    }
    app.world_mut().flush();

    // Sanity precondition: the misconfiguration is wired and the
    // body's `GeodeticConfigC.planet` does point at the rotation-less
    // Earth. If this fails the test is no longer driving the AT.03
    // assert.
    let planet_field = {
        let world = app.world_mut();
        let mut q = world.query::<&GeodeticConfigC>();
        q.iter(world)
            .next()
            .expect("body must carry GeodeticConfigC after spawn_bevy")
            .planet
    };
    assert_eq!(
        planet_field, earth,
        "GeodeticConfigC.planet must point at the rotation-less Earth entity for AT.03 to fire"
    );

    step(&mut app);
}
