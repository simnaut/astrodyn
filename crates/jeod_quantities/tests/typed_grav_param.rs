//! Tier 1 unit tests for [`GravParam<P>`]'s planet-phantom guards.
//!
//! These tests cover the positive runtime behavior of the new typed μ
//! surface (round-trip through `from_si`, `relabel`, and the
//! per-planet factories). The compile-fail counterparts — that
//! mismatched planet phantoms refuse to compile — live as
//! `compile_fail` rustdoc doctests on
//! [`jeod_quantities::dims::GravParam`] and on
//! [`jeod_math::OrbitalElements::from_cartesian_typed`], so any
//! regression that re-erases the planet phantom fails CI.

use jeod_quantities::prelude::*;

#[test]
fn from_si_pins_planet_phantom() {
    // The witness-gated constructor takes a planet via turbofish.
    let mu_earth = GravParam::<Earth>::from_si(3.986_004_415e14);
    assert_eq!(mu_earth.value, 3.986_004_415e14);
    // The same numeric value, pinned to a different planet, is a
    // distinct type — confirmed at runtime via `Debug` (the type-level
    // guard is exercised by the compile_fail doctest on the struct).
    let mu_sun = GravParam::<Sun>::from_si(1.327_124_400_18e20);
    assert!(format!("{mu_earth:?}").contains("Earth"));
    assert!(format!("{mu_sun:?}").contains("Sun"));
}

#[test]
fn m3_per_s2_for_pins_planet_phantom() {
    // The unit-rated factory matches `from_si` semantics; preferred
    // surface for mission code that wants the planet load-bearing on
    // the call site.
    let mu = 4.902_799_806_931_69e12_f64.m3_per_s2_for::<Moon>();
    assert_eq!(mu.value, 4.902_799_806_931_69e12);
}

#[test]
fn km3_per_s2_for_scales_to_si() {
    // 398600 km^3/s^2 = 3.986e14 m^3/s^2.
    let mu_e: GravParam<Earth> = 398_600.0_f64.km3_per_s2_for::<Earth>();
    assert!((mu_e.value - 3.986e14).abs() < 1e9);
}

#[test]
fn relabel_round_trip_preserves_value() {
    // The relabel boundary is the explicit escape hatch from the
    // dynamic-registry world (`SelfPlanet`) to a planet-pinned typed
    // consumer. It must preserve the SI value bit-for-bit.
    let mu_any: GravParam<SelfPlanet> = 3.986_004_415e14_f64.m3_per_s2();
    let mu_earth: GravParam<Earth> = mu_any.relabel();
    assert_eq!(mu_earth.value, mu_any.value);
}

#[test]
fn default_is_zero() {
    let zero: GravParam<Earth> = GravParam::default();
    assert_eq!(zero.value, 0.0);
}

#[test]
fn copy_clone_preserve_value() {
    let mu_earth = 3.986e14_f64.m3_per_s2_for::<Earth>();
    let copied = mu_earth;
    let cloned = mu_earth;
    assert_eq!(copied.value, mu_earth.value);
    assert_eq!(cloned.value, mu_earth.value);
}

#[test]
fn equality_within_planet() {
    let a = 3.986e14_f64.m3_per_s2_for::<Earth>();
    let b = 3.986e14_f64.m3_per_s2_for::<Earth>();
    assert_eq!(a, b);
}

#[test]
fn debug_includes_planet_name() {
    let mu = 1.327e20_f64.m3_per_s2_for::<Sun>();
    let dbg = format!("{mu:?}");
    assert!(dbg.contains("Sun"), "expected 'Sun' in {dbg}");
    assert!(
        dbg.contains("1327") || dbg.contains("1.327"),
        "expected the value in {dbg}"
    );
}
