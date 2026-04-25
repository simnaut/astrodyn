//! Integration tests for `HarmonicDegree` — exercises the public surface
//! (newtype construction, ordering, default-as-zero, raw integer round
//! trip) plus the accessor.

use jeod_quantities::prelude::*;

#[test]
fn prelude_exposes_harmonic_degree() {
    let _d: HarmonicDegree = HarmonicDegree::new(8);
}

#[test]
fn ordering_works_via_prelude() {
    let degrees = [
        HarmonicDegree(8),
        HarmonicDegree(2),
        HarmonicDegree(70),
        HarmonicDegree(0),
    ];
    let mut sorted = degrees.to_vec();
    sorted.sort();
    assert_eq!(
        sorted,
        vec![
            HarmonicDegree(0),
            HarmonicDegree(2),
            HarmonicDegree(8),
            HarmonicDegree(70),
        ]
    );
}

#[test]
fn default_is_zero() {
    assert_eq!(HarmonicDegree::default(), HarmonicDegree::new(0));
    assert_eq!(HarmonicDegree::default().get(), 0);
}

#[test]
fn round_trip_through_usize_field() {
    // Mirrors how `jeod_gravity::gravity_controls::GravityControl`
    // currently stores degree/order as `usize`. A typed
    // `GravityControlTyped` will use `HarmonicDegree` and convert at
    // the boundary via these From impls.
    let raw: usize = 8;
    let typed: HarmonicDegree = raw.into();
    let raw_back: usize = typed.into();
    assert_eq!(raw, raw_back);
    assert_eq!(typed.get(), 8);
}
