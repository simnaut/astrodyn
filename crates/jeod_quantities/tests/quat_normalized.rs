//! `NormalizedQuat` invariant preservation.

use jeod_quantities::prelude::*;

#[test]
fn accepts_exact_identity() {
    let q = JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]);
    assert!(NormalizedQuat::new(q).is_ok());
}

#[test]
fn rejects_zero_quaternion() {
    let q = JeodQuat::from_array([0.0, 0.0, 0.0, 0.0]);
    let err = NormalizedQuat::new(q).expect_err("zero must be rejected");
    assert!(err.deviation > err.tolerance);
}

#[test]
fn rejects_scale_2_quaternion() {
    let q = JeodQuat::from_array([2.0, 0.0, 0.0, 0.0]);
    let err = NormalizedQuat::new(q).expect_err("scale-2 must be rejected");
    assert!(err.deviation > err.tolerance);
}

#[test]
fn accepts_non_trivial_unit_quaternion() {
    // Quaternion for a 90° rotation about Z: q = [cos(45°), 0, 0, sin(45°)]
    let c = (core::f64::consts::FRAC_PI_4).cos();
    let s = (core::f64::consts::FRAC_PI_4).sin();
    let q = JeodQuat::from_array([c, 0.0, 0.0, s]);
    assert!(NormalizedQuat::new(q).is_ok(), "norm = {}", q.norm());
}

#[test]
fn renormalize_produces_unit_output() {
    let q = JeodQuat::from_array([3.0, 0.0, 4.0, 0.0]); // norm 5
    let n = NormalizedQuat::renormalize(q).expect("non-zero input renormalizes");
    assert!((n.inner().norm() - 1.0).abs() < 1e-12);
}

#[test]
fn renormalize_rejects_zero() {
    let q = JeodQuat::from_array([0.0, 0.0, 0.0, 0.0]);
    assert!(NormalizedQuat::renormalize(q).is_none());
}

#[test]
fn custom_tolerance_accepts_near_unit() {
    // Norm slightly off: simulate tiny floating-point drift.
    let q = JeodQuat::from_array([1.0 + 1e-9, 0.0, 0.0, 0.0]);
    assert!(NormalizedQuat::new_with_tolerance(q, 1e-6).is_ok());
    assert!(NormalizedQuat::new_with_tolerance(q, 1e-12).is_err());
}

#[test]
fn layout_conversion_round_trip() {
    let original = JeodQuat::from_array([0.5, 0.5, 0.5, 0.5]);
    let as_last = original.to_scalar_last();
    let back = as_last.to_scalar_first();
    assert_eq!(original.data, back.data);
}
