//! Property-based round-trip tests for `jeod_quantities` algebraic laws.
//!
//! Three suites:
//!   1. `FrameTransform` composition associativity:
//!      `(a * b) * c ≈ a * (b * c)` when applied to a `Position`.
//!   2. Quaternion normalization idempotency for both `NormalizedQuat::renormalize`
//!      and `JeodQuat::normalize` (in-place).
//!   3. `F64Ext` unit-conversion round-trips (km→m, deg→rad→deg, km/s→m/s, etc.).
//!
//! All `f64` strategies are bounded to non-NaN, non-inf, magnitudes in
//! `1e-9..1e9` to keep the laws inside the regime where double-precision
//! arithmetic is well-behaved.

use glam::DVec3;
use jeod_quantities::prelude::*;
use proptest::prelude::*;
use uom::si::{
    angle::{degree, radian},
    length::{kilometer, meter},
    time::{hour, second},
    velocity::{kilometer_per_second, meter_per_second},
};

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Finite, non-zero `f64` in a magnitude range that keeps round-trips stable
/// (avoids subnormals and overflow).
fn finite_scalar() -> impl Strategy<Value = f64> {
    prop_oneof![
        (1.0e-9_f64..1.0e9_f64),
        (1.0e-9_f64..1.0e9_f64).prop_map(|x| -x),
    ]
}

/// Arbitrary `JeodQuat` (scalar-first, left-transformation) built from four
/// finite components. Guaranteed to have non-zero norm by component bounds.
fn arbitrary_jeod_quat() -> impl Strategy<Value = JeodQuat> {
    (
        finite_scalar(),
        finite_scalar(),
        finite_scalar(),
        finite_scalar(),
    )
        .prop_filter("non-zero, finite norm", |(a, b, c, d)| {
            let n2 = a * a + b * b + c * c + d * d;
            n2.is_finite() && n2 > 1.0e-18
        })
        .prop_map(|(a, b, c, d)| JeodQuat::from_array([a, b, c, d]))
}

/// A `NormalizedQuat<ScalarFirst, LeftTransform>` (i.e. a witnessed unit-norm
/// JEOD quaternion) drawn from the arbitrary-`JeodQuat` strategy via
/// `renormalize`.
fn arbitrary_unit_jeod_quat() -> impl Strategy<Value = NormalizedQuat<ScalarFirst, LeftTransform>> {
    arbitrary_jeod_quat().prop_filter_map("renormalize succeeds", NormalizedQuat::renormalize)
}

/// A `Position<F>` with finite, bounded components.
fn arbitrary_position<F: Frame>() -> impl Strategy<Value = jeod_quantities::aliases::Position<F>> {
    (finite_scalar(), finite_scalar(), finite_scalar())
        .prop_map(|(x, y, z)| Qty3::from_raw_si(DVec3::new(x, y, z)))
}

// Property 1: FrameTransform composition associativity. Pick four distinct
// frames A=RootInertial, B=Ecef, C=PlanetFixed<Earth>, D=PlanetFixed<Moon> (all
// sealed `Frame` impls reachable from the prelude) and assert
// `(a * b) * c ≈ a * (b * c)` after applying to a `Position<A>`.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn frame_transform_composition_is_associative(
        qa in arbitrary_unit_jeod_quat(),
        qb in arbitrary_unit_jeod_quat(),
        qc in arbitrary_unit_jeod_quat(),
        p in arbitrary_position::<RootInertial>(),
    ) {
        type A = RootInertial;
        type B = Ecef;
        type C = PlanetFixed<Earth>;
        type D = PlanetFixed<Moon>;

        let a: FrameTransform<A, B> = FrameTransform::from_quat(qa);
        let b: FrameTransform<B, C> = FrameTransform::from_quat(qb);
        let c: FrameTransform<C, D> = FrameTransform::from_quat(qc);

        let left: FrameTransform<A, D> = (a * b) * c;
        let right: FrameTransform<A, D> = a * (b * c);

        let pl = left.apply(p).raw_si();
        let pr = right.apply(p).raw_si();

        // Tolerance scales with the input magnitude — three matrix
        // multiplies on a unit rotation accumulate <1e-12 relative drift.
        let scale = p.raw_si().length().max(1.0);
        let diff = (pl - pr).length();
        prop_assert!(
            diff < 1.0e-12 * scale,
            "associativity drift = {diff} (scale={scale})\n  pl = {pl:?}\n  pr = {pr:?}",
        );
    }
}

// Property 2: Quaternion normalization idempotency. For both
// `NormalizedQuat::renormalize` (1/sqrt scaling) and `JeodQuat::normalize`
// (JEOD Padé fast path with canonical hemisphere), the second pass agrees
// with the first to within a few ULPs.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn quat_normalize_is_idempotent(q in arbitrary_jeod_quat()) {
        // --- Path 1: NormalizedQuat::renormalize ---
        let n1 = NormalizedQuat::renormalize(q)
            .expect("non-zero norm by strategy filter");
        let n2 = NormalizedQuat::renormalize(n1.inner())
            .expect("unit-norm quat re-normalizes");
        let a = n1.inner().data;
        let b = n2.inner().data;
        for i in 0..4 {
            prop_assert!(
                (a[i] - b[i]).abs() <= 1.0e-15,
                "renormalize idempotency failed at i={i}: {} vs {}",
                a[i], b[i],
            );
        }
        // The witnessed quaternion is unit-norm to within DEFAULT_TOLERANCE.
        let n = n1.inner().norm();
        prop_assert!(
            (n - 1.0).abs() <= NormalizedQuat::<ScalarFirst, LeftTransform>::DEFAULT_TOLERANCE,
            "renormalized quat not unit-norm: {n}",
        );

        // --- Path 2: JeodQuat::normalize (in-place, JEOD Padé fast path) ---
        let mut j1 = q;
        j1.normalize();
        let mut j2 = j1;
        j2.normalize();
        for i in 0..4 {
            prop_assert!(
                (j1.data[i] - j2.data[i]).abs() <= 1.0e-15,
                "JeodQuat::normalize idempotency failed at i={i}: {} vs {}",
                j1.data[i], j2.data[i],
            );
        }
        // JEOD normalization places result in canonical hemisphere (q0 >= 0).
        prop_assert!(j1.data[0] >= 0.0, "canonical hemisphere violated: q0={}", j1.data[0]);
    }
}

// Property 3: F64Ext unit-conversion round-trips. The construction methods
// on `f64` produce `uom`-typed quantities; `.get::<base>()` must invert the
// construction exactly for power-of-ten factors and within ~1e-13 relative
// for irrational factors such as π/180.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn unit_conversion_round_trips(x in finite_scalar()) {
        let scale = x.abs().max(1.0);

        // Power-of-ten conversions: x.km() → m and x.km_per_s() → m/s
        // are exact factors of 1000; tolerate <1e-9 relative for fp.
        prop_assert!(
            (x.km().get::<meter>() - x * 1000.0).abs() <= 1.0e-9 * scale,
            "km→m: got {}, expected {}", x.km().get::<meter>(), x * 1000.0,
        );
        prop_assert!(
            (x.m().get::<kilometer>() - x * 0.001).abs() <= 1.0e-12 * scale,
        );
        prop_assert!(
            (x.km_per_s().get::<meter_per_second>() - x * 1000.0).abs() <= 1.0e-9 * scale,
        );
        prop_assert!(
            (x.km_per_s().get::<kilometer_per_second>() - x).abs() <= 1.0e-12 * scale,
        );
        // Time: 1 hour == 3600 s exactly; round-trip s → hour → s.
        prop_assert!(
            (x.hours().get::<second>() - x * 3600.0).abs() <= 1.0e-9 * scale,
        );
        prop_assert!(
            (x.s().get::<hour>().hours().get::<second>() - x).abs() <= 1.0e-12 * scale,
        );

        // Irrational factor π/180: deg → rad → deg drifts within ~1e-13.
        let round = x.deg().get::<radian>().rad().get::<degree>();
        let rel = (round - x).abs() / scale;
        prop_assert!(rel <= 1.0e-13, "deg↔rad rel drift {rel} (round={round}, x={x})");
    }
}
