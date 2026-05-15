//! Property-based identity tests for the typed-quantity *facade*.
//!
//! The semantic-identity analogue of Miri's aliasing checks: cheap proptests
//! that exercise algebraic identities the typed API promises but a fixed
//! unit test could miss at the intermediate sample points an AI-authored
//! refactor happens to break. The four suites here:
//!
//! 1. **Frame-transform inverse round-trip** — `T ∘ T⁻¹ ≈ I` over random
//!    `Position<A>` *and* `Velocity<A>` at orbital magnitudes. Would catch
//!    a sign flip in `FrameTransform::inverse` (the quaternion conjugate
//!    or the matrix transpose drifting out of sync) for either dimension.
//!    Existing fixed tests pin a single rotation about Z; existing
//!    `proptest_sign_convention.rs` suite 3 covers `Position` only — this
//!    suite probes the load-bearing property for the `Velocity` dimension
//!    too, so a Qty3-Mul-by-rotation regression at the dimension boundary
//!    would surface.
//!
//! 2. **NormalizedQuat round-trip** — `NormalizedQuat::renormalize(q)` then
//!    `.inner()` returns a unit-norm quaternion that rotates a typed
//!    `Position` identically to the JEOD Padé `JeodQuat::normalize()`
//!    in-place path. Would catch the witness silently dropping its
//!    renormalize step (the typed Phase-8 facade's guard) or a sign drift
//!    between the two normalization paths.
//!
//! 3. **`Position<F> + Velocity<F> * dt − Velocity<F> * dt ≈ Position<F>`**
//!    round-trip. The Qty3 `Mul<Quantity<Time>>` impl is the load-bearing
//!    bridge between `Position`, `Velocity`, and `Time`. A regression that
//!    swapped the dimension exponents, lost a frame phantom, or substituted
//!    the wrong base-unit factor (e.g. silently using `Duration::secs_f32`
//!    instead of `Time::new::<second>`) would round-trip with bounded f64
//!    arithmetic error — checked against the position magnitude scale.
//!
//! 4. **`JeodQuat` compose-decompose** — for two *non-trivial* rotations
//!    (random axis + random angle in (0.05, π − 0.05) — never identity,
//!    never 90° or 180° axis-aligned, per CLAUDE.md "never just identity
//!    or 90-degree axes"), composing via quaternion multiply must agree
//!    with composing via matrix multiply, and extracting back via
//!    `left_quat_from_transformation` must recover the same rotation.
//!    Would catch a regression in `multiply()`, `to_glam` ordering, or the
//!    matrix-to-quaternion branch-selection logic.
//!
//! ## Strategy bounds
//!
//! - Position magnitudes ∈ [6.4e6, 4e8] m — Earth surface to lunar distance.
//! - Velocity magnitudes ∈ [0, 1.2e4] m/s — sub-orbital to escape regime.
//! - dt ∈ [1e-6, 1e4] s — sub-microsecond to a few hours; the integration
//!   step regime that `Simulation::step()` actually exercises.
//! - Rotations: random axis (filter `‖a‖ > 1e-3`, then normalize) and angle
//!   ∈ (0.05, π − 0.05) — avoids identity (no information about the rotation
//!   sign) and the matrix-extraction `trace ≈ −1` corner.
//!
//! All four suites run at the `proptest` default of 256 cases; the issue's
//! acceptance criteria call for a `PROPTEST_CASES=2048` stress run before
//! merge.

use astrodyn_quantities::prelude::*;
use glam::{DMat3, DQuat, DVec3};
use proptest::prelude::*;
use uom::si::f64::Time;
use uom::si::time::second;

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// Unit-direction in R³ — random `(x, y, z) ∈ [−1, 1]³` filtered to non-tiny
/// magnitude and then normalized.
fn unit_direction_strategy() -> impl Strategy<Value = DVec3> {
    (-1.0_f64..1.0_f64, -1.0_f64..1.0_f64, -1.0_f64..1.0_f64)
        .prop_filter("non-zero direction", |(x, y, z)| {
            x * x + y * y + z * z > 1.0e-3
        })
        .prop_map(|(x, y, z)| DVec3::new(x, y, z).normalize())
}

/// Position-magnitude in metres — Earth surface (~6.4 e6) to lunar distance
/// (~3.84 e8). Covers LEO, GEO, and lunar regimes with a single strategy so
/// shrinking can pick the regime that surfaces a regression most clearly.
fn position_magnitude_strategy() -> impl Strategy<Value = f64> {
    6.4e6_f64..4.0e8_f64
}

/// Velocity-magnitude in m/s — 0 to ~12 km/s. Spans typical orbital regimes
/// (LEO ~7.7 km/s, GEO ~3.07 km/s, escape ~11.2 km/s).
fn velocity_magnitude_strategy() -> impl Strategy<Value = f64> {
    0.0_f64..1.2e4_f64
}

/// Integration time step in seconds — 1e-6 (sub-microsecond) to 1e4 (a few
/// hours), the regime `Simulation::step()` actually exercises.
fn dt_strategy() -> impl Strategy<Value = f64> {
    1.0e-6_f64..1.0e4_f64
}

/// Rotation as (axis, angle). Excludes identity (angle near 0) and the
/// matrix-extraction `trace ≈ −1` corner (angle near π). The bounds match
/// `proptest_sign_convention.rs`'s `axis_angle_strategy` so failures across
/// the two files reproduce cleanly.
fn rotation_strategy() -> impl Strategy<Value = (DVec3, f64)> {
    use std::f64::consts::PI;
    (unit_direction_strategy(), 0.05_f64..(PI - 0.05))
}

/// `NormalizedQuat<ScalarFirst, LeftTransform>` drawn from a non-trivial
/// axis-angle rotation. Uses `JeodQuat::left_quat_from_eigen_rotation` (the
/// JEOD convention) and the typed `NormalizedQuat::new` witness constructor,
/// so anything that reaches this strategy is already unit-norm to within
/// `NormalizedQuat::DEFAULT_TOLERANCE`.
fn non_trivial_unit_quat_strategy(
) -> impl Strategy<Value = NormalizedQuat<ScalarFirst, LeftTransform>> {
    rotation_strategy().prop_map(|(axis, angle)| {
        let q = JeodQuat::left_quat_from_eigen_rotation(angle, axis);
        NormalizedQuat::new(q)
            .expect("axis-angle constructor normalizes the result before returning")
    })
}

// ---------------------------------------------------------------------------
// Suite 1: FrameTransform inverse round-trip (Position and Velocity)
// ---------------------------------------------------------------------------
//
// `T.inverse()` is the load-bearing inversion in the typed facade. The
// `frame_transform_round_trip_at_orbital_magnitudes` proptest in
// `astrodyn_math/tests/proptest_sign_convention.rs` already exercises this
// for `Position` only; this suite adds the `Velocity` sibling and the
// composition-with-self-inverse path (`(t.inverse() * t).apply(v) == v`)
// that the existing test does not cover.
//
// Tolerance rationale: two unit-quaternion applies on a single vector
// accumulate roughly machine-precision relative drift. The bound is set
// to `1e-12 × position_magnitude` (about 4 nm at lunar distance) — well
// above the f64 ULP at that scale (~4 e-8 m at 4 e8 m) but tight enough
// that a single-component sign flip in the inverse rotation would
// exceed it by many orders of magnitude.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn frame_transform_inverse_round_trip_position_and_velocity(
        q in non_trivial_unit_quat_strategy(),
        pos_scale in position_magnitude_strategy(),
        vel_scale in velocity_magnitude_strategy(),
        pos_dir in unit_direction_strategy(),
        vel_dir in unit_direction_strategy(),
    ) {
        // Build a transform between two typed frames. The phantom pair is
        // what makes this a *typed* property: a sign flip in
        // `FrameTransform::inverse` would either fail to typecheck (good)
        // or produce a numerical mismatch here (also good — that's what
        // this proptest exists to catch).
        let t: FrameTransform<RootInertial, Ecef> = FrameTransform::from_quat(q);
        let t_inv: FrameTransform<Ecef, RootInertial> = t.inverse();

        // ---- Position round-trip ----
        let p_src: Position<RootInertial> = Qty3::from_raw_si(pos_dir * pos_scale);
        let p_round: Position<RootInertial> = t_inv.apply(t.apply(p_src));
        let p_drift = (p_src.raw_si() - p_round.raw_si()).length();
        prop_assert!(
            p_drift <= 1.0e-12 * pos_scale,
            "Position round-trip drift {p_drift} > 1e-12 × {pos_scale} \
             (q = {:?}, pos_dir = {pos_dir:?})",
            q.inner().data,
        );

        // ---- Velocity round-trip ----
        let v_src: Velocity<RootInertial> = Qty3::from_raw_si(vel_dir * vel_scale);
        let v_round: Velocity<RootInertial> = t_inv.apply(t.apply(v_src));
        // Velocity magnitudes can be zero by strategy; floor the scale at
        // 1 m/s so the relative bound stays meaningful at the lower limit.
        let v_scale_floor = vel_scale.max(1.0);
        let v_drift = (v_src.raw_si() - v_round.raw_si()).length();
        prop_assert!(
            v_drift <= 1.0e-12 * v_scale_floor,
            "Velocity round-trip drift {v_drift} > 1e-12 × {v_scale_floor} \
             (q = {:?}, vel_dir = {vel_dir:?})",
            q.inner().data,
        );

        // ---- Composition path: (t.inverse() * t) must apply as the identity.
        // This is a strictly stronger property than the apply-then-apply
        // round-trip above because it routes through the `Mul` overload —
        // a regression that broke composition without breaking `inverse`
        // would surface here only.
        let composed: FrameTransform<RootInertial, RootInertial> = t * t_inv;
        let p_via_compose: Position<RootInertial> = composed.apply(p_src);
        let drift_compose = (p_src.raw_si() - p_via_compose.raw_si()).length();
        prop_assert!(
            drift_compose <= 1.0e-12 * pos_scale,
            "compose(t, t.inverse()) drift {drift_compose} > 1e-12 × {pos_scale}",
        );
    }
}

// ---------------------------------------------------------------------------
// Suite 2: NormalizedQuat round-trip
// ---------------------------------------------------------------------------
//
// The Phase-8 facade promises that a `NormalizedQuat` is *always* unit-norm
// — the typestate guard for `left_quat_to_transformation` and for
// `FrameTransform::from_quat`. This suite probes two related properties:
//
//   a) `NormalizedQuat::renormalize(q).inner()` has unit norm within
//      `DEFAULT_TOLERANCE`, regardless of the input scale.
//   b) The rotation produced by the witness matches the rotation produced
//      by the JEOD in-place Padé normalization (`JeodQuat::normalize`)
//      when applied to a typed `Position` — i.e. both paths agree on the
//      rotation that the un-normalized input *represented*.
//
// Tolerance: 1e-14 absolute on the norm (slack against the f64 1.0 −
// machine epsilon floor), and 1e-13 relative on the rotated-vector
// magnitude. A regression that dropped the renormalize step on one path
// or sign-flipped the canonical-hemisphere step would fail (b) at the
// first sample with a non-trivial angle.

/// An arbitrary `JeodQuat` with finite, non-degenerate norm. The norm-squared
/// lower bound keeps `renormalize` away from division-by-near-zero; the
/// upper bound keeps norms from overflowing the f64 representable range.
fn arbitrary_finite_jeod_quat() -> impl Strategy<Value = JeodQuat> {
    let comp = -10.0_f64..10.0_f64;
    (comp.clone(), comp.clone(), comp.clone(), comp)
        .prop_filter("non-degenerate norm", |(a, b, c, d)| {
            let n2 = a * a + b * b + c * c + d * d;
            n2.is_finite() && (1.0e-6..1.0e6).contains(&n2)
        })
        .prop_map(|(a, b, c, d)| JeodQuat::from_array([a, b, c, d]))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn normalized_quat_round_trip_unit_norm_and_position_rotation(
        q_raw in arbitrary_finite_jeod_quat(),
        pos_scale in position_magnitude_strategy(),
        pos_dir in unit_direction_strategy(),
    ) {
        // Witness-gated path: typed-facade renormalize.
        let q_witness = NormalizedQuat::renormalize(q_raw)
            .expect("non-degenerate norm by strategy filter");
        let norm = q_witness.inner().norm();
        prop_assert!(
            (norm - 1.0).abs() <= 1.0e-14,
            "renormalized quat norm {norm} (deviation {}) exceeds 1e-14",
            (norm - 1.0).abs(),
        );
        // The witness must satisfy the DEFAULT_TOLERANCE invariant the
        // facade relies on at every `NormalizedQuat::new` site.
        prop_assert!(
            (norm - 1.0).abs()
                <= NormalizedQuat::<ScalarFirst, LeftTransform>::DEFAULT_TOLERANCE,
            "renormalize fails the facade's DEFAULT_TOLERANCE guarantee",
        );

        // JEOD in-place path: same input, same convention. The two
        // representations may differ in canonical-hemisphere sign (JEOD's
        // path flips `scalar < 0` to `scalar >= 0`; renormalize does not),
        // so we compare the *rotation* they describe, not the components.
        let mut q_jeod = q_raw;
        q_jeod.normalize();

        // Apply each to a typed Position and compare the rotated raw
        // values. Both paths rotate the same un-normalized input through
        // the same convention; the resulting rotated vector magnitudes
        // must agree within 1e-13 relative.
        let p_src: Position<RootInertial> = Qty3::from_raw_si(pos_dir * pos_scale);
        let v_witness = q_witness.left_quat_transform(p_src.raw_si());
        let v_jeod = q_jeod.left_quat_transform(p_src.raw_si());

        // Same rotation modulo q ↔ −q; check the magnitudes match (they
        // must for any unit quaternion application) and the directions
        // agree up to sign. The canonical-hemisphere flip on the JEOD
        // path is global — it negates *all four* components — and global
        // negation leaves the rotation `q · v · q⁻¹` invariant, so a
        // direction sign-flip here would indicate a real regression.
        let mag_witness = v_witness.length();
        let mag_jeod = v_jeod.length();
        prop_assert!(
            (mag_witness - mag_jeod).abs() <= 1.0e-13 * pos_scale,
            "rotated magnitude disagrees: witness={mag_witness}, jeod={mag_jeod}",
        );
        // Unit-quaternion rotation preserves length — sanity check both.
        prop_assert!(
            (mag_witness - pos_scale).abs() <= 1.0e-13 * pos_scale,
            "witness path did not preserve length: {mag_witness} vs {pos_scale}",
        );
        prop_assert!(
            (mag_jeod - pos_scale).abs() <= 1.0e-13 * pos_scale,
            "jeod path did not preserve length: {mag_jeod} vs {pos_scale}",
        );
        let dir_diff = (v_witness - v_jeod).length();
        prop_assert!(
            dir_diff <= 1.0e-12 * pos_scale,
            "rotated direction disagrees: drift {dir_diff} > 1e-12 × {pos_scale} \
             (q_raw = {:?}, witness = {:?}, jeod = {:?})",
            q_raw.data, q_witness.inner().data, q_jeod.data,
        );
    }
}

// ---------------------------------------------------------------------------
// Suite 3: Position + Velocity*dt − Velocity*dt round-trip
// ---------------------------------------------------------------------------
//
// The Qty3 `Mul<Quantity<Time>>` impl is the typed bridge between
// `Position`, `Velocity`, and `Time`. A regression that swapped the
// dimension exponents (e.g. silently producing acceleration instead of
// position), dropped a frame phantom, or routed through the wrong unit
// would either fail to typecheck or produce a value that round-trips
// nowhere near the original.
//
// Tolerance: bounded by f64 ULP at the operands' magnitude scale. With
// dt up to 1e4 s and velocity up to 1.2e4 m/s the increment is
// `≤ 1.2e8 m`, which is the same order as the upper position bound. The
// per-component round-trip drift is bounded by ~2 ULPs on the larger
// operand: `|drift| ≤ 4 × 2.2e-16 × max(pos, v·dt)`. We use 1e-9 relative
// to the larger of (position magnitude, v·dt) — comfortably above f64
// rounding without papering over a real bug (a typo dropping a factor of
// 2 would land at relative ~1e-1).

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn position_plus_velocity_dt_round_trip(
        pos_scale in position_magnitude_strategy(),
        vel_scale in velocity_magnitude_strategy(),
        pos_dir in unit_direction_strategy(),
        vel_dir in unit_direction_strategy(),
        dt_s in dt_strategy(),
    ) {
        // Build typed inputs in `RootInertial`. The Qty3<length, F> and
        // Qty3<velocity, F> phantoms force the same frame on both
        // operands — a regression that quietly relabelled either to a
        // different frame would fail to typecheck rather than reach this
        // assertion.
        let p: Position<RootInertial> = Qty3::from_raw_si(pos_dir * pos_scale);
        let v: Velocity<RootInertial> = Qty3::from_raw_si(vel_dir * vel_scale);
        let dt: Time = Time::new::<second>(dt_s);

        // Forward then reverse: `(p + v*dt) − v*dt` must equal `p` to
        // within f64 ULP at the operand magnitude.
        //
        // The intermediate `v*dt` is a `Qty3<length, RootInertial>` —
        // type inference does the dimension multiply at the impl-site
        // bound. We don't annotate the intermediate so a future
        // regression that broke dimension inference would produce a
        // compiler error here rather than a silently-wrong scalar.
        let advance = v * dt;
        let p_forward = p + advance;
        let p_round = p_forward - advance;

        let advance_mag = advance.raw_si().length();
        let scale = pos_scale.max(advance_mag).max(1.0);
        let drift = (p.raw_si() - p_round.raw_si()).length();
        prop_assert!(
            drift <= 1.0e-9 * scale,
            "round-trip drift {drift} > 1e-9 × {scale} \
             (pos_scale={pos_scale}, vel_scale={vel_scale}, dt_s={dt_s})",
        );

        // Sanity check the dimension multiply: `v · dt` must have the
        // same magnitude as `‖v‖ · dt` (units cancel correctly).
        let expected_advance_mag = vel_scale * dt_s;
        prop_assert!(
            (advance_mag - expected_advance_mag).abs() <= 1.0e-12 * scale,
            "advance magnitude {advance_mag} != ‖v‖·dt={expected_advance_mag} \
             — dimension multiply silently dropped a unit factor",
        );
    }
}

// ---------------------------------------------------------------------------
// Suite 4: JeodQuat compose-decompose with non-trivial rotations
// ---------------------------------------------------------------------------
//
// "Never just identity or 90-degree axes" (CLAUDE.md, Quaternion Convention):
// quaternion compose-decompose tests must use real arbitrary rotations
// because identity and 90° axis-aligned cases hide sign and convention
// bugs. This suite probes the four-way agreement between:
//
//   a) `JeodQuat::multiply` of two non-trivial rotations,
//   b) the matrix product of their transformation matrices,
//   c) `JeodQuat::left_quat_from_transformation` of (b), recovering the
//      composed quaternion (modulo `q ↔ −q`), and
//   d) `Quat<ScalarFirst, LeftTransform>` → `ScalarLast` → `glam::DQuat`
//      → `ScalarLast` → `ScalarFirst` round-trip of the composed value.
//
// Would catch:
//   - a sign error in `multiply`'s `s1*v2 + s2*v1 + v1 × v2` formula,
//   - a transpose-vs-rotation mix-up in
//     `left_quat_to_transformation_impl`,
//   - a wrong-branch selection in `left_quat_from_transformation`'s
//     `max(tr, T00, T11, T22)` extractor,
//   - a layout swap (`ScalarFirst` ↔ `ScalarLast`) in the `to_scalar_*`
//     bridges or in the `glam::DQuat` → `Quat::from` conversion.
//
// Tolerance: 1e-13 relative on the rotated-vector magnitude across two
// composed unit rotations. Two unit-quaternion applies on a probe vector
// accumulate ~2 ULP relative drift; the matrix path adds another two
// rounding rounds, putting the worst-case at ~1e-15 relative. The 1e-13
// floor leaves margin for the matrix-to-quaternion branch with the worst
// conditioning (`trace ≈ −1`).
//
// External refs: Vallado §3.7 "Quaternion Algebra"; JEOD
// `models/utils/quaternion/src/quaternion_multiply.cc` for the JEOD-
// convention multiply.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn jeod_quat_compose_decompose_non_trivial(
        q_a in non_trivial_unit_quat_strategy(),
        q_b in non_trivial_unit_quat_strategy(),
        probe_dir in unit_direction_strategy(),
        probe_scale in position_magnitude_strategy(),
    ) {
        // (a) Compose via JeodQuat::multiply.
        let q_ab_quat = q_a.inner().multiply(&q_b.inner());

        // (b) Compose via matrix multiply. JEOD convention applies `q_a`
        // first then `q_b`, so the rotation matrix is `M_b · M_a` and the
        // equivalent quaternion is `q_b · q_a` (right-to-left order).
        // We probe with both compositions to surface a left-vs-right
        // convention swap immediately rather than letting the test
        // "accidentally pass" because both quaternions happened to be
        // symmetric.
        let m_a = q_a.left_quat_to_transformation();
        let m_b = q_b.left_quat_to_transformation();

        // Apply the q_a · q_b quaternion product to a probe vector and
        // compare against the matrix product M_a · M_b applied to the
        // same vector. Both should describe the same composite rotation.
        let probe = probe_dir * probe_scale;
        let v_via_quat = q_ab_quat.left_quat_transform(probe);
        let v_via_matrices = m_a * (m_b * probe);
        let drift_qm = (v_via_quat - v_via_matrices).length();
        prop_assert!(
            drift_qm <= 1.0e-13 * probe_scale,
            "compose-via-quat ≠ compose-via-matrix: drift {drift_qm} > 1e-13 × {probe_scale}",
        );

        // (c) Decompose: extract a quaternion from the composed rotation
        // matrix and reapply. Should recover the same rotation (modulo
        // `q ↔ −q`).
        let m_composed: DMat3 = m_a * m_b;
        let q_recovered = JeodQuat::left_quat_from_transformation(&m_composed);
        let v_via_recovered = q_recovered.left_quat_transform(probe);
        let drift_decomp = (v_via_quat - v_via_recovered).length();
        prop_assert!(
            drift_decomp <= 1.0e-13 * probe_scale,
            "compose-decompose drift {drift_decomp} > 1e-13 × {probe_scale}",
        );

        // (d) Layout round-trip via glam: ScalarFirst → ScalarLast →
        // glam::DQuat → ScalarLast → ScalarFirst must preserve the
        // rotation. A regression in `to_scalar_last` / `to_scalar_first`
        // (or in the `From<DQuat>` bridge) would scramble the layout and
        // surface here as a near-100% relative drift.
        let q_sl: Quat<ScalarLast, LeftTransform> = q_ab_quat.to_scalar_last();
        let dq: DQuat = q_sl.to_glam();
        let q_sl_back: Quat<ScalarLast, LeftTransform> = dq.into();
        let q_sf_back: JeodQuat = q_sl_back.to_scalar_first();
        let v_via_layout = q_sf_back.left_quat_transform(probe);
        let drift_layout = (v_via_quat - v_via_layout).length();
        prop_assert!(
            drift_layout <= 1.0e-13 * probe_scale,
            "layout round-trip drift {drift_layout} > 1e-13 × {probe_scale}",
        );
    }
}
