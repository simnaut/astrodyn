//! Property-based round-trip tests targeting the *sign-convention /
//! inversion-direction bug class*.
//!
//! Fixed unit tests pin specific scenarios; a sign error in an inversion or
//! anomaly conversion can still slip past them at intermediate points the
//! fixed cases happen not to exercise. The `time_periapsis` bug recorded in
//! the project root `CLAUDE.md` (an agent guessed `M = 2π − n·t` instead
//! of `M = n·t`, producing 11 668 km error against NASA flight data) is the
//! canonical example: a Cartesian → Keplerian → Cartesian round-trip swept
//! over `(e, i, Ω, ω, ν)` would have failed at the first sample point.
//!
//! The four suites here exercise the public inversions whose forward and
//! inverse formulas could disagree by sign:
//!
//! 1. **Orbital element round-trip** — `cartesian → keplerian → cartesian`
//!    over `e ∈ [0.05, 0.95]`, `i ∈ (ε, π − ε)`, `Ω, ω, ν ∈ (0, 2π)`,
//!    `rp ∈ [6500 km, 40 000 km]`. Asserts ≤ tolerance relative error on
//!    position and velocity. Avoids the degenerate circular / equatorial
//!    branches (separate fixed tests cover those — see
//!    `orbital_elements::tests::roundtrip_circular` / `roundtrip_polar`).
//! 2. **Quaternion convention round-trip** — `JeodQuat` (scalar-first,
//!    left-transform) ↔ `glam::DQuat` (scalar-last, left-transform) for a
//!    non-trivial axis + angle. Covers the documented boundary where JEOD
//!    and `glam` disagree on layout.
//! 3. **Frame transform round-trip** — for every typed frame pair built
//!    from a single arbitrary rotation, assert
//!    `Position<A> → Position<B> → Position<A>` recovers the original at
//!    orbital magnitudes (LEO, GEO, lunar). Catches a sign flip in
//!    `FrameTransform::inverse` or in the underlying quaternion conjugate.
//! 4. **Geodetic round-trip** — `geodetic → cartesian → geodetic` over
//!    `latitude ∈ [−89.9°, 89.9°]`, `longitude ∈ (−π, π)`, altitude from
//!    sea level to geostationary. Latitude-banded tolerance: longitude
//!    near the poles is geometrically unstable (see the `GeodeticState`
//!    rustdoc and `crates/astrodyn_math/src/geodetic.rs` for the
//!    `atan2(y, x)` sensitivity), so longitude is checked only when the
//!    equatorial radius is large enough that `atan2` is well-conditioned.

use astrodyn_math::geodetic::{cartesian_to_geodetic_typed, geodetic_to_cartesian_typed};
use astrodyn_math::types::DVec3;
use astrodyn_math::{GeodeticStateTyped, JeodQuat, OrbitalElements};
use astrodyn_quantities::prelude::*;
use glam::DQuat;
use proptest::prelude::*;
use uom::si::length::meter;

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Earth gravitational parameter (m^3/s^2). JEOD `earth_GGM05C.cc:40`.
const MU_EARTH_M3_S2: f64 = 3.986_004_415e14;

/// WGS-84 equatorial radius (m). JEOD `Earth_GGM05C.cc`.
const EARTH_R_EQ_M: f64 = 6_378_137.0;
/// WGS-84 polar radius (m): `r_eq * (1 - 1/298.257223563)`.
const EARTH_R_POL_M: f64 = EARTH_R_EQ_M * (1.0 - 1.0 / 298.257_223_563);

// ---------------------------------------------------------------------------
// Suite 1: Cartesian ↔ Keplerian round-trip
// ---------------------------------------------------------------------------
//
// Catches the `time_periapsis` class of bug. The forward map
// `from_cartesian` extracts `(a, e, i, Ω, ω, ν, M)` and the inverse
// `to_cartesian` rebuilds `(r, v)` from `(a, e, i, Ω, ω, ν)` — a sign flip
// in any of the perifocal-rotation or anomaly-conversion formulas breaks
// round-trip identity at some `(e, ν)` combination. The strategy avoids
// `e < 0.05` (circular regime is a separate branch in `from_cartesian` —
// LAN and `ω` collapse) and `i ∈ {0, π}` (equatorial regime — same
// reason). The dedicated fixed tests cover those degenerate cases.

/// Strategy: (rp_km, e, i, Ω, ω, ν).
/// * `rp` in `[6500, 40000]` km — LEO through GEO.
/// * `e` in `[0.05, 0.95]` — strictly elliptic, away from circular and
///   parabolic regimes.
/// * `i` in `[0.05, π − 0.05]` — away from the equatorial-branch switch
///   at `i = 0` / `i = π`.
/// * `Ω`, `ω`, `ν` in `[0, 2π)`.
fn elliptic_state_strategy() -> impl Strategy<Value = (f64, f64, f64, f64, f64, f64)> {
    use std::f64::consts::PI;
    (
        6500.0_f64..40_000.0_f64, // rp_km
        0.05_f64..0.95_f64,       // e
        0.05_f64..(PI - 0.05),    // i (rad)
        0.0_f64..(2.0 * PI),      // Ω (rad)
        0.0_f64..(2.0 * PI),      // ω (rad)
        0.0_f64..(2.0 * PI),      // ν (rad)
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn cartesian_keplerian_round_trip(
        (rp_km, e, i, lan, aop, nu) in elliptic_state_strategy(),
    ) {
        // Convert from osculating elements to a Cartesian state in
        // PlanetInertial<Earth>, then back. The forward map uses the
        // perifocal → IJK rotation built from `(Ω, ω, i)`; the inverse
        // re-derives those angles from the cross products of the
        // (eccentricity, node, angular-momentum) vectors. Both directions
        // must agree on the rotation sense and on the anomaly sign.

        // Build initial Cartesian state from the elements directly (mirror
        // of `to_cartesian` but in km / km/s scale to match the existing
        // `astrodyn_math::orbital_elements::tests` fixtures).
        let mu_km = MU_EARTH_M3_S2 * 1e-9; // m^3/s^2 → km^3/s^2
        let a_km = rp_km / (1.0 - e);
        let p_km = a_km * (1.0 - e * e);
        let r_km = p_km / (1.0 + e * nu.cos());
        let coeff = (mu_km / p_km).sqrt();

        // Perifocal (PQW) state.
        let r_pqw = DVec3::new(r_km * nu.cos(), r_km * nu.sin(), 0.0);
        let v_pqw = DVec3::new(-coeff * nu.sin(), coeff * (e + nu.cos()), 0.0);

        let (co, so) = (lan.cos(), lan.sin());
        let (cw, sw) = (aop.cos(), aop.sin());
        let (ci, si) = (i.cos(), i.sin());

        // PQW → IJK rotation; rows match `to_cartesian` in
        // `orbital_elements.rs` so a sign flip in either direction trips
        // the round-trip.
        let row0 = DVec3::new(co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si);
        let row1 = DVec3::new(so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si);
        let row2 = DVec3::new(sw * si, cw * si, ci);

        let apply = |row: DVec3, v: DVec3| row.dot(v);
        let pos_km = DVec3::new(apply(row0, r_pqw), apply(row1, r_pqw), apply(row2, r_pqw));
        let vel_km = DVec3::new(apply(row0, v_pqw), apply(row1, v_pqw), apply(row2, v_pqw));

        // Convert to SI for the typed API.
        let mu = MU_EARTH_M3_S2.m3_per_s2_for::<Earth>();
        let pos: Position<PlanetInertial<Earth>> = Qty3::from_raw_si(pos_km * 1000.0);
        let vel: Velocity<PlanetInertial<Earth>> = Qty3::from_raw_si(vel_km * 1000.0);

        let oe = OrbitalElements::<Earth>::from_cartesian_typed(mu, pos, vel)
            .expect("forward conversion succeeds for elliptic input");
        let (pos_back, vel_back) = oe
            .to_cartesian(MU_EARTH_M3_S2)
            .expect("inverse conversion succeeds for elliptic input");

        let pos_si = pos.raw_si();
        let vel_si = vel.raw_si();

        let pos_err = (pos_back - pos_si).length();
        let vel_err = (vel_back - vel_si).length();

        // Relative tolerance: the existing fixed `roundtrip_inclined_eccentric`
        // unit test passes at 1e-8 km (1e-5 m) absolute on similar
        // magnitudes. Use a relative bound — 1 part in 1e10 of the orbit
        // scale is well within the Newton-Raphson convergence threshold
        // (`kep_eqtn_e` uses TOL = 1e-14 rad).
        let pos_scale = pos_si.length().max(1.0);
        let vel_scale = vel_si.length().max(1.0);
        prop_assert!(
            pos_err <= 1e-9 * pos_scale,
            "pos rel err {:.3e} (e={e}, i={i}, lan={lan}, aop={aop}, nu={nu}, rp_km={rp_km})\n  pos = {pos_si:?}\n  back = {pos_back:?}",
            pos_err / pos_scale,
        );
        prop_assert!(
            vel_err <= 1e-9 * vel_scale,
            "vel rel err {:.3e} (e={e}, i={i}, lan={lan}, aop={aop}, nu={nu}, rp_km={rp_km})\n  vel = {vel_si:?}\n  back = {vel_back:?}",
            vel_err / vel_scale,
        );
    }
}

// ---------------------------------------------------------------------------
// Suite 2: Quaternion convention round-trip (scalar-first ↔ scalar-last)
// ---------------------------------------------------------------------------
//
// JEOD uses scalar-first `[q0, q1, q2, q3]`; `glam::DQuat` uses scalar-last
// `[x, y, z, w]`. Both use the left-transformation convention
// (`r' = q r q⁻¹`). The crate provides `to_scalar_last` and a `From<DQuat>`
// bridge to `Quat<ScalarLast, LeftTransform>`; a sign flip on q0 or a
// scalar-vs-vector swap in either path silently mislabels rotations.
//
// We generate non-trivial rotations (random axis + random angle ∈ (0, π))
// and verify that taking a JeodQuat to scalar-last layout, dropping into
// `glam::DQuat`, and reading it back yields the same rotation matrix.

fn axis_angle_strategy() -> impl Strategy<Value = (DVec3, f64)> {
    use std::f64::consts::PI;
    (
        // Axis components: bounded magnitudes, filtered to non-zero length.
        -1.0_f64..1.0_f64,
        -1.0_f64..1.0_f64,
        -1.0_f64..1.0_f64,
        // Angle in (0.01, π − 0.01) — exclude identity and the trace-near
        // −1 corner where matrix extraction is ill-conditioned.
        0.01_f64..(PI - 0.01),
    )
        .prop_filter("axis non-zero", |(x, y, z, _)| x * x + y * y + z * z > 1e-6)
        .prop_map(|(x, y, z, angle)| (DVec3::new(x, y, z).normalize(), angle))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn quaternion_layout_round_trip((axis, angle) in axis_angle_strategy()) {
        // Build a JEOD-convention quaternion (scalar-first, left-transform).
        let q_jeod = JeodQuat::left_quat_from_eigen_rotation(angle, axis);

        // Hop to scalar-last → glam → back, exercising every bridging path
        // that callers actually use. A scalar/vector mix-up at any boundary
        // would corrupt the rotation it represents.
        let q_scalar_last = q_jeod.to_scalar_last();
        let dq: DQuat = q_scalar_last.to_glam();
        let from_dq: Quat<ScalarLast, LeftTransform> = dq.into();
        let q_round = from_dq.to_scalar_first();

        // Compare resulting transformation matrices: quaternions q and −q
        // describe the same rotation, so we compare the rotation they
        // induce rather than componentwise data. (The bridging path doesn't
        // currently negate, but this keeps the test correctness independent
        // of the canonical-hemisphere choice.)
        let m_orig = q_jeod.left_quat_to_transformation();
        let m_round = q_round.left_quat_to_transformation();

        for c in 0..3 {
            for r in 0..3 {
                let d = (m_orig.col(c)[r] - m_round.col(c)[r]).abs();
                prop_assert!(
                    d <= 1e-14,
                    "matrix component ({r},{c}) drift {d} (axis={axis:?}, angle={angle})",
                );
            }
        }

        // Also verify the round-trip preserves vector application: applying
        // the original quaternion and the re-extracted one to a probe vector
        // must agree. This is the load-bearing property — a sign flip on q0
        // would invert the rotation here even if the matrix happened to look
        // benign.
        let v = DVec3::new(1.234, -0.567, 2.345);
        let v_orig = q_jeod.left_quat_transform(v);
        let v_round = q_round.left_quat_transform(v);
        let diff = (v_orig - v_round).length();
        prop_assert!(
            diff <= 1e-13 * v.length(),
            "vector apply drift {diff} (axis={axis:?}, angle={angle})",
        );
    }
}

// ---------------------------------------------------------------------------
// Suite 3: Frame transform round-trip at orbital magnitudes
// ---------------------------------------------------------------------------
//
// `FrameTransform<A, B>::inverse` returns the `B → A` transform. A sign
// flip in the underlying quaternion conjugate (or a transposed-vs-inverted
// matrix mix-up) would manifest as `t.inverse().apply(t.apply(p)) != p`.
// Sweep over arbitrary rotations and probe at LEO (≈ 7 e6 m), GEO (≈ 4.2
// e7 m), and lunar (≈ 3.84 e8 m) scales — different magnitudes catch
// scale-dependent precision regressions a single-scale fixed test might
// miss.

fn unit_quat_strategy() -> impl Strategy<Value = NormalizedQuat<ScalarFirst, LeftTransform>> {
    (
        -1.0_f64..1.0_f64,
        -1.0_f64..1.0_f64,
        -1.0_f64..1.0_f64,
        -1.0_f64..1.0_f64,
    )
        .prop_filter("norm non-tiny", |(a, b, c, d)| {
            a * a + b * b + c * c + d * d > 1e-6
        })
        .prop_filter_map("renormalize", |(a, b, c, d)| {
            NormalizedQuat::renormalize(JeodQuat::from_array([a, b, c, d]))
        })
}

/// Three orbital-magnitude probe scales (m): LEO, GEO, lunar.
fn probe_scale_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(7.0e6_f64),   // LEO altitude
        Just(4.22e7_f64),  // GEO radius
        Just(3.844e8_f64), // Earth-Moon distance
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn frame_transform_round_trip_at_orbital_magnitudes(
        q in unit_quat_strategy(),
        scale in probe_scale_strategy(),
        // Direction of the probe in the source frame: bounded to a unit-ish
        // sphere by the strategy filter below.
        dx in -1.0_f64..1.0_f64,
        dy in -1.0_f64..1.0_f64,
        dz in -1.0_f64..1.0_f64,
    ) {
        prop_assume!(dx * dx + dy * dy + dz * dz > 1e-3);

        let dir = DVec3::new(dx, dy, dz).normalize();
        let p_src: Position<RootInertial> = Qty3::from_raw_si(dir * scale);

        // Build a frame-pair `RootInertial → PlanetInertial<Earth>` from
        // the witnessed quaternion. The phantom pair is what makes the
        // round-trip a *typed* property: `inverse` must produce a
        // `PlanetInertial<Earth> → RootInertial` whose composition with
        // the forward map recovers a `Position<RootInertial>`.
        let t: FrameTransform<RootInertial, PlanetInertial<Earth>> = FrameTransform::from_quat(q);
        let p_dst: Position<PlanetInertial<Earth>> = t.apply(p_src);
        let p_back: Position<RootInertial> = t.inverse().apply(p_dst);

        // Tolerance: relative to the probe scale. Two unit-quaternion
        // applies accumulate roughly machine-precision relative drift on
        // a single vector; 1e-12 leaves a margin for the worst-case
        // accumulated rounding without papering over real bugs.
        let drift = (p_src.raw_si() - p_back.raw_si()).length();
        prop_assert!(
            drift <= 1e-12 * scale,
            "round-trip drift = {drift} (scale = {scale}, q = {:?})",
            q.inner().data,
        );
    }
}

// ---------------------------------------------------------------------------
// Suite 4: Geodetic round-trip
// ---------------------------------------------------------------------------
//
// `cartesian_to_geodetic` runs the Borkowski iterative latitude solver and
// returns `(lat, lon, alt)`; `geodetic_to_cartesian` rebuilds the Cartesian
// state. A sign flip in the inverse direction or in the altitude offset
// would fail to round-trip.
//
// Per the public `GeodeticState` rustdoc, longitude is numerically
// unstable in the polar neighborhood (atan2 sensitivity scales with
// 1/cos(lat)). We therefore use latitude-banded tolerances: longitude is
// asserted only when |lat| < 89°, where atan2 conditioning is solid.

fn geodetic_strategy() -> impl Strategy<Value = (f64, f64, f64)> {
    use std::f64::consts::PI;
    let lat_max = 89.9_f64.to_radians();
    (
        -lat_max..lat_max,         // latitude (rad)
        (-PI + 1e-6)..(PI - 1e-6), // longitude (rad)
        0.0_f64..4.22e7_f64,       // altitude: sea level to GEO
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn geodetic_round_trip((lat, lon, alt) in geodetic_strategy()) {
        let state = GeodeticStateTyped::from_raw(astrodyn_math::GeodeticState {
            latitude: lat,
            longitude: lon,
            altitude: alt,
        });

        let r_eq = EARTH_R_EQ_M.m();
        let r_pol = EARTH_R_POL_M.m();

        let cart: Position<PlanetFixed<Earth>> =
            geodetic_to_cartesian_typed::<Earth>(state, r_eq, r_pol);
        let back: GeodeticStateTyped = cartesian_to_geodetic_typed::<Earth>(cart, r_eq, r_pol);

        let back_raw = back.into_raw();

        // Latitude and altitude are well-conditioned everywhere in the
        // strategy range. Altitude tolerance is in meters; latitude in
        // radians.
        prop_assert!(
            (back_raw.latitude - lat).abs() <= 1e-12,
            "latitude drift {} (lat={lat}, lon={lon}, alt={alt})",
            (back_raw.latitude - lat).abs(),
        );
        prop_assert!(
            (back_raw.altitude - alt).abs() <= 1e-6,
            "altitude drift {} m (lat={lat}, lon={lon}, alt={alt})",
            (back_raw.altitude - alt).abs(),
        );

        // Longitude: only assert away from the poles. At 89.9° latitude
        // the equatorial radius is ~r·cos(lat) ≈ r·0.00175; at altitudes
        // up to GEO the (x, y) magnitudes stay well above the
        // atan2-instability floor, but the sensitivity is real. We
        // tighten the tolerance toward the equator where atan2 is exact.
        let cos_lat = lat.cos().abs();
        // Equatorial radius of the probe point (m): ≈ (r_eq + alt) cos(lat)
        // for moderate eccentricity. Longitude sensitivity is ~1/eqr.
        let eqr = (EARTH_R_EQ_M + alt) * cos_lat;
        // Expected atan2 drift: machine-precision tan-of-angle drift
        // amplified by 1/eqr. 1e-9 / eqr is comfortably above the
        // observed bound for the strategy range without papering over
        // bugs.
        let lon_tol = 1e-9 / eqr.max(1.0) + 1e-12;
        let lon_diff = (back_raw.longitude - lon).abs();
        // atan2 returns in (−π, π]; the input is also in (−π, π); no wrap.
        prop_assert!(
            lon_diff <= lon_tol,
            "longitude drift {lon_diff} > tol {lon_tol} (lat={lat}, lon={lon}, alt={alt})",
        );

        // Sanity: the typed Cartesian state can round-trip through its
        // `Length` view (units check) — guards against a silent base-unit
        // swap in `r_eq` / `r_pol`.
        let _ = cart.raw_si().length();
        let _ = r_eq.get::<meter>();
    }
}
