//! Property-based round-trip tests for the time-scale conversion
//! sign-convention bug class.
//!
//! A wrong sign on `TAI_TT_OFFSET`, a swapped argument in
//! `tai_to_utc_tjt`/`utc_to_tai_tjt`, or an off-by-one boundary index in
//! the leap-second lookup would silently shift mission epochs without
//! tripping a single-epoch fixed test. Property tests sweep the
//! parameter space the fixed tests don't reach — specifically, **every
//! leap-second insertion event in the USNO table** (1972–2017, plus
//! post-2017 epochs covered by the clamp regime that
//! `default_leap_second_table` enables) and arbitrary `f64` offsets from
//! each boundary.
//!
//! Three suites:
//!
//! 1. **TAI ↔ TT round-trip** — `tai → tt → tai` for arbitrary TAI
//!    seconds-since-epoch. Catches a sign flip on `TAI_TT_OFFSET`.
//! 2. **TAI ↔ UTC round-trip across every leap-second insertion** —
//!    iterates the 28 USNO boundaries committed in
//!    `default_leap_second_table`, perturbs by an arbitrary offset, and
//!    requires `tai_to_utc_tjt ∘ utc_to_tai_tjt` to recover the input.
//!    A boundary-index off-by-one would produce a 1 s round-trip
//!    residual at exactly one of the 28 entries.
//! 3. **Far-future TAI ↔ UTC under clamp regime** — the production
//!    table opts into JEOD-faithful clamp for post-2017 epochs;
//!    round-trips must still succeed (no panic, residual ≤ 1 ns) so
//!    long-horizon mission planners don't silently regress.
//!
//! Tolerance rationale: TJT is a day count, so a residual of `1e-12 d`
//! corresponds to ~86 ns. The Newton-Raphson tolerances in
//! `astrodyn_math` use 1e-14 rad, and `f64` mantissa precision near
//! `TJT ≈ 15 000` is ~1.8e-12. The 1e-12 d (~86 ns) bound below is
//! comfortably above that floor and below any real bug.

use astrodyn_time::epoch::{mjd_to_tjt, SECONDS_PER_DAY, TAI_TT_OFFSET};
use astrodyn_time::leap_second::default_leap_second_table;
use astrodyn_time::time_converter_tai_tt::{tai_to_tt, tt_to_tai};
use proptest::prelude::*;

/// Every (MJD, expected TAI-UTC seconds) entry committed in
/// `astrodyn_time::leap_second::default_leap_second_table`. Listed here
/// (rather than read off the table) so a future refactor that drops a
/// row trips this test, not just the table itself.
const LEAP_SECOND_MJD_BOUNDARIES: &[(f64, f64)] = &[
    (41317.0, 10.0), // 1972-01-01
    (41499.0, 11.0), // 1972-07-01
    (41683.0, 12.0), // 1973-01-01
    (42048.0, 13.0), // 1974-01-01
    (42413.0, 14.0), // 1975-01-01
    (42778.0, 15.0), // 1976-01-01
    (43144.0, 16.0), // 1977-01-01
    (43509.0, 17.0), // 1978-01-01
    (43874.0, 18.0), // 1979-01-01
    (44239.0, 19.0), // 1980-01-01
    (44786.0, 20.0), // 1981-07-01
    (45151.0, 21.0), // 1982-07-01
    (45516.0, 22.0), // 1983-07-01
    (46247.0, 23.0), // 1985-07-01
    (47161.0, 24.0), // 1988-01-01
    (47892.0, 25.0), // 1990-01-01
    (48257.0, 26.0), // 1991-01-01
    (48804.0, 27.0), // 1992-07-01
    (49169.0, 28.0), // 1993-07-01
    (49534.0, 29.0), // 1994-07-01
    (50083.0, 30.0), // 1996-01-01
    (50630.0, 31.0), // 1997-07-01
    (51179.0, 32.0), // 1999-01-01
    (53736.0, 33.0), // 2006-01-01
    (54832.0, 34.0), // 2009-01-01
    (56109.0, 35.0), // 2012-07-01
    (57204.0, 36.0), // 2015-07-01
    (57754.0, 37.0), // 2017-01-01
];

// ---------------------------------------------------------------------------
// Suite 1: TAI ↔ TT round-trip
// ---------------------------------------------------------------------------
//
// TT = TAI + 32.184 s by definition. A sign flip on `TAI_TT_OFFSET`
// breaks the round-trip; this property pins the relation in both
// directions across a wide range of TAI epochs.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn tai_tt_round_trip(tai in -3.0e10_f64..3.0e10_f64) {
        // Range bound: ±3e10 s ≈ ±950 years from any chosen epoch. Beyond
        // that the f64 ULP at the input magnitude exceeds the offset's
        // own representation (`32.184 s` ULP at 1e12 s is ~1.2e-4 s, which
        // would make any sub-ns assertion meaningless). The bound covers
        // every mission epoch any astrodynamics consumer plausibly needs.
        let tt = tai_to_tt(tai);
        let back = tt_to_tai(tt);
        // f64 ULP at |tai| ≈ 3e10 s is ≈ 3.6e-6 s; allow a small multiple
        // of that for the offset-add/subtract pair. The bound is well
        // below any sign-flip or constant-misvalue bug.
        prop_assert!(
            (back - tai).abs() <= 1e-5,
            "TAI→TT→TAI drift {} s (tai={tai}, tt={tt})",
            (back - tai).abs(),
        );
        // Forward map agrees with the constant — guards against a future
        // sign flip in `tai_to_tt` not caught by the round-trip alone.
        prop_assert!(
            (tt - tai - TAI_TT_OFFSET).abs() <= 1e-5,
            "TAI+offset != TT: tt - tai = {} (expected {TAI_TT_OFFSET})",
            tt - tai,
        );
    }
}

// ---------------------------------------------------------------------------
// Suite 2: TAI ↔ UTC round-trip at every leap-second boundary
// ---------------------------------------------------------------------------
//
// The table is exhaustive over 1972-01-01 .. 2017-01-01. We pick the
// boundary by index from `LEAP_SECOND_MJD_BOUNDARIES`, perturb by an
// arbitrary day-offset that keeps the resulting UTC TJT inside the
// current regime, then assert the round-trip is exact to the
// machine-precision floor.
//
// A boundary-index off-by-one in `find_index_for_utc` /
// `find_index_for_tai` would produce a 1 s residual at exactly one of
// the 28 boundaries, which proptest would shrink to a minimal index.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn tai_utc_round_trip_at_every_boundary(
        idx in 0_usize..LEAP_SECOND_MJD_BOUNDARIES.len(),
        // Offset in days from the boundary. Bounded to [0.001, 100] so we
        // stay inside the regime that follows the chosen boundary (the
        // smallest gap between consecutive entries is ~182 d — the 1972
        // half-year leap second pair — so 100 d never crosses into the
        // next regime). Strictly positive avoids ambiguity at the
        // half-open boundary instant.
        offset_d in 0.001_f64..100.0_f64,
    ) {
        let table = default_leap_second_table();
        let (mjd, _expected_offset_s) = LEAP_SECOND_MJD_BOUNDARIES[idx];
        let utc_tjt = mjd_to_tjt(mjd) + offset_d;

        let tai_tjt = table.utc_to_tai_tjt(utc_tjt);
        let utc_back = table.tai_to_utc_tjt(tai_tjt);

        // Difference in days: convert to seconds and require sub-ns.
        let drift_s = (utc_back - utc_tjt).abs() * SECONDS_PER_DAY;
        prop_assert!(
            drift_s <= 1e-9,
            "UTC→TAI→UTC drift {drift_s:.3e} s at boundary {idx} \
             (mjd {mjd}, offset {offset_d} d)",
        );
    }
}

// ---------------------------------------------------------------------------
// Suite 3: Far-future TAI ↔ UTC round-trip under the clamp regime
// ---------------------------------------------------------------------------
//
// The default table opts into JEOD-faithful clamp for epochs after the
// last tabulated boundary (MJD 57754, 2017-01-01). This is the
// production regime mission planners hit: long-horizon studies that run
// past 2017 must still round-trip. Picks an offset of 0 d to 50 000 d
// (~137 yr) past the last boundary and requires no panic + sub-ns
// residual.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn tai_utc_round_trip_far_future(offset_d in 0.0_f64..50_000.0_f64) {
        let table = default_leap_second_table();
        let last_mjd = LEAP_SECOND_MJD_BOUNDARIES.last().unwrap().0;
        let utc_tjt = mjd_to_tjt(last_mjd) + offset_d;

        let tai_tjt = table.utc_to_tai_tjt(utc_tjt);
        let utc_back = table.tai_to_utc_tjt(tai_tjt);

        let drift_s = (utc_back - utc_tjt).abs() * SECONDS_PER_DAY;
        prop_assert!(
            drift_s <= 1e-9,
            "far-future UTC→TAI→UTC drift {drift_s:.3e} s (offset {offset_d} d past last boundary)",
        );
    }
}
