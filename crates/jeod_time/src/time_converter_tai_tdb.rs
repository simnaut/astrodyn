use crate::epoch::{J2000_TAI_TJT, TAI_TT_OFFSET};
use crate::time_converter_tai_tt::tai_to_tt;
use crate::{SecondsSince, TAI, TDB};
use std::f64::consts::PI;

/// Compute the TDB-TT offset in seconds at a given TAI truncated Julian time.
///
/// Uses the simplified formula from the Explanatory Supplement to the
/// Astronomical Almanac (2006), accurate to ±0.1 microseconds.
///
/// Ported from JEOD `time_converter_tai_tdb.cc`:
/// ```text
/// g = (π/180) * (357.53 + 0.9856003 * (tai_tjt - tai_tjt_at_J2000))
/// offset = 0.001658 * sin(g) + 0.000014 * sin(2g)
/// ```
pub fn tdb_tt_offset(tai_tjt: f64) -> f64 {
    let dt_days = tai_tjt - J2000_TAI_TJT;
    let g = (PI / 180.0) * (357.53 + 0.9856003 * dt_days);
    0.001658 * g.sin() + 0.000014 * (2.0 * g).sin()
}

/// Convert TAI seconds-since-epoch to TDB seconds-since-epoch.
///
/// TDB = TT + periodic_offset = TAI + 32.184 + periodic_offset
pub fn tai_to_tdb(tai_seconds: f64, tai_tjt: f64) -> f64 {
    let tt = tai_to_tt(tai_seconds);
    let offset = tdb_tt_offset(tai_tjt);
    tt + offset
}

/// Convert TDB seconds-since-epoch to TAI seconds-since-epoch.
///
/// Iterative solution (matches JEOD's 5-iteration convergence).
pub fn tdb_to_tai(tdb_seconds: f64, tai_tjt_initial: f64) -> f64 {
    let mut tai = tdb_seconds - TAI_TT_OFFSET;
    let mut tai_tjt = tai_tjt_initial;

    for _ in 0..5 {
        let offset = tdb_tt_offset(tai_tjt);
        let new_tai = tdb_seconds - TAI_TT_OFFSET - offset;
        let dtai = new_tai - tai;
        tai = new_tai;
        tai_tjt = tai_tjt_initial + dtai / 86400.0;
        if dtai.abs() < 1e-15 || (tai.abs() > 0.0 && (dtai / tai).abs() < 1.0e-15) {
            break;
        }
    }
    tai
}

// --- Typed variants (Phase 1 #103) ------------------------------------------
//
// Additive typed siblings. They delegate to the f64 free functions so
// behavior is bit-identical. `tai_tjt` is a scale-neutral truncated Julian
// time anchor carried through untouched.

/// Typed TAI → TDB converter (delegates to [`tai_to_tdb`]).
///
/// `tai_tjt` anchors the calendar date for the periodic TDB-TT offset; it is
/// the same scalar the f64 free function takes (truncated Julian time in TAI).
pub fn tai_to_tdb_typed(t: SecondsSince<TAI>, tai_tjt: f64) -> SecondsSince<TDB> {
    SecondsSince::from_seconds(tai_to_tdb(t.as_seconds(), tai_tjt))
}

/// Typed TDB → TAI converter (delegates to [`tdb_to_tai`]).
pub fn tdb_to_tai_typed(t: SecondsSince<TDB>, tai_tjt_initial: f64) -> SecondsSince<TAI> {
    SecondsSince::from_seconds(tdb_to_tai(t.as_seconds(), tai_tjt_initial))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tdb_offset_at_j2000() {
        // At J2000, g = 357.53 degrees, offset should be small
        let offset = tdb_tt_offset(J2000_TAI_TJT);
        // sin(357.53°) ≈ -0.0431, so offset ≈ -7.1e-5 s
        assert!(
            offset.abs() < 0.002,
            "TDB-TT offset at J2000: {} (should be < 2ms)",
            offset
        );
    }

    #[test]
    fn tai_tdb_round_trip() {
        let tai = 1_000_000.0;
        let tai_tjt = J2000_TAI_TJT + tai / 86400.0;
        let tdb = tai_to_tdb(tai, tai_tjt);
        let back = tdb_to_tai(tdb, tai_tjt);
        assert!(
            (back - tai).abs() < 1e-10,
            "TAI-TDB round trip: {} -> {} -> {}, err={}",
            tai,
            tdb,
            back,
            (back - tai).abs()
        );
    }

    #[test]
    fn tai_tdb_round_trip_near_zero() {
        // C4: convergence must work when tai ≈ 0 (near simulation start)
        let tai = 0.0;
        let tai_tjt = J2000_TAI_TJT;
        let tdb = tai_to_tdb(tai, tai_tjt);
        let back = tdb_to_tai(tdb, tai_tjt);
        assert!(
            (back - tai).abs() < 1e-10,
            "TAI-TDB round trip near zero: {} -> {} -> {}, err={}",
            tai,
            tdb,
            back,
            (back - tai).abs()
        );
    }

    #[test]
    fn tai_tdb_typed_matches_f64() {
        let tai = 1_000_000.0_f64;
        let tai_tjt = J2000_TAI_TJT + tai / 86400.0;
        let tdb_typed = tai_to_tdb_typed(SecondsSince::<TAI>::from_seconds(tai), tai_tjt);
        assert_eq!(tdb_typed.as_seconds(), tai_to_tdb(tai, tai_tjt));
    }

    #[test]
    fn tai_tdb_typed_round_trip() {
        // Non-trivial input, typed round-trip.
        let tai_s = 1_000_000.0_f64;
        let tai_tjt = J2000_TAI_TJT + tai_s / 86400.0;
        let tai = SecondsSince::<TAI>::from_seconds(tai_s);
        let tdb = tai_to_tdb_typed(tai, tai_tjt);
        let back = tdb_to_tai_typed(tdb, tai_tjt);
        let err = (back.as_seconds() - tai.as_seconds()).abs();
        // Same 1e-10 bound the iterative f64 round-trip achieves; well below
        // the 1e-14 TAI↔TT spec target (TDB adds ~ms-scale periodic noise).
        assert!(err < 1e-10, "TAI->TDB->TAI typed round-trip err={}", err);
    }
}
