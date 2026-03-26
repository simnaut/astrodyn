use crate::epoch::{TAI_TT_OFFSET, J2000_TAI_TJT};
use crate::time_converter_tai_tt::tai_to_tt;
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
            tai, tdb, back, (back - tai).abs()
        );
    }
}
