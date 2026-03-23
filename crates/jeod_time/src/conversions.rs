use crate::epoch::{TAI_TT_OFFSET, J2000_TAI_TJT};
use std::f64::consts::PI;

/// Convert TAI seconds-since-epoch to TT seconds-since-epoch.
/// TT = TAI + 32.184s (exact by definition).
pub fn tai_to_tt(tai_seconds: f64) -> f64 {
    tai_seconds + TAI_TT_OFFSET
}

/// Convert TT seconds-since-epoch to TAI seconds-since-epoch.
pub fn tt_to_tai(tt_seconds: f64) -> f64 {
    tt_seconds - TAI_TT_OFFSET
}

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
        if tai.abs() > 0.0 && (dtai / tai).abs() < 1.0e-15 {
            break;
        }
    }
    tai
}

/// Convert UT1 days-since-J2000 to Greenwich Mean Sidereal Time (GMST).
///
/// Returns GMST as a fraction of a sidereal day (0.0 to 1.0, wraps).
///
/// Ported from JEOD `time_converter_ut1_gmst.cc`:
/// ```text
/// dd = ut1_days - 0.000738762
/// gmst = 0.7790572733 + 1.002737909350795*dd + 8.0775E-16*dd^2 - 1.5E-24*dd^3
/// ```
pub fn ut1_to_gmst_days(ut1_days_since_j2000: f64) -> f64 {
    let dd = ut1_days_since_j2000 - 0.000738762;
    let dd2 = dd * dd;
    let dd3 = dd2 * dd;
    let gmst = 0.7790572733 + 1.002737909350795 * dd + 8.0775e-16 * dd2 - 1.5e-24 * dd3;
    // Return fractional part (0..1)
    gmst - gmst.floor()
}

/// Convert UT1 days-since-J2000 to GMST in radians (0 to 2π).
pub fn ut1_to_gmst_radians(ut1_days_since_j2000: f64) -> f64 {
    ut1_to_gmst_days(ut1_days_since_j2000) * 2.0 * PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tai_tt_exact_offset() {
        assert_eq!(tai_to_tt(0.0), 32.184);
        assert_eq!(tai_to_tt(100.0), 132.184);
        assert_eq!(tt_to_tai(32.184), 0.0);
    }

    #[test]
    fn tai_tt_round_trip() {
        let tai = 123456.789;
        let tt = tai_to_tt(tai);
        let back = tt_to_tai(tt);
        assert!((back - tai).abs() < 1e-15);
    }

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
    fn gmst_at_j2000() {
        // The GMST formula constant term gives 0.7790572733 fractional days
        // at dd=0. But dd = ut1_days - 0.000738762, so at ut1_days=0,
        // dd = -0.000738762 and gmst_days = 0.7790572733 - 1.002737909350795*0.000738762
        //                                  ≈ 0.7783165
        // = 280.19° (this is GMST at J2000 TT, not J2000 UT1)
        let gmst_days = ut1_to_gmst_days(0.0);
        let gmst_deg = gmst_days * 360.0;
        assert!(
            (gmst_deg - 280.19).abs() < 0.1,
            "GMST at UT1 days=0: {} degrees",
            gmst_deg
        );
    }

    #[test]
    fn gmst_is_bounded() {
        for days in [-1000.0, 0.0, 365.25, 3652.5] {
            let gmst = ut1_to_gmst_days(days);
            assert!(
                (0.0..1.0).contains(&gmst),
                "GMST fractional days out of range: {} at days={}",
                gmst,
                days
            );
        }
    }
}
