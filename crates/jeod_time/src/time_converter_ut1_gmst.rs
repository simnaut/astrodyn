use std::f64::consts::PI;

/// Convert UT1 days-since-J2000 to Greenwich Mean Sidereal Time (GMST).
///
/// Returns GMST in sidereal days since J2000, matching JEOD's
/// `TimeGMST::set_time_by_days()` output (accumulated, not wrapped).
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
    0.7790572733 + 1.002737909350795 * dd + 8.0775e-16 * dd2 - 1.5e-24 * dd3
}

/// Convert UT1 days-since-J2000 to GMST in sidereal seconds since J2000.
///
/// Matches JEOD's `TimeGMST::seconds` field.
pub fn ut1_to_gmst_seconds(ut1_days_since_j2000: f64) -> f64 {
    ut1_to_gmst_days(ut1_days_since_j2000) * 86400.0
}

/// Convert UT1 days-since-J2000 to GMST in radians (0 to 2π).
///
/// Takes the fractional-day part of GMST and converts to radians.
pub fn ut1_to_gmst_radians(ut1_days_since_j2000: f64) -> f64 {
    let gmst_days = ut1_to_gmst_days(ut1_days_since_j2000);
    let fractional = gmst_days - gmst_days.floor();
    fractional * 2.0 * PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmst_at_j2000() {
        // At ut1_days=0, dd = -0.000738762
        // gmst_days ≈ 0.7783165
        // = 280.19° (GMST at J2000 TT, not J2000 UT1)
        let gmst_days = ut1_to_gmst_days(0.0);
        let gmst_deg = (gmst_days - gmst_days.floor()) * 360.0;
        assert!(
            (gmst_deg - 280.19).abs() < 0.1,
            "GMST at UT1 days=0: {} degrees",
            gmst_deg
        );
    }

    #[test]
    fn gmst_accumulates() {
        // GMST should grow over time (not wrap to [0,1))
        let gmst_0 = ut1_to_gmst_days(0.0);
        let gmst_365 = ut1_to_gmst_days(365.25);
        // After ~1 year, GMST should have accumulated ~366.25 sidereal days
        assert!(
            gmst_365 > gmst_0 + 365.0,
            "GMST should accumulate: gmst_0={}, gmst_365={}",
            gmst_0,
            gmst_365
        );
    }

    #[test]
    fn gmst_seconds_matches_days() {
        let days = 100.0;
        let gmst_days = ut1_to_gmst_days(days);
        let gmst_secs = ut1_to_gmst_seconds(days);
        assert!(
            (gmst_secs - gmst_days * 86400.0).abs() < 1e-10,
            "seconds should be days * 86400"
        );
    }

    #[test]
    fn gmst_radians_is_bounded() {
        for days in [-1000.0, 0.0, 365.25, 3652.5] {
            let gmst_rad = ut1_to_gmst_radians(days);
            assert!(
                (0.0..std::f64::consts::TAU).contains(&gmst_rad),
                "GMST radians out of range: {} at days={}",
                gmst_rad,
                days
            );
        }
    }
}
