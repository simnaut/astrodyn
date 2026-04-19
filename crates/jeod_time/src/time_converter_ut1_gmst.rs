use crate::{SecondsSince, GMST};
use std::f64::consts::PI;
use uom::si::angle::radian;
use uom::si::f64::Angle;

/// Convert UT1 days since noon 2000-01-01 to Greenwich Mean Sidereal Time.
///
/// Returns GMST in sidereal days since J2000, matching JEOD's
/// `TimeGMST::set_time_by_days()` output (accumulated, not wrapped).
///
/// The argument is `d_u = JD(UT1) - 2451545.0 = ut1_tjt - 11544.5`, i.e.
/// the number of UT1 days since noon on 2000-01-01. This is the independent
/// variable in the Astronomical Almanac GMST polynomial (Aoki et al. 1982).
///
/// Matches JEOD `time_converter_ut1_gmst.cc` when called with the equivalent
/// of `ut1_ptr->trunc_julian_time - 11544.5`.
pub fn ut1_to_gmst_days(du: f64) -> f64 {
    let du2 = du * du;
    let du3 = du2 * du;
    0.7790572733 + 1.002737909350795 * du + 8.0775e-16 * du2 - 1.5e-24 * du3
}

/// Convert UT1 days since noon 2000-01-01 to GMST in sidereal seconds.
///
/// Matches JEOD's `TimeGMST::seconds` field.
pub fn ut1_to_gmst_seconds(du: f64) -> f64 {
    ut1_to_gmst_days(du) * 86400.0
}

/// Convert UT1 days since noon 2000-01-01 to GMST in radians (0 to 2π).
///
/// Takes the fractional-day part of GMST and converts to radians.
pub fn ut1_to_gmst_radians(du: f64) -> f64 {
    let gmst_days = ut1_to_gmst_days(du);
    let fractional = gmst_days - gmst_days.floor();
    fractional * 2.0 * PI
}

// --- Typed variants (Phase 1 #103) ------------------------------------------
//
// The GMST polynomial (Aoki et al. 1982) is parameterised by `du`, UT1 days
// since J2000 noon — a scale-neutral scalar anchor rather than a
// `SecondsSince<UT1>`. The typed wrappers therefore keep the `du: f64` input
// and return typed output: a `uom::si::f64::Angle` for the wrapped radian
// form (GMST's primary representation) and `SecondsSince<GMST>` for the
// accumulated sidereal-seconds form.

/// Typed UT1 → GMST accumulated sidereal seconds (delegates to
/// [`ut1_to_gmst_seconds`]).
pub fn ut1_to_gmst_seconds_typed(du: f64) -> SecondsSince<GMST> {
    SecondsSince::from_seconds(ut1_to_gmst_seconds(du))
}

/// Typed UT1 → GMST angle wrapped to `[0, 2π)` (delegates to
/// [`ut1_to_gmst_radians`]).
pub fn ut1_to_gmst_angle(du: f64) -> Angle {
    Angle::new::<radian>(ut1_to_gmst_radians(du))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmst_at_j2000() {
        // At du=0 (noon 2000-01-01), GMST = 0.7790572733 sidereal days
        // = 280.46° — the standard Astronomical Almanac value.
        let gmst_days = ut1_to_gmst_days(0.0);
        let gmst_deg = (gmst_days - gmst_days.floor()) * 360.0;
        assert!(
            (gmst_deg - 280.46).abs() < 0.01,
            "GMST at noon J2000: {:.4} degrees, expected ~280.46",
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

    #[test]
    fn ut1_gmst_typed_matches_f64() {
        let du = 365.25_f64;
        let typed_secs = ut1_to_gmst_seconds_typed(du);
        assert_eq!(typed_secs.as_seconds(), ut1_to_gmst_seconds(du));

        let typed_angle = ut1_to_gmst_angle(du);
        assert_eq!(typed_angle.get::<radian>(), ut1_to_gmst_radians(du));
    }

    #[test]
    fn ut1_gmst_typed_angle_is_bounded() {
        for days in [-1000.0, 0.0, 365.25, 3652.5] {
            let angle_rad = ut1_to_gmst_angle(days).get::<radian>();
            assert!(
                (0.0..std::f64::consts::TAU).contains(&angle_rad),
                "typed GMST angle out of range: {} at days={}",
                angle_rad,
                days
            );
        }
    }
}
