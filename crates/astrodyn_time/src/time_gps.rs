//! GPS time scale — constant offset from TAI plus GPS week/day decomposition.
//!
//! Ported from JEOD `time_gps.cc` and `time_converter_tai_gps.cc`.
//!
//! GPS time = TAI - 19 seconds (the number of leap seconds accumulated
//! by the GPS epoch, January 6, 1980 00:00:00 UTC).

use crate::epoch::SECONDS_PER_DAY;

/// TAI-GPS offset in seconds: GPS = TAI - 19s.
///
/// This is the number of leap seconds that had accumulated by the GPS
/// epoch (1980-01-06 00:00:00 UTC). The offset is constant by definition.
pub const TAI_GPS_OFFSET: f64 = 19.0;

/// GPS epoch as TAI TJT.
///
/// From JEOD `time_gps.cc`: GPS epoch is midnight Jan 5/6, 1980 UTC.
/// UTC TJT at that point is 4244.0. TAI TJT = UTC TJT + 19/86400.
pub const GPS_EPOCH_TAI_TJT: f64 = 4244.0 + 19.0 / SECONDS_PER_DAY;

/// Convert TAI seconds-since-epoch to GPS seconds-since-epoch.
pub fn tai_to_gps(tai_seconds: f64) -> f64 {
    tai_seconds - TAI_GPS_OFFSET
}

/// Convert GPS seconds-since-epoch to TAI seconds-since-epoch.
pub fn gps_to_tai(gps_seconds: f64) -> f64 {
    gps_seconds + TAI_GPS_OFFSET
}

/// GPS week/day decomposition from GPS elapsed days since GPS epoch.
///
/// Matches JEOD `TimeGPS::set_time_by_seconds()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpsTimeComponents {
    /// Seconds elapsed in the current (partial) day.
    pub seconds_of_day: f64,
    /// Seconds elapsed in the current (partial) week.
    pub seconds_of_week: f64,
    /// Number of whole days this week (0-6).
    pub day_of_week: i32,
    /// Number of 1024-week rollovers since GPS epoch.
    pub rollover_count: i32,
    /// Week number within current 1024-week rollover period.
    pub week: i32,
    /// Number of 8192-week (13-bit) rollovers since GPS epoch.
    pub rollover_count_13_bit: i32,
    /// Week number within current 8192-week period.
    pub week_13_bit: i32,
}

/// Decompose GPS elapsed days into week/day components.
///
/// `gps_days` is the number of days elapsed since the GPS epoch.
///
/// Ported from JEOD `time_gps.cc::set_time_by_seconds()`.
pub fn gps_components(gps_days: f64) -> GpsTimeComponents {
    let gps_time_int = gps_days.floor() as i32;
    let seconds_of_day = (gps_days - gps_time_int as f64) * SECONDS_PER_DAY;

    // 10-bit rollover: 1024 weeks = 7168 days
    let rollover_count = gps_time_int.div_euclid(7168);
    let gps_in_rollover = gps_time_int.rem_euclid(7168);

    // 13-bit rollover: 8192 weeks = 57344 days
    let rollover_count_13_bit = gps_time_int.div_euclid(57344);
    let gps_in_13_bit_rollover = gps_time_int.rem_euclid(57344);
    let week_13_bit = gps_in_13_bit_rollover / 7;

    let week = gps_in_rollover / 7;
    let day_of_week = gps_in_rollover % 7;
    let seconds_of_week = day_of_week as f64 * SECONDS_PER_DAY + seconds_of_day;

    GpsTimeComponents {
        seconds_of_day,
        seconds_of_week,
        day_of_week,
        rollover_count,
        week,
        rollover_count_13_bit,
        week_13_bit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tai_gps_offset() {
        assert_eq!(tai_to_gps(0.0), -19.0);
        assert_eq!(tai_to_gps(100.0), 81.0);
        assert_eq!(gps_to_tai(-19.0), 0.0);
    }

    #[test]
    fn tai_gps_round_trip() {
        let tai = 123456.789;
        let gps = tai_to_gps(tai);
        let back = gps_to_tai(gps);
        assert!((back - tai).abs() < 1e-15);
    }

    #[test]
    fn gps_components_basic() {
        // 0 days = start of GPS epoch
        let c = gps_components(0.0);
        assert_eq!(c.day_of_week, 0);
        assert_eq!(c.week, 0);
        assert_eq!(c.rollover_count, 0);

        // 7 days = 1 week
        let c = gps_components(7.0);
        assert_eq!(c.week, 1);
        assert_eq!(c.day_of_week, 0);

        // 7168 days = first 1024-week rollover
        let c = gps_components(7168.0);
        assert_eq!(c.rollover_count, 1);
        assert_eq!(c.week, 0);
        assert_eq!(c.day_of_week, 0);
    }

    #[test]
    fn gps_components_mid_week() {
        // 3.5 days = Wednesday noon
        let c = gps_components(3.5);
        assert_eq!(c.day_of_week, 3);
        assert!((c.seconds_of_day - 43200.0).abs() < 1e-6);
        assert!(
            (c.seconds_of_week - (3.0 * 86400.0 + 43200.0)).abs() < 1e-6,
            "seconds_of_week: {}",
            c.seconds_of_week
        );
    }
}
