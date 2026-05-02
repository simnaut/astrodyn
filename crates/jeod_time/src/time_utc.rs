//! UTC time scale — calendar representation and TAI↔UTC conversion.
//!
//! Ported from JEOD `time_utc.cc` and `time_standard.cc`.
//!
//! UTC tracks Coordinated Universal Time with both a seconds-since-epoch
//! representation and a Gregorian calendar decomposition. The TAI↔UTC
//! conversion uses the leap second table (see `leap_second.rs`).

use crate::epoch::SECONDS_PER_DAY;

/// UTC epoch as TAI TJT: Jan 1, 2000 11:58:55.816 UTC.
/// From JEOD `time_utc.cc`: `tjt_at_epoch = 11544.49925712963`.
pub const UTC_EPOCH_TAI_TJT: f64 = 11_544.499_257_129_63;

/// Gregorian calendar date+time representation.
///
/// Matches JEOD's `TimeStandard` calendar fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalendarDate {
    /// Gregorian year (e.g., `2025`).
    pub year: i32,
    /// Calendar month, 1–12.
    pub month: i32,
    /// Day of month, 1–31.
    pub day: i32,
    /// Hour of day, 0–23.
    pub hour: i32,
    /// Minute of hour, 0–59.
    pub minute: i32,
    /// Second of minute, in `[0.0, 61.0)`. The standard range is
    /// `[0.0, 60.0)`; values in `[60.0, 61.0)` are reserved for the
    /// positive UTC leap second (`23:59:60` and any fractional second
    /// within it).
    pub second: f64,
}

impl CalendarDate {
    /// Create a new calendar date.
    pub fn new(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }
}

/// Convert a Gregorian calendar date to truncated Julian time (TJT).
///
/// Ported from JEOD `time_standard.cc::convert_from_calendar()`.
/// Coverage: March 1, 1600 onward.
///
/// # Panics
/// Panics if the date is before 1600-03-01 or if any field is out of range.
pub fn calendar_to_tjt(cal: &CalendarDate) -> f64 {
    assert!(
        (1..=12).contains(&cal.month),
        "month must be 1..=12, got {}",
        cal.month
    );
    assert!(
        (1..=31).contains(&cal.day),
        "day must be 1..=31, got {}",
        cal.day
    );
    assert!(
        (0..=23).contains(&cal.hour),
        "hour must be 0..=23, got {}",
        cal.hour
    );
    assert!(
        (0..=59).contains(&cal.minute),
        "minute must be 0..=59, got {}",
        cal.minute
    );
    assert!(
        cal.second >= 0.0 && cal.second < 61.0,
        "second must be in [0, 61), got {}",
        cal.second
    );
    assert!(
        cal.year > 1600 || (cal.year == 1600 && cal.month >= 3),
        "date must be 1600-03-01 or later, got {}-{:02}-{:02}",
        cal.year,
        cal.month,
        cal.day
    );

    // Adjust year to represent whole years since March 1, 1600
    let y = cal.year - 1601 + (cal.month + 9) / 12;

    let n_400 = y / 400;
    let y_400 = y - 400 * n_400;

    let n_100 = y_400 / 100;
    let y_100 = y_400 - 100 * n_100;

    let n_4 = y_100 / 4;
    let n_1 = y_100 - 4 * n_4;

    // Partial months since end of February in a 4-year boundary
    let m = cal.month - 2 + 12 * (1 - ((cal.month + 9) / 12) + n_1);

    (cal.day as f64) + ((30.585 * m as f64).floor()) - (2 * n_1) as f64 - 31.0
        + (1461 * n_4) as f64
        + (36524 * n_100) as f64
        + (146097 * n_400) as f64
        - 134493.0
        + (cal.hour as f64 / 24.0)
        + (cal.minute as f64 / 1440.0)
        + (cal.second / SECONDS_PER_DAY)
}

/// Convert truncated Julian time (TJT) to a Gregorian calendar date.
///
/// Ported from JEOD `time_standard.cc::calculate_calendar_values()`.
/// Coverage: March 1, 1600 onward.
///
/// Note: this function does not represent UTC leap seconds. The `second`
/// field is always in `[0, 60)` — values within 1e-6 of 60.0 are rounded
/// up to the next minute. A `23:59:60` leap second instant cannot be
/// produced. This matches JEOD's `calculate_calendar_values()` behavior.
pub fn tjt_to_calendar(tjt: f64) -> CalendarDate {
    let mut julian_day = tjt.floor() as i32;

    // Minutes in fractional day
    let temp = 1440.0 * (tjt - julian_day as f64);
    let mut calendar_minute = temp as i32;
    let mut calendar_second = 60.0 * (temp - calendar_minute as f64);

    // Clock resolution: JEOD uses 1e-6 by default
    let clock_resolution = 1e-6;
    if calendar_second > 60.0 - clock_resolution {
        calendar_minute += 1;
        if calendar_minute >= 1440 {
            julian_day += 1;
            calendar_minute -= 1440;
        }
        calendar_second = 0.0;
    }

    let calendar_hour = calendar_minute / 60;
    calendar_minute -= calendar_hour * 60;

    // Convert integral part to day, month, year
    let jd = julian_day + 134493; // offset from March 1, 1600
    let mut n_400 = jd / 146097;
    if jd < 0 {
        n_400 -= 1;
    }
    let r_400 = jd - 146097 * n_400 + 1;

    let n_100 = (r_400 as f64 / 36524.3) as i32;
    let r_100 = r_400 - 36524 * n_100 - 1;

    let n_4 = r_100 / 1461;
    let r_4 = r_100 - 1461 * n_4 + 1;

    let n_1 = (r_4 as f64 / 365.3) as i32;

    let r_4a = r_4 + 30 + 2 * n_1;
    let m = (r_4a as f64 / 30.585) as i32;

    let calendar_day = r_4a - (30.585 * m as f64) as i32;
    let calendar_month = m + 2 - 12 * ((m + 1) / 12);
    let calendar_year = 1600 + 400 * n_400 + 100 * n_100 + 4 * n_4 + ((m + 1) / 12);

    CalendarDate {
        year: calendar_year,
        month: calendar_month,
        day: calendar_day,
        hour: calendar_hour,
        minute: calendar_minute,
        second: calendar_second,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_to_tjt_j2000() {
        // J2000.0 = 2000-01-01 12:00:00 TT
        // TJT for J2000 TT = 11544.5
        let cal = CalendarDate::new(2000, 1, 1, 12, 0, 0.0);
        let tjt = calendar_to_tjt(&cal);
        assert!(
            (tjt - 11544.5).abs() < 1e-10,
            "J2000 TJT: expected 11544.5, got {}",
            tjt
        );
    }

    #[test]
    fn calendar_to_tjt_round_trip() {
        let dates = [
            CalendarDate::new(1972, 1, 1, 0, 0, 0.0),
            CalendarDate::new(2000, 1, 1, 12, 0, 0.0),
            CalendarDate::new(2024, 6, 15, 14, 30, 45.123),
            CalendarDate::new(1980, 1, 6, 0, 0, 0.0), // GPS epoch
        ];

        for cal in &dates {
            let tjt = calendar_to_tjt(cal);
            let back = tjt_to_calendar(tjt);
            assert_eq!(back.year, cal.year, "year mismatch for {:?}", cal);
            assert_eq!(back.month, cal.month, "month mismatch for {:?}", cal);
            assert_eq!(back.day, cal.day, "day mismatch for {:?}", cal);
            assert_eq!(back.hour, cal.hour, "hour mismatch for {:?}", cal);
            assert_eq!(back.minute, cal.minute, "minute mismatch for {:?}", cal);
            assert!(
                (back.second - cal.second).abs() < 1e-6,
                "second mismatch for {:?}: got {}",
                cal,
                back.second
            );
        }
    }

    #[test]
    fn tjt_to_calendar_j2000() {
        let cal = tjt_to_calendar(11544.5);
        assert_eq!(cal.year, 2000);
        assert_eq!(cal.month, 1);
        assert_eq!(cal.day, 1);
        assert_eq!(cal.hour, 12);
        assert_eq!(cal.minute, 0);
        assert!(cal.second.abs() < 1e-6);
    }

    #[test]
    fn leap_year_feb_29() {
        // 2000 is a leap year
        let cal = CalendarDate::new(2000, 2, 29, 0, 0, 0.0);
        let tjt = calendar_to_tjt(&cal);
        let back = tjt_to_calendar(tjt);
        assert_eq!(back.year, 2000);
        assert_eq!(back.month, 2);
        assert_eq!(back.day, 29);
    }

    #[test]
    fn dec_31_to_jan_1_transition() {
        let dec31 = CalendarDate::new(2016, 12, 31, 23, 59, 59.0);
        let tjt_dec31 = calendar_to_tjt(&dec31);
        // 2 seconds later should be 2017-01-01 00:00:01
        let tjt_next = tjt_dec31 + 2.0 / SECONDS_PER_DAY;
        let cal = tjt_to_calendar(tjt_next);
        assert_eq!(cal.year, 2017);
        assert_eq!(cal.month, 1);
        assert_eq!(cal.day, 1);
        assert_eq!(cal.hour, 0);
        assert_eq!(cal.minute, 0);
        assert!((cal.second - 1.0).abs() < 1e-6);
    }
}
