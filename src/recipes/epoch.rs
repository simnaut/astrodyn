//! Mission epoch recipes.
//!
//! Each function returns a [`SimulationTime`] anchored at a named
//! reference epoch.
//!
//! ```
//! use astrodyn::recipes::epoch;
//! let t = epoch::j2000();
//! assert!(t.tai_tjt > 10_000.0);
//! ```

use crate::SimulationTime;
use astrodyn_time::epoch::SECONDS_PER_DAY;
use astrodyn_time::leap_second::default_leap_second_table;

/// J2000 reference epoch: 2000-01-01 11:58:55.816 TAI (12:00:00 TT).
pub fn j2000() -> SimulationTime {
    SimulationTime::at_j2000(default_leap_second_table())
}

/// Epoch from a TAI truncated Julian time (days since 1969-12-24 00:00:00).
///
/// Used by JEOD verification simulations to anchor reference state at
/// run-specific UTC dates.
pub fn at_tai_tjt(tai_tjt: f64) -> SimulationTime {
    SimulationTime::new(tai_tjt, default_leap_second_table())
}

/// Clementine lunar mission epoch: 1994-03-01 00:00:00 UTC.
///
/// Matches the JEOD `SIM_Earth_Moon RUN_clem` reference scenario
/// (`JD = 2449412.5`, `MJD = 49412`, `TJT = MJD - 40000 = 9412`).
/// TAI-UTC at this instant is 28 s (the 29th leap second was added
/// 1994-07-01), so TAI TJT = 9412 + 28/86400.
pub fn clementine_1994() -> SimulationTime {
    at_tai_tjt(9_412.0 + 28.0 / 86_400.0)
}

/// Dawn-at-Mars epoch: 2009-02-17 23:00:00 UTC (TAI-UTC = 34s).
///
/// Anchors `crates/astrodyn_runner/examples/mars_orbit.rs`.
pub fn dawn_mars_2009() -> SimulationTime {
    at_tai_tjt(14_879.958_727)
}

/// Epoch from a UTC calendar instant (Gregorian).
///
/// Internally computes the Julian Date via Meeus's formula (Astronomical
/// Algorithms, ch. 7), converts to UTC TJT, looks up the leap-second
/// offset, and returns a [`SimulationTime`] anchored at the corresponding
/// TAI TJT. The mandatory leap-second table is the default JEOD-derived
/// one — call [`SimulationTime::new`] directly for a custom table.
///
/// `second` is `f64` to allow fractional seconds (`03.51` etc.).
///
/// ```
/// use astrodyn::recipes::epoch;
/// // J2000.0 corresponds to 2000-01-01 11:58:55.816 UTC (TAI-UTC = 32 s,
/// // TT-TAI = 32.184 s, so UTC = TT − 64.184 s ≈ 11:58:55.816).
/// let t = epoch::at_utc(2000, 1, 1, 11, 58, 55.816);
/// assert!((t.tai_tjt_at_epoch - 11_544.499_627_5).abs() < 1e-7);
/// ```
pub fn at_utc(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: f64,
) -> SimulationTime {
    assert!(
        (1..=12).contains(&month),
        "at_utc: month must be in 1..=12, got {month}"
    );
    assert!(
        (1..=31).contains(&day),
        "at_utc: day must be in 1..=31, got {day}"
    );
    assert!(hour < 24, "at_utc: hour must be in 0..24, got {hour}");
    assert!(minute < 60, "at_utc: minute must be in 0..60, got {minute}");
    assert!(
        second.is_finite() && (0.0..=60.0).contains(&second),
        "at_utc: second must be in 0.0..=60.0, got {second}"
    );

    // Meeus chapter 7: shift Jan/Feb into the previous year so the
    // (m+1)/0.6 leap-month adjustment indexes correctly.
    let (y, m) = if month <= 2 {
        (year - 1, (month + 12) as i32)
    } else {
        (year, month as i32)
    };
    let a = y.div_euclid(100);
    let b = 2 - a + a.div_euclid(4); // Gregorian correction (post-1582)
    let jd_at_day_start = (365.25 * (y + 4716) as f64).floor()
        + (30.6001 * (m + 1) as f64).floor()
        + day as f64
        + b as f64
        - 1524.5;
    let frac_day = (hour as f64 * 3600.0 + minute as f64 * 60.0 + second) / SECONDS_PER_DAY;
    // UTC TJT = JD − 2400000.5 (MJD) − 40000 (TJT offset)
    let utc_tjt = jd_at_day_start + frac_day - 2_440_000.5;
    // Production epoch helpers opt in to JEOD-faithful clamp behavior on the
    // leap-second table (#485 H2) so post-2017 epochs (no new leap seconds
    // inserted; IERS extends the table's last value) and pre-1972 epochs do
    // not panic the construction. Callers needing strict OOR rejection
    // (asserting a covered epoch) should construct the `LeapSecondTable`
    // without `.with_clamp_out_of_range(true)` and call `SimulationTime::new`
    // directly.
    let leap_table = default_leap_second_table().with_clamp_out_of_range(true);
    let tai_tjt = leap_table.utc_to_tai_tjt(utc_tjt);
    SimulationTime::new(tai_tjt, leap_table)
}

/// Epoch from an ISO-8601 / RFC-3339 UTC string.
///
/// Accepts `"YYYY-MM-DDTHH:MM:SS[.fff]Z"` (canonical RFC 3339 with the
/// `Z` UTC marker) or the equivalent space-separated form
/// `"YYYY-MM-DD HH:MM:SS[.fff]"` (without the `Z`). Time-zone offsets
/// other than `Z` / UTC are not parsed — UTC is the only supported
/// reference.
///
/// ```
/// use astrodyn::recipes::epoch;
/// // NESC CC8 epoch — 2026-01-28 06:42:03.51 UTC.
/// let t = epoch::at_iso("2026-01-28T06:42:03.51Z");
/// assert!(t.tai_tjt_at_epoch > 21_000.0 && t.tai_tjt_at_epoch < 21_100.0);
/// ```
pub fn at_iso(s: &str) -> SimulationTime {
    let bytes = s.as_bytes();
    assert!(
        bytes.len() >= 19,
        "at_iso: timestamp too short ({} bytes); expected RFC 3339 form like \
         '2026-01-28T06:42:03.51Z'",
        bytes.len()
    );

    let parse_u32 = |slice: &str, label: &str| -> u32 {
        slice
            .parse::<u32>()
            .unwrap_or_else(|e| panic!("at_iso: invalid {label} {slice:?}: {e}"))
    };
    let year: i32 = s[0..4]
        .parse()
        .unwrap_or_else(|e| panic!("at_iso: invalid year {:?}: {e}", &s[0..4]));
    let month = parse_u32(&s[5..7], "month");
    let day = parse_u32(&s[8..10], "day");
    let sep = bytes[10];
    assert!(
        sep == b'T' || sep == b' ',
        "at_iso: expected 'T' or ' ' separator at byte 10, got {:?}",
        sep as char
    );
    let hour = parse_u32(&s[11..13], "hour");
    let minute = parse_u32(&s[14..16], "minute");
    // Seconds: from byte 17 to first 'Z'/'z' (or end if no Z marker).
    let end = s.find(['Z', 'z']).unwrap_or(s.len());
    let sec_str = &s[17..end];
    let second: f64 = sec_str
        .parse()
        .unwrap_or_else(|e| panic!("at_iso: invalid second {sec_str:?}: {e}"));
    at_utc(year, month, day, hour, minute, second)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "epoch-recipe tests assert bit-exact recovery of TJT field values"
)]
mod tests {
    use super::*;

    /// `at_utc` at J2000 should match the existing `J2000_TAI_TJT` constant
    /// to within a microsecond. J2000.0 is 2000-01-01 12:00 TT, which is
    /// 2000-01-01 11:58:55.816 UTC at TAI-UTC = 32 s.
    #[test]
    fn at_utc_matches_j2000_constant() {
        let t = at_utc(2000, 1, 1, 11, 58, 55.816);
        let expected = astrodyn_time::epoch::J2000_TAI_TJT;
        let err = (t.tai_tjt_at_epoch - expected).abs();
        assert!(
            err < 1e-7,
            "at_utc(J2000) tai_tjt = {}, expected {expected}, err = {err}",
            t.tai_tjt_at_epoch
        );
    }

    /// `at_iso` should round-trip to the same `tai_tjt` as `at_utc` for
    /// matching inputs, including fractional seconds.
    #[test]
    fn at_iso_matches_at_utc_with_fractional_seconds() {
        let from_iso = at_iso("2026-01-28T06:42:03.51Z");
        let from_utc = at_utc(2026, 1, 28, 6, 42, 3.51);
        assert_eq!(from_iso.tai_tjt_at_epoch, from_utc.tai_tjt_at_epoch);
    }

    /// `at_iso` should accept the space-separated NESC-style form too.
    #[test]
    fn at_iso_accepts_space_separator() {
        let from_t = at_iso("2026-01-28T06:42:03.51Z");
        let from_space = at_iso("2026-01-28 06:42:03.51");
        assert_eq!(from_t.tai_tjt_at_epoch, from_space.tai_tjt_at_epoch);
    }

    /// `at_utc` for 2026-01-28 06:42:03.51 UTC (CC8 epoch) should yield a
    /// TAI TJT consistent with the hand-computed value:
    ///
    ///   JD(0h UT)        = 2_461_068.5
    ///   frac_day(06:42:03.51)
    ///                    = 24_123.51 / 86_400 = 0.279_207_291_666_…
    ///   UTC TJT          = JD + frac − 2_440_000.5 = 21_068.279_207_291_666_…
    ///   TAI − UTC at 2026 = 37 s = 37/86_400 = 0.000_428_240_740_740_…
    ///   TAI TJT          = 21_068.279_635_532_407_…
    #[test]
    fn at_utc_matches_cc8_epoch() {
        let t = at_utc(2026, 1, 28, 6, 42, 3.51);
        let expected_tai_tjt = 21_068.279_635_532_407_f64;
        let err = (t.tai_tjt_at_epoch - expected_tai_tjt).abs();
        assert!(
            err < 1e-9,
            "at_utc(CC8) tai_tjt = {}, expected {expected_tai_tjt}, err = {err}",
            t.tai_tjt_at_epoch
        );
    }

    /// [`clementine_1994`] must anchor at the JEOD `SIM_Earth_Moon RUN_clem`
    /// epoch (1994-03-01 00:00:00 UTC, TAI-UTC = 28 s). The recipe and the
    /// hand-rolled `SimulationTime::new(9412.0 + 28.0/86400.0, …)` in
    /// `astrodyn_verif_jeod::setups::earth_moon_clem` must yield identical
    /// TAI TJTs so the recipe can serve as the canonical entry point.
    #[test]
    fn clementine_1994_matches_run_clem_epoch() {
        let from_recipe = clementine_1994();
        let expected_tai_tjt = 9_412.0 + 28.0 / 86_400.0;
        let err = (from_recipe.tai_tjt_at_epoch - expected_tai_tjt).abs();
        assert!(
            err < 1e-12,
            "clementine_1994 tai_tjt = {}, expected {expected_tai_tjt}, err = {err}",
            from_recipe.tai_tjt_at_epoch
        );
        // Cross-check against the calendar form. UTC -> TAI adds 28 s,
        // which `at_utc` looks up from the default leap-second table.
        let from_utc = at_utc(1994, 3, 1, 0, 0, 0.0);
        assert_eq!(from_recipe.tai_tjt_at_epoch, from_utc.tai_tjt_at_epoch);
    }
}
