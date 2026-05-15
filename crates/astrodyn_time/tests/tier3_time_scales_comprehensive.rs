//! Comprehensive time scale verification tests.
//!
//! These tests exercise the full TimeManager across extended propagation,
//! verifying that all time scale relationships remain correct over long
//! durations and at critical boundaries (leap seconds, GPS week rollovers,
//! MET holds, DYN scale factor changes, UDE custom epochs).

use astrodyn_time::epoch::{mjd_to_tjt, SECONDS_PER_DAY, TAI_TT_OFFSET};
use astrodyn_time::leap_second::default_leap_second_table;
use astrodyn_time::time_converter_tai_tdb;
use astrodyn_time::time_gps;
use astrodyn_time::time_utc::{calendar_to_tjt, tjt_to_calendar, CalendarDate};
use astrodyn_time::{TimeManager, TimeScaleId};

/// Verify all time scales maintain correct relationships over 24-hour propagation.
///
/// Advances the TimeManager by 86400s in 1s steps, checking at each step:
///   TT = TAI + 32.184s (exact)
///   GPS = TAI - 19s (exact)
///   |TDB - TT| < 2ms (periodic correction bounded)
///   UTC elapsed ~= TAI elapsed (no leap second crossed near J2000)
#[test]
fn tier3_time_all_scales_24h_propagation() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());

    let dt = 1.0; // 1-second steps
    let total_steps = 86400; // 24 hours

    for step in 1..=total_steps {
        mgr.advance(dt);
        let tai = mgr.get_seconds(TimeScaleId::TAI);
        let expected_tai = step as f64;

        // TAI should advance exactly
        assert!(
            (tai - expected_tai).abs() < 1e-10,
            "Step {}: TAI = {}, expected {}",
            step,
            tai,
            expected_tai
        );

        // TT = TAI + 32.184s (exact by definition)
        let tt = mgr.get_seconds(TimeScaleId::TT);
        assert!(
            (tt - (tai + TAI_TT_OFFSET)).abs() < 1e-10,
            "Step {}: TT - TAI = {}, expected {}",
            step,
            tt - tai,
            TAI_TT_OFFSET
        );

        // GPS = TAI - 19s (exact by definition)
        let gps = mgr.get_seconds(TimeScaleId::GPS);
        assert!(
            (gps - (tai - 19.0)).abs() < 1e-12,
            "Step {}: GPS = {}, expected {}",
            step,
            gps,
            tai - 19.0
        );

        // |TDB - TT| < 2ms (Fairhead & Bretagnon periodic correction)
        let tdb = mgr.get_seconds(TimeScaleId::TDB);
        let tdb_tt_diff = (tdb - tt).abs();
        assert!(
            tdb_tt_diff < 0.002,
            "Step {}: |TDB - TT| = {} s, exceeds 2 ms bound",
            step,
            tdb_tt_diff
        );

        // UTC elapsed should approximately equal TAI elapsed
        // (no leap second boundary crossed within 24h of J2000)
        let utc = mgr.utc_seconds;
        assert!(
            (utc - expected_tai).abs() < 1e-4,
            "Step {}: UTC elapsed = {}, expected ~{}",
            step,
            utc,
            expected_tai
        );
    }

    // Final check: DYN = TAI at default scale factor
    let dyn_final = mgr.get_seconds(TimeScaleId::DYN);
    assert!(
        (dyn_final - 86400.0).abs() < 1e-10,
        "DYN after 24h: {}, expected 86400",
        dyn_final
    );
}

/// Test UTC behavior around multiple leap second insertions.
///
/// Verifies that the TAI-UTC offset changes by exactly 1s at each
/// known leap second boundary. Tests three boundaries:
///   2012-07-01 (TAI-UTC: 34 -> 35)
///   2015-07-01 (TAI-UTC: 35 -> 36)
///   2017-01-01 (TAI-UTC: 36 -> 37)
#[test]
fn tier3_time_utc_leap_second_boundaries() {
    // The 2017 boundary is the last entry; probing one second past it
    // (`utc_tjt_after = 17754 + 1/86400`) is strictly OOR, so opt in to
    // JEOD-faithful clamp behavior (#485 H2).
    let table = default_leap_second_table().with_clamp_out_of_range(true);

    // Test boundaries: (MJD of leap second, TAI-UTC before, TAI-UTC after)
    let boundaries: &[(f64, f64, f64)] = &[
        (56109.0, 34.0, 35.0), // 2012-07-01
        (57204.0, 35.0, 36.0), // 2015-07-01
        (57754.0, 36.0, 37.0), // 2017-01-01
    ];

    for &(mjd, before_offset, after_offset) in boundaries {
        let utc_tjt_boundary = mjd_to_tjt(mjd);

        // Just before the boundary (1 second before in UTC TJT)
        let utc_tjt_before = utc_tjt_boundary - 1.0 / SECONDS_PER_DAY;
        let tai_utc_before = table.tai_utc_at_utc_tjt(utc_tjt_before);
        assert_eq!(
            tai_utc_before, before_offset,
            "Before MJD {}: TAI-UTC = {}, expected {}",
            mjd, tai_utc_before, before_offset
        );

        // At/after the boundary
        let utc_tjt_after = utc_tjt_boundary + 1.0 / SECONDS_PER_DAY;
        let tai_utc_after = table.tai_utc_at_utc_tjt(utc_tjt_after);
        assert_eq!(
            tai_utc_after, after_offset,
            "After MJD {}: TAI-UTC = {}, expected {}",
            mjd, tai_utc_after, after_offset
        );

        // The change should be exactly 1 second
        assert_eq!(
            after_offset - before_offset,
            1.0,
            "Leap second at MJD {} should change TAI-UTC by exactly 1s",
            mjd
        );

        // Round-trip: TAI -> UTC -> TAI should be consistent on both sides
        let tai_tjt_before = table.utc_to_tai_tjt(utc_tjt_before);
        let utc_back = table.tai_to_utc_tjt(tai_tjt_before);
        assert!(
            (utc_back - utc_tjt_before).abs() < 1e-12,
            "Round-trip before MJD {}: err = {}",
            mjd,
            (utc_back - utc_tjt_before).abs()
        );

        let tai_tjt_after = table.utc_to_tai_tjt(utc_tjt_after);
        let utc_back_after = table.tai_to_utc_tjt(tai_tjt_after);
        assert!(
            (utc_back_after - utc_tjt_after).abs() < 1e-12,
            "Round-trip after MJD {}: err = {}",
            mjd,
            (utc_back_after - utc_tjt_after).abs()
        );
    }
}

/// Verify GPS week number computation and the 1024-week rollover.
///
/// Tests GPS week/day decomposition at known dates and verifies
/// the 10-bit week counter rollover at 7168 days.
#[test]
fn tier3_time_gps_week_rollover() {
    // GPS epoch: 1980-01-06 00:00:00 UTC
    // At 0 days: week=0, day=0
    let c0 = time_gps::gps_components(0.0);
    assert_eq!(c0.week, 0, "GPS epoch: week should be 0");
    assert_eq!(c0.day_of_week, 0, "GPS epoch: day_of_week should be 0");
    assert_eq!(
        c0.rollover_count, 0,
        "GPS epoch: rollover_count should be 0"
    );

    // After exactly 1 week (7 days)
    let c1 = time_gps::gps_components(7.0);
    assert_eq!(c1.week, 1, "After 7 days: week should be 1");
    assert_eq!(c1.day_of_week, 0, "After 7 days: day_of_week should be 0");

    // First rollover at 1024 weeks = 7168 days
    // This corresponds to August 22, 1999 (Sunday, since 7168 is a multiple of 7
    // from the 1980-01-06 Sunday epoch)
    let c_rollover = time_gps::gps_components(7168.0);
    assert_eq!(
        c_rollover.rollover_count, 1,
        "At 7168 days: rollover_count should be 1"
    );
    assert_eq!(
        c_rollover.week, 0,
        "At 7168 days: week should be 0 (rolled over)"
    );
    assert_eq!(
        c_rollover.day_of_week, 0,
        "At 7168 days: day_of_week should be 0"
    );

    // Just before the rollover: day 7167 = week 1023, day 6
    let c_pre_rollover = time_gps::gps_components(7167.0);
    assert_eq!(
        c_pre_rollover.rollover_count, 0,
        "At 7167 days: still in first rollover period"
    );
    assert_eq!(
        c_pre_rollover.week, 1023,
        "At 7167 days: week should be 1023"
    );
    assert_eq!(
        c_pre_rollover.day_of_week, 6,
        "At 7167 days: day_of_week should be 6"
    );

    // Second rollover at 2 * 7168 = 14336 days (April 7, 2019, Sunday)
    let c_second_rollover = time_gps::gps_components(14336.0);
    assert_eq!(
        c_second_rollover.rollover_count, 2,
        "At 14336 days: rollover_count should be 2"
    );
    assert_eq!(c_second_rollover.week, 0, "At 14336 days: week should be 0");

    // Verify 13-bit rollover (8192 weeks = 57344 days)
    let c_13bit = time_gps::gps_components(57344.0);
    assert_eq!(
        c_13bit.rollover_count_13_bit, 1,
        "At 57344 days: 13-bit rollover_count should be 1"
    );
    assert_eq!(
        c_13bit.week_13_bit, 0,
        "At 57344 days: 13-bit week should be 0"
    );

    // Verify GPS seconds decomposition: halfway through a day
    let c_half = time_gps::gps_components(10.5); // 10 days + 12 hours
    assert_eq!(c_half.day_of_week, 3, "10.5 days: day_of_week should be 3");
    assert!(
        (c_half.seconds_of_day - 43200.0).abs() < 1e-6,
        "10.5 days: seconds_of_day should be 43200"
    );
    assert_eq!(c_half.week, 1, "10.5 days: week should be 1");
}

/// Verify MET hold and resume behavior.
///
/// Creates MET, advances to 100s, holds for 50s, releases, advances 50s more.
/// MET should freeze during hold and resume correctly afterward.
#[test]
fn tier3_time_met_hold_resume() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());
    mgr.add_met(0.0); // MET epoch at TAI=0

    // Advance 100s -- MET should be 100
    for _ in 0..100 {
        mgr.advance(1.0);
    }
    let met_100 = mgr.get_seconds(TimeScaleId::MET);
    assert!(
        (met_100 - 100.0).abs() < 1e-10,
        "After 100s: MET = {}, expected 100",
        met_100
    );

    // Activate hold
    mgr.met.as_mut().unwrap().hold = true;

    // Advance 50s while held -- MET should stay at 100
    for _ in 0..50 {
        mgr.advance(1.0);
    }
    let met_held = mgr.get_seconds(TimeScaleId::MET);
    assert!(
        (met_held - 100.0).abs() < 1e-10,
        "During hold: MET = {}, expected 100 (frozen)",
        met_held
    );

    // TAI should have advanced to 150
    assert!(
        (mgr.get_seconds(TimeScaleId::TAI) - 150.0).abs() < 1e-10,
        "TAI should be 150 during MET hold"
    );

    // Release hold
    mgr.met.as_mut().unwrap().hold = false;

    // Advance 50s more -- MET should resume from 100 and reach 150
    for _ in 0..50 {
        mgr.advance(1.0);
    }
    let met_after = mgr.get_seconds(TimeScaleId::MET);
    assert!(
        (met_after - 150.0).abs() < 1e-10,
        "After hold release + 50s: MET = {}, expected 150",
        met_after
    );

    // TAI should be at 200
    assert!(
        (mgr.get_seconds(TimeScaleId::TAI) - 200.0).abs() < 1e-10,
        "TAI should be 200 after hold release"
    );
}

/// Verify DYN scale factor effects.
///
/// DYN with scale_factor = 2.0: advance simtime by 100s -> TAI and DYN advance by 200s.
/// DYN with scale_factor = 0.5: advance simtime by 100s -> TAI and DYN advance by 50s.
#[test]
fn tier3_time_dyn_scale_factor() {
    // Test 2x speed
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());
    mgr.set_scale_factor(2.0);
    mgr.advance(100.0);

    let tai = mgr.get_seconds(TimeScaleId::TAI);
    let dyn_s = mgr.get_seconds(TimeScaleId::DYN);
    // At 2x scale, both TAI and DYN advance by simtime * scale_factor = 200s
    assert!(
        (tai - 200.0).abs() < 1e-10,
        "At 2x: TAI = {}, expected 200",
        tai
    );
    assert!(
        (dyn_s - 200.0).abs() < 1e-10,
        "At 2x: DYN = {}, expected 200",
        dyn_s
    );
    // Simtime should be 100 (raw)
    assert!(
        (mgr.simtime - 100.0).abs() < 1e-10,
        "At 2x: simtime = {}, expected 100",
        mgr.simtime
    );

    // Test 0.5x speed
    let mut mgr2 = TimeManager::at_j2000(default_leap_second_table());
    mgr2.set_scale_factor(0.5);
    mgr2.advance(100.0);

    let tai2 = mgr2.get_seconds(TimeScaleId::TAI);
    let dyn_s2 = mgr2.get_seconds(TimeScaleId::DYN);
    assert!(
        (tai2 - 50.0).abs() < 1e-10,
        "At 0.5x: TAI = {}, expected 50",
        tai2
    );
    assert!(
        (dyn_s2 - 50.0).abs() < 1e-10,
        "At 0.5x: DYN = {}, expected 50",
        dyn_s2
    );
    assert!(
        (mgr2.simtime - 100.0).abs() < 1e-10,
        "At 0.5x: simtime = {}, expected 100",
        mgr2.simtime
    );

    // Test mid-sim scale factor change
    let mut mgr3 = TimeManager::at_j2000(default_leap_second_table());
    mgr3.advance(100.0); // 100s at 1x -> TAI=100
    mgr3.set_scale_factor(3.0);
    mgr3.advance(50.0); // 50s at 3x -> TAI += 150 -> TAI=250

    let tai3 = mgr3.get_seconds(TimeScaleId::TAI);
    assert!(
        (tai3 - 250.0).abs() < 1e-10,
        "Mid-sim scale change: TAI = {}, expected 250",
        tai3
    );
}

/// Verify UDE with custom epoch.
///
/// UDE epoch at TAI=1000s: at TAI=1000 UDE=0, at TAI=2000 UDE=1000.
/// Also tests clock decomposition.
#[test]
fn tier3_time_ude_custom_epoch() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());
    let idx = mgr.add_ude(1000.0); // UDE epoch at TAI=1000s

    // Before epoch: UDE should be negative
    mgr.advance(500.0);
    let ude_500 = mgr.get_ude_seconds(idx).unwrap();
    assert!(
        (ude_500 - (-500.0)).abs() < 1e-10,
        "Before UDE epoch: UDE = {}, expected -500",
        ude_500
    );

    // At epoch: UDE should be 0
    mgr.advance(500.0); // TAI now 1000
    let ude_0 = mgr.get_ude_seconds(idx).unwrap();
    assert!(
        ude_0.abs() < 1e-10,
        "At UDE epoch: UDE = {}, expected 0",
        ude_0
    );

    // After epoch: UDE should count elapsed seconds
    mgr.advance(1000.0); // TAI now 2000
    let ude_1000 = mgr.get_ude_seconds(idx).unwrap();
    assert!(
        (ude_1000 - 1000.0).abs() < 1e-10,
        "After UDE epoch: UDE = {}, expected 1000",
        ude_1000
    );

    // Verify clock decomposition: 1000s = 0d 0h 16m 40s
    let ude = mgr.get_ude(idx);
    assert_eq!(ude.clock_day, 0, "UDE clock_day should be 0");
    assert_eq!(ude.clock_hour, 0, "UDE clock_hour should be 0");
    assert_eq!(ude.clock_minute, 16, "UDE clock_minute should be 16");
    assert!(
        (ude.clock_second - 40.0).abs() < 1e-6,
        "UDE clock_second = {}, expected 40",
        ude.clock_second
    );

    // Test multiple UDEs with different epochs
    let idx2 = mgr.add_ude(500.0); // UDE2 epoch at TAI=500
                                   // TAI is now 2000, so UDE2 = 2000 - 500 = 1500
    let ude2 = mgr.get_ude_seconds(idx2).unwrap();
    assert!(
        (ude2 - 1500.0).abs() < 1e-10,
        "Second UDE: UDE = {}, expected 1500",
        ude2
    );
}

/// Test calendar-to-TJT round-trip for an extended set of dates.
///
/// Includes edge cases: leap year Feb 29, non-leap-year century,
/// pre-J2000, far future, end-of-year boundary.
#[test]
fn tier3_time_calendar_roundtrip_extended() {
    let test_dates = [
        CalendarDate::new(1970, 1, 1, 0, 0, 0.0),       // Unix epoch
        CalendarDate::new(2000, 1, 1, 12, 0, 0.0),      // J2000
        CalendarDate::new(2024, 2, 29, 0, 0, 0.0),      // Leap year
        CalendarDate::new(2100, 3, 1, 0, 0, 0.0),       // Not a leap year (divisible by 100)
        CalendarDate::new(1999, 12, 31, 23, 59, 59.0),  // End of millennium
        CalendarDate::new(2000, 3, 1, 0, 0, 0.0),       // 2000 IS a leap year (div by 400)
        CalendarDate::new(1980, 1, 6, 0, 0, 0.0),       // GPS epoch
        CalendarDate::new(2023, 6, 15, 14, 30, 45.123), // Arbitrary mid-year
        CalendarDate::new(1972, 1, 1, 0, 0, 0.0),       // First leap second year
        CalendarDate::new(2050, 7, 4, 12, 0, 0.0),      // Future date
    ];

    for cal in &test_dates {
        let tjt = calendar_to_tjt(cal);
        let back = tjt_to_calendar(tjt);

        assert_eq!(
            back.year, cal.year,
            "Year mismatch for {:?}: got {}",
            cal, back.year
        );
        assert_eq!(
            back.month, cal.month,
            "Month mismatch for {:?}: got {}",
            cal, back.month
        );
        assert_eq!(
            back.day, cal.day,
            "Day mismatch for {:?}: got {}",
            cal, back.day
        );
        assert_eq!(
            back.hour, cal.hour,
            "Hour mismatch for {:?}: got {}",
            cal, back.hour
        );
        assert_eq!(
            back.minute, cal.minute,
            "Minute mismatch for {:?}: got {}",
            cal, back.minute
        );
        assert!(
            (back.second - cal.second).abs() < 1e-6,
            "Second mismatch for {:?}: got {}, expected {}",
            cal,
            back.second,
            cal.second
        );
    }
}

/// Verify TDB-TT relationship over 1 year.
///
/// The TDB-TT offset is a small periodic correction (Fairhead & Bretagnon).
/// Over an entire year, the maximum |TDB - TT| should remain under 2 ms.
#[test]
fn tier3_time_tt_tdb_relationship() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());

    let mut max_diff: f64 = 0.0;
    let step = 3600.0; // 1 hour steps
    let total_steps = (365.25 * 24.0) as usize; // ~1 year

    for _ in 0..total_steps {
        mgr.advance(step);
        let tt = mgr.get_seconds(TimeScaleId::TT);
        let tdb = mgr.get_seconds(TimeScaleId::TDB);
        let diff = (tdb - tt).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // The maximum TDB-TT offset should be bounded by ~1.658 ms
    // (the amplitude of the primary sinusoidal term) plus a tiny
    // contribution from the second harmonic (~0.014 ms).
    assert!(
        max_diff < 0.002,
        "Max |TDB - TT| over 1 year: {} s, exceeds 2 ms bound",
        max_diff
    );

    // The maximum should be at least ~1 ms (not degenerate)
    assert!(
        max_diff > 0.001,
        "Max |TDB - TT| over 1 year: {} s, suspiciously small (expected ~1.66 ms)",
        max_diff
    );

    // Verify the TDB-TT offset function directly at J2000 (fresh manager, before propagation)
    let mgr_j2000 = TimeManager::at_j2000(default_leap_second_table());
    let offset_j2000 = time_converter_tai_tdb::tdb_tt_offset(mgr_j2000.tai_tjt);
    assert!(
        offset_j2000.abs() < 0.002,
        "TDB-TT at J2000: {} s",
        offset_j2000
    );
}

/// Verify GMST increases monotonically with UT1 over 24 hours.
///
/// Over 24 hours of UT1, GMST should advance by approximately
/// 86636.56s (the full sidereal-equivalent delta — 24h plus the ~236.56s
/// sidereal excess over a solar day).
#[test]
fn tier3_time_gmst_increases_with_ut1() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());
    let gmst_initial = mgr.get_seconds(TimeScaleId::GMST);

    let step = 60.0; // 1-minute steps
    let total_steps = 1440; // 24 hours
    let mut prev_gmst = gmst_initial;

    for i in 1..=total_steps {
        mgr.advance(step);
        let gmst = mgr.get_seconds(TimeScaleId::GMST);

        // GMST should increase monotonically (accumulated, not wrapped)
        assert!(
            gmst > prev_gmst,
            "Step {}: GMST decreased from {} to {}",
            i,
            prev_gmst,
            gmst
        );
        prev_gmst = gmst;
    }

    let gmst_final = mgr.get_seconds(TimeScaleId::GMST);
    let gmst_delta = gmst_final - gmst_initial;

    // Sidereal day = solar day * 366.25/365.25.
    // Over 24h of solar time, accumulated GMST "seconds" should increase by the
    // full sidereal-equivalent delta: 86400 * (366.25/365.25) ~= 86636.56 s.
    let expected_gmst_delta = 86400.0 * (366.25 / 365.25);
    assert!(
        (gmst_delta - expected_gmst_delta).abs() < 2.0,
        "GMST delta over 24h: {} s, expected ~{} s (full sidereal-equivalent delta)",
        gmst_delta,
        expected_gmst_delta
    );
}

/// Verify GPS time through TimeManager honors the GPS = TAI - 19 invariant.
///
/// GPS is offset from TAI by exactly 19 seconds (the TAI-UTC offset at the
/// GPS epoch, 1980-01-06). Since the offset is fixed (no leap seconds in GPS),
/// this relationship must hold exactly at every propagation step.
#[test]
fn tier3_time_gps_through_manager() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());

    // Check at t=0 as well as throughout propagation.
    let gps0 = mgr.get_seconds(TimeScaleId::GPS);
    let tai0 = mgr.get_seconds(TimeScaleId::TAI);
    assert!(
        (gps0 - (tai0 - 19.0)).abs() < 1e-12,
        "At t=0: GPS = {}, TAI - 19 = {}",
        gps0,
        tai0 - 19.0
    );

    for _ in 0..3600 {
        mgr.advance(10.0); // 10s steps for 10 hours
        let gps = mgr.get_seconds(TimeScaleId::GPS);
        let tai = mgr.get_seconds(TimeScaleId::TAI);
        assert!(
            (gps - (tai - 19.0)).abs() < 1e-12,
            "GPS != TAI - 19 at TAI={}: GPS = {}, expected {}",
            tai,
            gps,
            tai - 19.0
        );
    }
}

/// Verify TimeManager TDB Julian Date at J2000 and after propagation.
///
/// At J2000, TDB JD should be close to 2451545.0.
/// After 365.25 days, TDB JD should be close to 2451545.0 + 365.25.
#[test]
fn tier3_time_tdb_julian_date_propagation() {
    let mgr0 = TimeManager::at_j2000(default_leap_second_table());
    let jd0 = mgr0.tdb_julian_date();
    // At J2000, TDB differs from TT by the bounded periodic correction
    // (~ms, i.e., ~2e-8 days), so the JD should be within a very tight
    // tolerance of the canonical 2451545.0. This catches unit/offset
    // regressions that a loose multi-second bound would miss.
    assert!(
        (jd0 - 2_451_545.0).abs() < 1.0e-7,
        "TDB JD at J2000: {}, expected ~2451545.0",
        jd0
    );

    // Propagate 1 year
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());
    let one_year_s = 365.25 * SECONDS_PER_DAY;
    mgr.advance(one_year_s);
    let jd1 = mgr.tdb_julian_date();
    // TDB JD should advance by approximately 365.25 days. The delta is
    // deterministic: TAI advances by exactly 365.25 days, and the TDB-TT
    // periodic correction contributes at most ~2 * 1.67 ms ~= 4e-8 days of
    // endpoint-dependent drift, so the tolerance can be very tight.
    assert!(
        (jd1 - jd0 - 365.25).abs() < 1.0e-7,
        "TDB JD after 1 year: delta = {}, expected ~365.25",
        jd1 - jd0
    );
}

/// Verify that all time scales reset correctly after forward-reverse cycle.
///
/// Advance 1 hour forward at 1x, then reverse for 1 hour at -1x.
/// All time scales should return to their initial values.
#[test]
fn tier3_time_forward_reverse_all_scales() {
    let initial = TimeManager::at_j2000(default_leap_second_table());
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());
    mgr.add_met(0.0);
    let ude_idx = mgr.add_ude(500.0);

    // Forward 1 hour
    mgr.advance(3600.0);

    // Verify scales have changed
    assert!(
        (mgr.tai_seconds - 3600.0).abs() < 1e-10,
        "TAI should be 3600 after forward"
    );

    // Reverse
    mgr.set_scale_factor(-1.0);
    mgr.advance(3600.0);

    // TAI should return to 0
    assert!(
        mgr.tai_seconds.abs() < 1e-10,
        "TAI should return to 0: got {}",
        mgr.tai_seconds
    );

    // TT should return to initial (32.184)
    let tt = mgr.get_seconds(TimeScaleId::TT);
    assert!(
        (tt - TAI_TT_OFFSET).abs() < 1e-10,
        "TT should return to 32.184: got {}",
        tt
    );

    // GPS should return to initial (-19)
    let gps = mgr.get_seconds(TimeScaleId::GPS);
    assert!(
        (gps - (-19.0)).abs() < 1e-10,
        "GPS should return to -19: got {}",
        gps
    );

    // GMST should return to initial
    let gmst_initial_s = initial.get_seconds(TimeScaleId::GMST);
    let gmst = mgr.get_seconds(TimeScaleId::GMST);
    assert!(
        (gmst - gmst_initial_s).abs() < 1e-6,
        "GMST should return to initial: got {}, expected {}",
        gmst,
        gmst_initial_s
    );

    // MET should return to 0 (epoch at TAI=0)
    let met = mgr.get_seconds(TimeScaleId::MET);
    assert!(met.abs() < 1e-10, "MET should return to 0: got {}", met);

    // UDE should return to -500 (epoch at TAI=500, current TAI=0)
    let ude = mgr.get_ude_seconds(ude_idx).unwrap();
    assert!(
        (ude - (-500.0)).abs() < 1e-10,
        "UDE should return to -500: got {}",
        ude
    );
}
