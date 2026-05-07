//! Integration tests for the time scale network.
//!
//! Phase 5 tests: TAI↔UTC round-trip across leap second boundary,
//! GPS = TAI - 19s at multiple epochs, MET from configured epoch,
//! calendar↔JD round-trip, TimeManager full propagation.

use astrodyn_time::epoch::{jd_to_tjt, tjt_to_jd, SECONDS_PER_DAY};
use astrodyn_time::leap_second::default_leap_second_table;
use astrodyn_time::time_gps;
use astrodyn_time::time_met::MissionElapsedTime;
use astrodyn_time::time_utc::{calendar_to_tjt, tjt_to_calendar, CalendarDate};
use astrodyn_time::{TimeManager, TimeScaleId};

/// TAI→UTC round-trip across the 2016-12-31 23:59:60 leap second boundary.
///
/// On 2016-12-31, a leap second was inserted (TAI-UTC went from 36 to 37).
/// The boundary is at MJD 57754.0 = TJT 17754.0 (2017-01-01 00:00:00 UTC).
/// We test that our TAI→UTC→TAI round-trip is consistent on both sides.
#[test]
fn tai_utc_round_trip_across_leap_second_2017() {
    let table = default_leap_second_table();

    // Just before the 2017 leap second: 2016-12-31 23:59:59 UTC
    // UTC TJT = 17754.0 - 1/86400 = 17753.999988...
    let utc_tjt_before = 17754.0 - 1.0 / SECONDS_PER_DAY;
    let tai_tjt_before = table.utc_to_tai_tjt(utc_tjt_before);
    let utc_back_before = table.tai_to_utc_tjt(tai_tjt_before);
    assert!(
        (utc_back_before - utc_tjt_before).abs() < 1e-12,
        "Round trip before leap second: {} -> {} -> {}, err={}",
        utc_tjt_before,
        tai_tjt_before,
        utc_back_before,
        (utc_back_before - utc_tjt_before).abs()
    );

    // TAI-UTC before should be 36
    let tai_utc_before = table.tai_utc_at_utc_tjt(utc_tjt_before);
    assert_eq!(
        tai_utc_before, 36.0,
        "TAI-UTC before 2017 leap second should be 36"
    );

    // Just after the 2017 leap second: 2017-01-01 00:00:01 UTC
    let utc_tjt_after = 17754.0 + 1.0 / SECONDS_PER_DAY;
    let tai_tjt_after = table.utc_to_tai_tjt(utc_tjt_after);
    let utc_back_after = table.tai_to_utc_tjt(tai_tjt_after);
    assert!(
        (utc_back_after - utc_tjt_after).abs() < 1e-12,
        "Round trip after leap second: {} -> {} -> {}, err={}",
        utc_tjt_after,
        tai_tjt_after,
        utc_back_after,
        (utc_back_after - utc_tjt_after).abs()
    );

    // TAI-UTC after should be 37
    let tai_utc_after = table.tai_utc_at_utc_tjt(utc_tjt_after);
    assert_eq!(
        tai_utc_after, 37.0,
        "TAI-UTC after 2017 leap second should be 37"
    );
}

/// GPS = TAI - 19s at multiple epochs.
#[test]
fn gps_equals_tai_minus_19_at_multiple_epochs() {
    let test_tai_values = [0.0, 100.0, 86400.0, 1_000_000.0, -1000.0];
    for tai in test_tai_values {
        let gps = time_gps::tai_to_gps(tai);
        assert!(
            (gps - (tai - 19.0)).abs() < 1e-15,
            "GPS = TAI - 19 failed for TAI={}: GPS={}",
            tai,
            gps
        );
        // Round-trip
        let back = time_gps::gps_to_tai(gps);
        assert!(
            (back - tai).abs() < 1e-15,
            "GPS→TAI round trip failed for TAI={}",
            tai
        );
    }
}

/// MET = elapsed seconds from a configured epoch.
#[test]
fn met_elapsed_from_configured_epoch() {
    // Mission starts at TAI=5000s
    let mut met = MissionElapsedTime::new(5000.0);
    assert!(met.seconds.abs() < 1e-15, "MET at epoch should be 0");

    met.update(5000.0);
    assert!(met.seconds.abs() < 1e-15, "MET at epoch time should be 0");

    met.update(5060.0);
    assert!(
        (met.seconds - 60.0).abs() < 1e-15,
        "MET 60s after epoch: expected 60, got {}",
        met.seconds
    );

    met.update(10000.0);
    assert!(
        (met.seconds - 5000.0).abs() < 1e-15,
        "MET 5000s after epoch: expected 5000, got {}",
        met.seconds
    );
}

/// Calendar ↔ Julian Date round-trip for known dates.
#[test]
fn calendar_jd_round_trip() {
    let known_dates = [
        (CalendarDate::new(2000, 1, 1, 12, 0, 0.0), 2_451_545.0), // J2000
        (CalendarDate::new(1980, 1, 6, 0, 0, 0.0), 2_444_244.5),  // GPS epoch
        (CalendarDate::new(1972, 1, 1, 0, 0, 0.0), 2_441_317.5),  // First leap second
    ];

    for (cal, expected_jd) in &known_dates {
        let tjt = calendar_to_tjt(cal);
        let jd = tjt_to_jd(tjt);
        // These inputs have exact JD values; 1e-10 day ≈ 8.6 µs.
        assert!(
            (jd - expected_jd).abs() < 1e-10,
            "Calendar {:?} -> JD {}, expected {}",
            cal,
            jd,
            expected_jd
        );

        // Round-trip: JD -> TJT -> Calendar
        let tjt_back = jd_to_tjt(jd);
        assert!(
            (tjt_back - tjt).abs() < 1e-12,
            "JD->TJT round trip failed for {:?}",
            cal
        );

        let cal_back = tjt_to_calendar(tjt_back);
        assert_eq!(cal_back.year, cal.year, "Year mismatch for {:?}", cal);
        assert_eq!(cal_back.month, cal.month, "Month mismatch for {:?}", cal);
        assert_eq!(cal_back.day, cal.day, "Day mismatch for {:?}", cal);
        assert_eq!(cal_back.hour, cal.hour, "Hour mismatch for {:?}", cal);
        assert_eq!(cal_back.minute, cal.minute, "Minute mismatch for {:?}", cal);
        assert!(
            (cal_back.second - cal.second).abs() < 1e-6,
            "Second mismatch for {:?}: got {}",
            cal,
            cal_back.second
        );
    }
}

/// TimeManager: register all scales, advance TAI, verify all update correctly.
#[test]
fn time_manager_full_propagation() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());
    mgr.add_met(0.0); // MET epoch at simulation start
    mgr.add_ude(3600.0); // UDE epoch at TAI=3600s

    // Advance 2 hours
    let dt = 7200.0;
    mgr.advance(dt);

    // TAI
    assert!(
        (mgr.get_seconds(TimeScaleId::TAI) - dt).abs() < 1e-15,
        "TAI: expected {}, got {}",
        dt,
        mgr.get_seconds(TimeScaleId::TAI)
    );

    // TT = TAI + 32.184
    assert!(
        (mgr.get_seconds(TimeScaleId::TT) - (dt + 32.184)).abs() < 1e-10,
        "TT: expected {}, got {}",
        dt + 32.184,
        mgr.get_seconds(TimeScaleId::TT)
    );

    // GPS = TAI - 19
    assert!(
        (mgr.get_seconds(TimeScaleId::GPS) - (dt - 19.0)).abs() < 1e-15,
        "GPS: expected {}, got {}",
        dt - 19.0,
        mgr.get_seconds(TimeScaleId::GPS)
    );

    // DYN = TAI at scale_factor=1
    assert!(
        (mgr.get_seconds(TimeScaleId::DYN) - dt).abs() < 1e-15,
        "DYN: expected {}, got {}",
        dt,
        mgr.get_seconds(TimeScaleId::DYN)
    );

    // MET = elapsed from epoch=0 = TAI
    assert!(
        (mgr.get_seconds(TimeScaleId::MET) - dt).abs() < 1e-15,
        "MET: expected {}, got {}",
        dt,
        mgr.get_seconds(TimeScaleId::MET)
    );

    // UDE = TAI - 3600 = 7200 - 3600 = 3600
    let ude_s = mgr.get_ude_seconds(0).expect("UDE 0 should be registered");
    assert!(
        (ude_s - 3600.0).abs() < 1e-15,
        "UDE: expected 3600, got {}",
        ude_s
    );

    // TDB ≈ TT (within ~2ms)
    let tdb_tt_diff = (mgr.get_seconds(TimeScaleId::TDB) - mgr.get_seconds(TimeScaleId::TT)).abs();
    assert!(
        tdb_tt_diff < 0.002,
        "TDB-TT difference should be < 2ms, got {}",
        tdb_tt_diff
    );

    // GMST should have advanced from its initial value
    let gmst_initial = {
        let fresh = TimeManager::at_j2000(default_leap_second_table());
        fresh.get_seconds(TimeScaleId::GMST)
    };
    let gmst_delta = mgr.get_seconds(TimeScaleId::GMST) - gmst_initial;
    // 2 hours of sidereal time ≈ 7200 * (366.25/365.25) ≈ 7219.7 sidereal seconds
    assert!(
        (gmst_delta - 7219.7).abs() < 1.0,
        "GMST should advance ~7219.7 sidereal seconds in 2 hours, delta={}",
        gmst_delta
    );

    // Verify simtime tracks independently
    assert!((mgr.simtime - dt).abs() < 1e-15, "simtime should be {}", dt);
}

/// TimeManager: UTC seconds track correctly through the simulation.
#[test]
fn time_manager_utc_advances() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());
    // At J2000, TAI-UTC = 32s. Over a span with no leap seconds,
    // UTC seconds ≈ TAI seconds (both count elapsed time).
    mgr.advance(3600.0);
    // UTC elapsed should approximately equal TAI elapsed (no leap second
    // boundary crossed within 1 hour of J2000)
    // UTC goes through TJT day arithmetic, introducing ~5e-8 s numerical noise.
    assert!(
        (mgr.utc_seconds - 3600.0).abs() < 1e-6,
        "UTC elapsed: expected ~3600, got {}",
        mgr.utc_seconds
    );
}

/// TimeManager with DYN scale factor reversal.
#[test]
fn time_manager_dyn_reversal_round_trip() {
    let mut mgr = TimeManager::at_j2000(default_leap_second_table());

    // Forward 1 hour
    mgr.advance(3600.0);
    let tai_fwd = mgr.tai_seconds;
    let gps_fwd = mgr.gps_seconds;

    // Reverse time
    mgr.set_scale_factor(-1.0);
    mgr.advance(3600.0);

    assert!(
        mgr.tai_seconds.abs() < 1e-15,
        "TAI should return to 0 after reversal, got {}",
        mgr.tai_seconds
    );
    assert!(
        (mgr.gps_seconds - (-19.0)).abs() < 1e-15,
        "GPS should return to -19 after reversal, got {}",
        mgr.gps_seconds
    );

    let _ = (tai_fwd, gps_fwd);
}
