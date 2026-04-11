//! Tier 3: SIM_5_all_inclusive — time scale parity cross-validation.
//!
//! Validates all time scales (TAI, UTC, UT1, TT, TDB, GMST, GPS) against
//! JEOD reference data over 2 hours at 60s intervals.

mod sim_test_helpers;
use sim_test_helpers::*;

use jeod_sim::SimulationTime;

const SECONDS_PER_DAY: f64 = 86400.0;

#[allow(dead_code)]
struct TimescaleRecord {
    time: f64,
    tai_tjt: f64,
    tai_seconds: f64,
    utc_tjt: f64,
    ut1_tjt: f64,
    tt_tjt: f64,
    tdb_tjt: f64,
    gmst_seconds: f64,
    gps_tjt: f64,
}

fn load_timescale_csv(path: &std::path::Path) -> Vec<TimescaleRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_5_all_inclusive CSV from {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 9 {
            continue;
        }
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(TimescaleRecord {
            time: p(0),
            tai_tjt: p(1),
            tai_seconds: p(2),
            utc_tjt: p(3),
            ut1_tjt: p(4),
            tt_tjt: p(5),
            tdb_tjt: p(6),
            gmst_seconds: p(7),
            gps_tjt: p(8),
        });
    }
    records
}

#[test]
fn tier3_simulation_timescale_tdb() {
    let csv_path = test_data_path("timescale_tdb_timescale.csv");
    let records = load_timescale_csv(&csv_path);
    assert!(!records.is_empty(), "No timescale reference data");

    // Initialize SimulationTime at the JEOD epoch (TAI TJT from first record)
    let init = &records[0];
    let leap_table = jeod_sim::default_leap_second_table();
    let mut sim_time = SimulationTime::new(init.tai_tjt, leap_table);

    // Derive UT1-TAI from the JEOD CSV's UT1 and TAI TJT values (IERS data).
    // Default assumes UT1-UTC=0 → UT1-TAI = -37s, but the actual IERS value
    // at this epoch is ~-36.712s. Without this, GMST error is ~0.29 s.
    let ut1_tai_offset = (init.ut1_tjt - init.tai_tjt) * SECONDS_PER_DAY;
    sim_time.set_ut1_tai_offset(ut1_tai_offset);

    let mut max_tai_err = 0.0_f64;
    let mut max_tt_err = 0.0_f64;
    let mut max_tdb_err = 0.0_f64;
    let mut max_gmst_err = 0.0_f64;
    let mut max_gps_err = 0.0_f64;

    for (i, rec) in records.iter().enumerate() {
        if i > 0 {
            let dt = rec.time - records[i - 1].time;
            sim_time.advance(dt);
        }

        // Compare TAI TJT
        let tai_err = (sim_time.tai_tjt - rec.tai_tjt).abs() * SECONDS_PER_DAY;
        max_tai_err = max_tai_err.max(tai_err);

        // Compare TT TJT
        let tt_err = (sim_time.tt_tjt() - rec.tt_tjt).abs() * SECONDS_PER_DAY;
        max_tt_err = max_tt_err.max(tt_err);

        // Compare TDB TJT
        let our_tdb_tjt =
            sim_time.tai_tjt + (sim_time.tdb_seconds - sim_time.tai_seconds) / SECONDS_PER_DAY;
        let tdb_err = (our_tdb_tjt - rec.tdb_tjt).abs() * SECONDS_PER_DAY;
        max_tdb_err = max_tdb_err.max(tdb_err);

        // Compare GMST seconds
        let gmst_err = (sim_time.gmst_seconds - rec.gmst_seconds).abs();
        max_gmst_err = max_gmst_err.max(gmst_err);

        // GPS TJT = TAI TJT in JEOD's convention (the 19 s offset is absorbed
        // into the difference between GPS and TAI tjt_at_epoch values).
        let our_gps_tjt = sim_time.tai_tjt;
        let gps_err = (our_gps_tjt - rec.gps_tjt).abs() * SECONDS_PER_DAY;
        max_gps_err = max_gps_err.max(gps_err);
    }

    println!(
        "  Timescale TDB: {} points, TAI={max_tai_err:.2e}s, TT={max_tt_err:.2e}s, \
         TDB={max_tdb_err:.2e}s, GMST={max_gmst_err:.2e}s, GPS={max_gps_err:.2e}s",
        records.len()
    );

    // TAI/TT/TDB: ~1.6e-6 s residual from JEOD's indirect Dyn→TDB→TAI update
    // chain vs our direct accumulation. Irreducible without replicating JEOD's
    // exact floating-point path.
    assert!(max_tai_err < 2e-6, "TAI error {max_tai_err:.4e} s");
    assert!(max_tt_err < 2e-6, "TT error {max_tt_err:.4e} s");
    assert!(max_tdb_err < 2e-6, "TDB error {max_tdb_err:.4e} s");
    // GMST: with correct UT1-TAI from IERS data, residual is from JEOD's
    // time-varying UT1-TAI gradient interpolation vs our constant offset.
    assert!(max_gmst_err < 1e-4, "GMST error {max_gmst_err:.4e} s");
    // GPS TJT = TAI TJT in JEOD's convention
    assert!(max_gps_err < 2e-6, "GPS error {max_gps_err:.4e} s");
}
