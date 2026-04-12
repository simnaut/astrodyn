//! Tier 3: SIM_5_all_inclusive — time scale parity cross-validation.
//!
//! Validates TAI, TT, TDB, GMST, and GPS time scales against JEOD reference
//! data over 2 hours at 60s intervals, using `Simulation::step()` to advance
//! time through the full pipeline.

mod sim_test_helpers;
use sim_test_helpers::*;

use jeod_sim::{Simulation, SimulationTime};

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
        assert!(
            f.len() >= 9,
            "line {}: expected >=9 columns, got {}",
            i + 1,
            f.len()
        );
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

    let init = &records[0];
    let leap_table = jeod_sim::default_leap_second_table();
    let sim_time = SimulationTime::new(init.tai_tjt, leap_table);

    // Step size: records are at 60s intervals
    let dt = if records.len() > 1 {
        records[1].time - records[0].time
    } else {
        60.0
    };

    let mut sim = Simulation::new(sim_time, dt);
    // No bodies or sources needed — Simulation::step() still advances time.

    // Derive UT1-TAI from the JEOD CSV's UT1 and TAI TJT values (IERS data).
    let ut1_tai_offset = (init.ut1_tjt - init.tai_tjt) * SECONDS_PER_DAY;
    sim.time.set_ut1_tai_offset(ut1_tai_offset);

    let mut max_tai_err = 0.0_f64;
    let mut max_tt_err = 0.0_f64;
    let mut max_tdb_err = 0.0_f64;
    let mut max_gmst_err = 0.0_f64;
    let mut max_gps_err = 0.0_f64;

    for (i, rec) in records.iter().enumerate() {
        if i > 0 {
            sim.step();
        }

        let tai_err = (sim.time.tai_tjt - rec.tai_tjt).abs() * SECONDS_PER_DAY;
        max_tai_err = max_tai_err.max(tai_err);

        let tt_err = (sim.time.tt_tjt() - rec.tt_tjt).abs() * SECONDS_PER_DAY;
        max_tt_err = max_tt_err.max(tt_err);

        let our_tdb_tjt =
            sim.time.tai_tjt + (sim.time.tdb_seconds - sim.time.tai_seconds) / SECONDS_PER_DAY;
        let tdb_err = (our_tdb_tjt - rec.tdb_tjt).abs() * SECONDS_PER_DAY;
        max_tdb_err = max_tdb_err.max(tdb_err);

        let gmst_err = (sim.time.gmst_seconds - rec.gmst_seconds).abs();
        max_gmst_err = max_gmst_err.max(gmst_err);

        // GPS TJT = TAI TJT in JEOD's convention
        let our_gps_tjt = sim.time.tai_tjt;
        let gps_err = (our_gps_tjt - rec.gps_tjt).abs() * SECONDS_PER_DAY;
        max_gps_err = max_gps_err.max(gps_err);
    }

    println!(
        "  Timescale TDB: {} points, TAI={max_tai_err:.2e}s, TT={max_tt_err:.2e}s, \
         TDB={max_tdb_err:.2e}s, GMST={max_gmst_err:.2e}s, GPS={max_gps_err:.2e}s",
        records.len()
    );

    assert!(max_tai_err < 2e-6, "TAI error {max_tai_err:.4e} s");
    assert!(max_tt_err < 2e-6, "TT error {max_tt_err:.4e} s");
    assert!(max_tdb_err < 2e-6, "TDB error {max_tdb_err:.4e} s");
    assert!(max_gmst_err < 1e-4, "GMST error {max_gmst_err:.4e} s");
    assert!(max_gps_err < 2e-6, "GPS error {max_gps_err:.4e} s");
}
