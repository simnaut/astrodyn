//! Tier 3: JEOD time verification SIMs cross-validation.
//!
//! Exercises the Rust `SimulationTime` against JEOD-generated reference
//! CSVs from the six canonical time verification sims:
//!
//!   SIM_1_dyn_only       — DynamicTime only (no TAI)
//!   SIM_2_dyn_plus_STD   — Dyn + TAI
//!   SIM_3_dyn_plus_UDE   — Dyn + UDE
//!   SIM_4_common_usage   — TAI + UTC + UT1 across a leap-second boundary
//!   SIM_5_all_inclusive  — all 10+ scales incl. MET (UDE_initialized run)
//!   SIM_6_extension      — TAI (+ user-defined scale we don't test)
//!
//! These are time-only validations: no trajectory, no bodies, only propagated
//! time scale values at each checkpoint against JEOD's Trick ASCII output.
//! Required CSV files are written by `trick/generate_references.sh` (Docker).
//! Missing CSVs cause the test to panic with the exact regeneration command.
//!
//! The SIM_5 `RUN_UTC_initialized_tdb` run is covered by
//! `tier3_sim_timescale.rs`; here we cover the complementary
//! `RUN_UDE_initialized` variant plus all the other SIM_1..6 sims.

use astrodyn_verif_jeod::tier3_csv::test_data_path;

use astrodyn::{default_eop_table, default_leap_second_table};
use astrodyn::{SimulationTime, TimeScaleId};

const SECONDS_PER_DAY: f64 = 86400.0;

/// A parsed row from one of the time verification CSVs.
///
/// Only fields that appear in that sim's snippet are populated. Missing
/// columns are represented explicitly as `None` so per-sim tests can
/// `expect(...)` required fields — this prevents silent no-ops if Trick's
/// DRAscii drops a column (the exact failure mode called out in
/// `generate_references.sh`). Only fields asserted by at least one test are
/// parsed; `*.seconds` columns we don't check (utc/ut1/tt/tdb/gps, metveh2)
/// remain in the CSV but are ignored here.
#[derive(Debug, Clone, Default)]
struct TimeRow {
    /// Sim elapsed time `sys.exec.out.time` — always required.
    time: f64,
    dyn_seconds: Option<f64>,
    tai_tjt: Option<f64>,
    tai_seconds: Option<f64>,
    utc_tjt: Option<f64>,
    ut1_tjt: Option<f64>,
    tt_tjt: Option<f64>,
    tdb_tjt: Option<f64>,
    gmst_seconds: Option<f64>,
    gps_tjt: Option<f64>,
    ude_seconds: Option<f64>,
    metveh1_seconds: Option<f64>,
}

/// Read a time-verif CSV where the header maps JEOD variable names to columns.
///
/// Each row is placed into a `TimeRow` by looking up which column index holds
/// which JEOD variable; absent columns are left as `None`. Per-sim tests
/// `expect(...)` the fields they actually validate, so a silently-dropped
/// column surfaces as a panic instead of a NaN-masked no-op.
fn load_time_csv(path: &std::path::Path) -> Vec<TimeRow> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read time verification CSV from {}: {e}\n\
             Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
             -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
            path.display()
        )
    });

    let mut lines = content.lines();
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("empty CSV {}", path.display()));
    let headers: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

    // Map header → field setter. Headers always start with the JEOD variable
    // name and may carry a trailing " {unit}" annotation.
    fn name_of(h: &str) -> &str {
        h.split(|c: char| c.is_whitespace() || c == '{')
            .next()
            .unwrap_or(h)
            .trim()
    }
    let col = |name: &str| -> Option<usize> { headers.iter().position(|h| name_of(h) == name) };

    // Resolve once so per-row parsing is cheap. Only fields asserted by at
    // least one test are parsed; unused `*.seconds` columns (utc/ut1/tt/tdb/
    // gps, metveh2) are skipped even when present in the CSV.
    let i_time = col("sys.exec.out.time").expect("missing time column");
    let i_dyn = col("jeod_time.time_manager.dyn_time.seconds")
        .or_else(|| col("jeod_time.manager.dyn_time.seconds"));
    let i_tai_tjt = col("jeod_time.time_tai.trunc_julian_time")
        .or_else(|| col("jeod_time.tai.trunc_julian_time"));
    let i_tai_s = col("jeod_time.time_tai.seconds").or_else(|| col("jeod_time.tai.seconds"));
    let i_utc_tjt = col("jeod_time.time_utc.trunc_julian_time")
        .or_else(|| col("jeod_time.utc.trunc_julian_time"));
    let i_ut1_tjt = col("jeod_time.time_ut1.trunc_julian_time")
        .or_else(|| col("jeod_time.ut1.trunc_julian_time"));
    let i_tt_tjt = col("jeod_time.tt.trunc_julian_time");
    let i_tdb_tjt = col("jeod_time.tdb.trunc_julian_time");
    let i_gmst_s = col("jeod_time.gmst.seconds");
    let i_gps_tjt = col("jeod_time.gps.trunc_julian_time");
    let i_ude_s = col("jeod_time.time_ude.seconds");
    let i_met1_s = col("jeod_time.metveh1.seconds");

    let mut rows = Vec::new();
    for (li, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let p = |idx: usize| -> f64 {
            f[idx]
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("line {} col {}: parse failed: {e}", li + 2, idx))
        };
        let row = TimeRow {
            time: p(i_time),
            dyn_seconds: i_dyn.map(p),
            tai_tjt: i_tai_tjt.map(p),
            tai_seconds: i_tai_s.map(p),
            utc_tjt: i_utc_tjt.map(p),
            ut1_tjt: i_ut1_tjt.map(p),
            tt_tjt: i_tt_tjt.map(p),
            tdb_tjt: i_tdb_tjt.map(p),
            gmst_seconds: i_gmst_s.map(p),
            gps_tjt: i_gps_tjt.map(p),
            ude_seconds: i_ude_s.map(p),
            metveh1_seconds: i_met1_s.map(p),
        };
        rows.push(row);
    }
    assert!(!rows.is_empty(), "no data rows in {}", path.display());
    rows
}

/// Best-effort initial TAI TJT for a CSV: use the `tai.trunc_julian_time` at
/// t=0 if logged, otherwise fall back to J2000. SIM_1/SIM_3 don't log TAI at
/// all — those sims don't have TAI in the first place.
fn initial_tai_tjt(first: &TimeRow) -> f64 {
    // SIM_1 / SIM_3 have no TAI — arbitrary anchor; everything is
    // relative to simulation epoch.
    first.tai_tjt.unwrap_or(astrodyn::J2000_TAI_TJT)
}

// ── SIM_1_dyn_only ──────────────────────────────────────────────────────────

/// SIM_1: DynamicTime-only sim. No TAI, no calendar. JEOD logs only
/// `time_manager.dyn_time.seconds`, which equals sim-time when
/// `scale_factor = 1`. We verify our `SimulationTime.get_seconds(DYN)` tracks it.
#[test]
fn tier3_time_v1_dyn_only() {
    let csv = test_data_path("time_v1_dyn_only_time_v1.csv");
    let rows = load_time_csv(&csv);

    let mut mgr = SimulationTime::new(initial_tai_tjt(&rows[0]), default_leap_second_table());
    let dt = if rows.len() > 1 {
        rows[1].time - rows[0].time
    } else {
        1.0
    };

    let mut max_dyn_err = 0.0_f64;
    for (i, rec) in rows.iter().enumerate() {
        if i > 0 {
            mgr.advance(dt);
        }
        let dyn_ref = rec
            .dyn_seconds
            .expect("SIM_1 CSV must log dyn_time.seconds");
        let e = (mgr.get_seconds(TimeScaleId::DYN) - dyn_ref).abs();
        max_dyn_err = max_dyn_err.max(e);
    }

    println!(
        "  time_v1 dyn_only: {} points, DYN={max_dyn_err:.2e}s",
        rows.len()
    );
    // DynamicTime is an exact integer-second track for dt=1 forward sim.
    assert!(max_dyn_err < 1e-12, "DYN error {max_dyn_err:.4e} s");
}

// ── SIM_2_dyn_plus_STD ──────────────────────────────────────────────────────

/// SIM_2 RUN_initialize_by_value: TAI initialized to TJT=10000 (early 1968,
/// pre-leap-second era). Verify TAI seconds and TAI TJT propagation. The
/// CSV logs `time_tai.trunc_julian_time`, `time_tai.seconds`, and
/// `time_manager.dyn_time.seconds`.
#[test]
fn tier3_time_v2_std() {
    let csv = test_data_path("time_v2_std_time_v2.csv");
    let rows = load_time_csv(&csv);
    let init = &rows[0];
    let init_tai_tjt = init.tai_tjt.expect("SIM_2 CSV must log TAI TJT");
    let init_tai_seconds = init.tai_seconds.expect("SIM_2 CSV must log TAI seconds");

    let mut mgr = SimulationTime::new(init_tai_tjt, default_leap_second_table());
    let dt = if rows.len() > 1 {
        rows[1].time - rows[0].time
    } else {
        1.0
    };

    // JEOD's `time_tai.seconds` is TAI seconds-since-TAI-epoch (offset from
    // TJT=0 in the absolute TAI time scale), so compare the _elapsed_ value
    // (rec.tai_seconds - init.tai_seconds) against ours.
    let mut max_tai_s_err = 0.0_f64;
    let mut max_tai_tjt_err = 0.0_f64;
    let mut max_dyn_err = 0.0_f64;

    for (i, rec) in rows.iter().enumerate() {
        if i > 0 {
            mgr.advance(dt);
        }
        let tai_tjt = rec.tai_tjt.expect("SIM_2 CSV must log TAI TJT");
        let tai_seconds = rec.tai_seconds.expect("SIM_2 CSV must log TAI seconds");
        let dyn_seconds = rec.dyn_seconds.expect("SIM_2 CSV must log DYN seconds");
        let elapsed = tai_seconds - init_tai_seconds;
        max_tai_s_err = max_tai_s_err.max((mgr.tai_seconds - elapsed).abs());
        max_tai_tjt_err = max_tai_tjt_err.max((mgr.tai_tjt - tai_tjt).abs() * SECONDS_PER_DAY);
        max_dyn_err = max_dyn_err.max((mgr.get_seconds(TimeScaleId::DYN) - dyn_seconds).abs());
    }

    println!(
        "  time_v2 std: {} points, TAI_s={max_tai_s_err:.2e}s, TAI_tjt={max_tai_tjt_err:.2e}s, \
         DYN={max_dyn_err:.2e}s",
        rows.len()
    );

    assert!(
        max_tai_s_err < 1e-9,
        "TAI seconds error {max_tai_s_err:.4e} s"
    );
    assert!(
        max_tai_tjt_err < 1e-6,
        "TAI TJT error {max_tai_tjt_err:.4e} s"
    );
    assert!(max_dyn_err < 1e-12, "DYN error {max_dyn_err:.4e} s");
}

// ── SIM_3_dyn_plus_UDE ──────────────────────────────────────────────────────

/// SIM_3 RUN_init_by_ude: UDE initialized via clock with `clock_second = -5`.
/// After initialization UDE starts at -5 s; DynamicTime starts at 0 and
/// the UDE epoch is at Dyn = +5 s. At each step: UDE = Dyn - epoch_in_parent.
#[test]
fn tier3_time_v3_ude() {
    let csv = test_data_path("time_v3_ude_time_v3.csv");
    let rows = load_time_csv(&csv);
    let init = &rows[0];
    let init_ude = init.ude_seconds.expect("SIM_3 CSV must log UDE seconds");
    let init_dyn = init.dyn_seconds.expect("SIM_3 CSV must log DYN seconds");

    let mut mgr = SimulationTime::new(initial_tai_tjt(init), default_leap_second_table());
    // JEOD's Dyn time scale is the sim epoch. UDE = Dyn - epoch_in_parent.
    // At t=0 JEOD reports UDE = init.ude_seconds and Dyn = init.dyn_seconds,
    // so epoch_in_parent = init.dyn_seconds - init.ude_seconds.
    let epoch_in_parent = init_dyn - init_ude;
    // Our SimulationTime::add_ude takes epoch_in_parent as TAI seconds (our DYN
    // is mirrored to TAI for scale_factor=1), so this matches directly.
    let idx = mgr.add_ude(epoch_in_parent);

    let dt = if rows.len() > 1 {
        rows[1].time - rows[0].time
    } else {
        1.0
    };

    let mut max_ude_err = 0.0_f64;
    let mut max_dyn_err = 0.0_f64;
    for (i, rec) in rows.iter().enumerate() {
        if i > 0 {
            mgr.advance(dt);
        }
        let ude_ref = rec.ude_seconds.expect("SIM_3 CSV must log UDE seconds");
        let dyn_ref = rec.dyn_seconds.expect("SIM_3 CSV must log DYN seconds");
        let our_ude = mgr.get_ude_seconds(idx).expect("UDE registered");
        max_ude_err = max_ude_err.max((our_ude - ude_ref).abs());
        max_dyn_err = max_dyn_err.max((mgr.get_seconds(TimeScaleId::DYN) - dyn_ref).abs());
    }

    println!(
        "  time_v3 ude: {} points, UDE={max_ude_err:.2e}s, DYN={max_dyn_err:.2e}s",
        rows.len()
    );

    assert!(max_ude_err < 1e-10, "UDE error {max_ude_err:.4e} s");
    assert!(max_dyn_err < 1e-12, "DYN error {max_dyn_err:.4e} s");
}

// ── SIM_4_common_usage ──────────────────────────────────────────────────────

/// SIM_4 RUN_JEOD2x: TAI + UTC + UT1 initialized at 1998-12-31 00:00 UTC,
/// sampled at 60 s cadence through t = 86460 s, crossing the 1999-01-01
/// leap-second boundary. Verifies that our leap-second-aware UTC stays
/// consistent with JEOD's and that TAI, UT1 stay exact.
#[test]
fn tier3_time_v4_common() {
    let csv = test_data_path("time_v4_common_time_v4.csv");
    let rows = load_time_csv(&csv);
    let init = &rows[0];
    let init_tai_tjt = init.tai_tjt.expect("SIM_4 CSV must log TAI TJT");
    let init_ut1_tjt = init.ut1_tjt.expect("SIM_4 CSV must log UT1 TJT");
    let init_tai_seconds = init.tai_seconds.expect("SIM_4 CSV must log TAI seconds");

    // Wire the IERS EOP table so UT1-TAI is interpolated linearly
    // between adjacent daily samples per JEOD's
    // `time_converter_tai_ut1.cc::convert_a_to_b`. The CSV's t=0
    // UT1-TJT is used purely as a sanity check on the table value at
    // the epoch; we do not feed it back as a constant override.
    let mut mgr = SimulationTime::new(init_tai_tjt, default_leap_second_table())
        .with_eop_table(default_eop_table());
    let init_eop_offset = mgr.ut1_tai_offset;
    let csv_offset = (init_ut1_tjt - init_tai_tjt) * SECONDS_PER_DAY;
    assert!(
        (init_eop_offset - csv_offset).abs() < 1e-3,
        "EOP table at t=0 ({init_eop_offset} s) disagrees with JEOD CSV \
         ({csv_offset} s); check the EOP fixture against the JEOD source"
    );

    let dt = if rows.len() > 1 {
        rows[1].time - rows[0].time
    } else {
        60.0
    };

    let mut max_tai_tjt_err = 0.0_f64;
    let mut max_utc_tjt_err = 0.0_f64;
    let mut max_ut1_tjt_err = 0.0_f64;
    let mut max_tai_s_err = 0.0_f64;

    for (i, rec) in rows.iter().enumerate() {
        if i > 0 {
            mgr.advance(dt);
        }
        let tai_tjt = rec.tai_tjt.expect("SIM_4 CSV must log TAI TJT");
        let utc_tjt = rec.utc_tjt.expect("SIM_4 CSV must log UTC TJT");
        let ut1_tjt = rec.ut1_tjt.expect("SIM_4 CSV must log UT1 TJT");
        let tai_seconds = rec.tai_seconds.expect("SIM_4 CSV must log TAI seconds");
        max_tai_tjt_err = max_tai_tjt_err.max((mgr.tai_tjt - tai_tjt).abs() * SECONDS_PER_DAY);
        let our_utc_tjt = mgr.leap_second_table.tai_to_utc_tjt(mgr.tai_tjt);
        max_utc_tjt_err = max_utc_tjt_err.max((our_utc_tjt - utc_tjt).abs() * SECONDS_PER_DAY);
        let our_ut1_tjt = mgr.tai_tjt + mgr.ut1_tai_offset / SECONDS_PER_DAY;
        max_ut1_tjt_err = max_ut1_tjt_err.max((our_ut1_tjt - ut1_tjt).abs() * SECONDS_PER_DAY);
        let elapsed = tai_seconds - init_tai_seconds;
        max_tai_s_err = max_tai_s_err.max((mgr.tai_seconds - elapsed).abs());
    }

    println!(
        "  time_v4 common: {} points, TAI_tjt={max_tai_tjt_err:.2e}s, \
         UTC_tjt={max_utc_tjt_err:.2e}s, UT1_tjt={max_ut1_tjt_err:.2e}s, \
         TAI_s={max_tai_s_err:.2e}s",
        rows.len()
    );

    // TAI is exact. UTC crosses a leap second: JEOD applies the true-UTC
    // convention (seconds=60 counts within the leap second) so our table
    // lookup should differ by at most rounding noise (< 1 μs).
    assert!(
        max_tai_tjt_err < 1e-6,
        "TAI TJT error {max_tai_tjt_err:.4e} s"
    );
    assert!(
        max_utc_tjt_err < 1e-6,
        "UTC TJT error {max_utc_tjt_err:.4e} s"
    );
    // UT1 is interpolated from the IERS EOP table per `EopTable`, so
    // the day-long drift the constant-offset path missed is gone; the
    // residual is at table-precision noise level.
    assert!(
        max_ut1_tjt_err < 1e-6,
        "UT1 TJT error {max_ut1_tjt_err:.4e} s"
    );
    assert!(
        max_tai_s_err < 1e-9,
        "TAI seconds error {max_tai_s_err:.4e} s"
    );
}

// ── SIM_5_all_inclusive (RUN_UDE_initialized) ───────────────────────────────

/// SIM_5 RUN_UDE_initialized: exercises all the standard time scales — TAI,
/// TT, TDB, UTC, UT1, GMST, GPS — plus the first of JEOD's two MET scales
/// (`metveh1`).
///
/// Scope note: JEOD's SIM_5 also runs a second MET (`metveh2`) with a
/// hold/release toggle during the run. Our `SimulationTime` currently tracks a
/// single MET at a time, so `metveh2` is out of scope for this Tier 3 check
/// and the CSV's `metveh2.seconds` column is deliberately not consumed. The
/// single-MET hold/release behavior is exercised in `tier3_sim_met.rs`; adding
/// a second slot to `SimulationTime` is tracked separately.
///
/// All `*.seconds` columns logged for the other scales (UTC/UT1/TT/TDB/GPS)
/// are redundant with the TJT values we assert against (same absolute time,
/// different representation), so we validate only one representation per
/// scale — whichever one JEOD treats as the primary output (TJT for calendar
/// scales, seconds for GMST).
#[test]
fn tier3_time_v5_all() {
    let csv = test_data_path("time_v5_all_time_v5.csv");
    let rows = load_time_csv(&csv);
    let init = &rows[0];
    let init_tai_tjt = init.tai_tjt.expect("SIM_5 CSV must log TAI TJT");
    let init_ut1_tjt = init.ut1_tjt.expect("SIM_5 CSV must log UT1 TJT");
    let init_met1 = init
        .metveh1_seconds
        .expect("SIM_5 CSV must log metveh1 seconds");

    let mut mgr = SimulationTime::new(init_tai_tjt, default_leap_second_table());
    let ut1_tai_offset = (init_ut1_tjt - init_tai_tjt) * SECONDS_PER_DAY;
    mgr.set_ut1_tai_offset(ut1_tai_offset);

    // MET epoch from the input.py: metveh1 epoch 1998-12-31 23:59:00 UTC,
    // initial clock_second = 50, so metveh1 at t=0 should be 50 s. The CSV's
    // t=0 value gives us the exact MET/TAI offset without reconstructing
    // calendar math: epoch_at_parent = -met_at_t0.
    mgr.add_met(-init_met1);

    let dt = if rows.len() > 1 {
        rows[1].time - rows[0].time
    } else {
        1.0
    };

    let mut max_tai_tjt_err = 0.0_f64;
    let mut max_tt_tjt_err = 0.0_f64;
    let mut max_tdb_tjt_err = 0.0_f64;
    let mut max_utc_tjt_err = 0.0_f64;
    let mut max_ut1_tjt_err = 0.0_f64;
    let mut max_gmst_err = 0.0_f64;
    let mut max_gps_tjt_err = 0.0_f64;
    let mut max_met1_err = 0.0_f64;

    for (i, rec) in rows.iter().enumerate() {
        if i > 0 {
            mgr.advance(dt);
        }
        let tai_tjt = rec.tai_tjt.expect("SIM_5 CSV must log TAI TJT");
        let tt_tjt = rec.tt_tjt.expect("SIM_5 CSV must log TT TJT");
        let tdb_tjt = rec.tdb_tjt.expect("SIM_5 CSV must log TDB TJT");
        let utc_tjt = rec.utc_tjt.expect("SIM_5 CSV must log UTC TJT");
        let ut1_tjt = rec.ut1_tjt.expect("SIM_5 CSV must log UT1 TJT");
        let gmst = rec.gmst_seconds.expect("SIM_5 CSV must log GMST seconds");
        let gps_tjt = rec.gps_tjt.expect("SIM_5 CSV must log GPS TJT");
        let met1 = rec
            .metveh1_seconds
            .expect("SIM_5 CSV must log metveh1 seconds");

        max_tai_tjt_err = max_tai_tjt_err.max((mgr.tai_tjt - tai_tjt).abs() * SECONDS_PER_DAY);
        max_tt_tjt_err = max_tt_tjt_err.max((mgr.tt_tjt() - tt_tjt).abs() * SECONDS_PER_DAY);
        // Our TDB TJT derived like in tier3_sim_timescale.rs
        let our_tdb_tjt = mgr.tai_tjt + (mgr.tdb_seconds - mgr.tai_seconds) / SECONDS_PER_DAY;
        max_tdb_tjt_err = max_tdb_tjt_err.max((our_tdb_tjt - tdb_tjt).abs() * SECONDS_PER_DAY);
        let our_utc_tjt = mgr.leap_second_table.tai_to_utc_tjt(mgr.tai_tjt);
        max_utc_tjt_err = max_utc_tjt_err.max((our_utc_tjt - utc_tjt).abs() * SECONDS_PER_DAY);
        let our_ut1_tjt = mgr.tai_tjt + mgr.ut1_tai_offset / SECONDS_PER_DAY;
        max_ut1_tjt_err = max_ut1_tjt_err.max((our_ut1_tjt - ut1_tjt).abs() * SECONDS_PER_DAY);
        max_gmst_err = max_gmst_err.max((mgr.gmst_seconds - gmst).abs());
        // GPS TJT: JEOD convention reports GPS TJT = TAI TJT (both measure
        // the same absolute time, differing only in offset representation).
        max_gps_tjt_err = max_gps_tjt_err.max((mgr.tai_tjt - gps_tjt).abs() * SECONDS_PER_DAY);

        let our_met1 = mgr.get_met_seconds().expect("MET registered");
        max_met1_err = max_met1_err.max((our_met1 - met1).abs());
    }

    println!(
        "  time_v5 all: {} points, TAI_tjt={max_tai_tjt_err:.2e}s, TT={max_tt_tjt_err:.2e}s, \
         TDB={max_tdb_tjt_err:.2e}s, UTC={max_utc_tjt_err:.2e}s, UT1={max_ut1_tjt_err:.2e}s, \
         GMST={max_gmst_err:.2e}s, GPS={max_gps_tjt_err:.2e}s, MET1={max_met1_err:.2e}s",
        rows.len()
    );

    assert!(
        max_tai_tjt_err < 2e-6,
        "TAI TJT error {max_tai_tjt_err:.4e} s"
    );
    assert!(max_tt_tjt_err < 2e-6, "TT TJT error {max_tt_tjt_err:.4e} s");
    assert!(
        max_tdb_tjt_err < 2e-6,
        "TDB TJT error {max_tdb_tjt_err:.4e} s"
    );
    assert!(
        max_utc_tjt_err < 1e-5,
        "UTC TJT error {max_utc_tjt_err:.4e} s"
    );
    assert!(
        max_ut1_tjt_err < 2e-6,
        "UT1 TJT error {max_ut1_tjt_err:.4e} s"
    );
    assert!(max_gmst_err < 1e-4, "GMST error {max_gmst_err:.4e} s");
    assert!(
        max_gps_tjt_err < 2e-6,
        "GPS TJT error {max_gps_tjt_err:.4e} s"
    );
    assert!(max_met1_err < 1e-10, "MET1 error {max_met1_err:.4e} s");
}

// ── SIM_6_extension ─────────────────────────────────────────────────────────

/// SIM_6 RUN_tai_initialized: TAI initialized by calendar (2005-12-31
/// 23:59:50 UTC + leap offset). SIM_6 registers a user-defined "new" time
/// scale that exists only in that sim's verif code — we don't port it.
/// We verify TAI propagation only.
#[test]
fn tier3_time_v6_ext() {
    let csv = test_data_path("time_v6_ext_time_v6.csv");
    let rows = load_time_csv(&csv);
    let init = &rows[0];
    let init_tai_tjt = init.tai_tjt.expect("SIM_6 CSV must log TAI TJT");
    let init_tai_seconds = init.tai_seconds.expect("SIM_6 CSV must log TAI seconds");

    let mut mgr = SimulationTime::new(init_tai_tjt, default_leap_second_table());
    let dt = if rows.len() > 1 {
        rows[1].time - rows[0].time
    } else {
        1.0
    };

    let mut max_tai_s_err = 0.0_f64;
    let mut max_tai_tjt_err = 0.0_f64;
    let mut max_dyn_err = 0.0_f64;

    for (i, rec) in rows.iter().enumerate() {
        if i > 0 {
            mgr.advance(dt);
        }
        let tai_tjt = rec.tai_tjt.expect("SIM_6 CSV must log TAI TJT");
        let tai_seconds = rec.tai_seconds.expect("SIM_6 CSV must log TAI seconds");
        let dyn_seconds = rec.dyn_seconds.expect("SIM_6 CSV must log DYN seconds");
        let elapsed = tai_seconds - init_tai_seconds;
        max_tai_s_err = max_tai_s_err.max((mgr.tai_seconds - elapsed).abs());
        max_tai_tjt_err = max_tai_tjt_err.max((mgr.tai_tjt - tai_tjt).abs() * SECONDS_PER_DAY);
        max_dyn_err = max_dyn_err.max((mgr.get_seconds(TimeScaleId::DYN) - dyn_seconds).abs());
    }

    println!(
        "  time_v6 ext: {} points, TAI_s={max_tai_s_err:.2e}s, TAI_tjt={max_tai_tjt_err:.2e}s, \
         DYN={max_dyn_err:.2e}s",
        rows.len()
    );

    assert!(
        max_tai_s_err < 1e-9,
        "TAI seconds error {max_tai_s_err:.4e} s"
    );
    assert!(
        max_tai_tjt_err < 1e-6,
        "TAI TJT error {max_tai_tjt_err:.4e} s"
    );
    assert!(max_dyn_err < 1e-12, "DYN error {max_dyn_err:.4e} s");
}
