//! Bevy ↔ runner parity for the SIM_1..6 time verification sims that
//! ship as the runner-side `tier3_sim_time_docker.rs`.
//!
//! Drives both runtimes — [`astrodyn_runner::Simulation`] and the Bevy
//! `populate_app::<Earth>` bridge — through a body-less
//! [`astrodyn::SimulationBuilder`] seeded from the JEOD CSV row-0 epoch
//! and asserts bit-identical [`astrodyn::SimulationTime`] field values
//! at every CSV-cadence checkpoint.
//!
//! ## SIM coverage
//!
//! All six SIM cases land here — SIM_1 (DynamicTime only), SIM_2
//! (Dyn + TAI), SIM_3 (Dyn + UDE), SIM_4 (TAI + UTC + UT1 across the
//! 1999-01-01 leap-second boundary, EOP-interpolated UT1), SIM_5
//! (all calendar scales + `metveh1` MET), SIM_6 (TAI + DYN). After the
//! #577 unification `SimulationTime` carries the optional EOP table /
//! MET / UDE state, so every feature the runner-side
//! `tier3_sim_time_docker` exercises is reachable through
//! `SimulationTimeR` and gets a bit-identity backstop here.
//!
//! The runner-side counterpart is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_time_docker.rs`; this
//! file carries the `bevy ≡ runner` half of the
//! `bevy ≡ runner ≈ JEOD` transitivity argument that the issue-#389
//! superset invariant requires.
//!
//! ## SIM_4 leap-second + EOP handling
//!
//! SIM_4's 86460 s window crosses the 1999-01-01 leap-second boundary.
//! Both runtimes consume the same `default_leap_second_table()` and the
//! same `default_eop_table()` through the shared `SimulationBuilder`
//! factory, and call the same `SimulationTime::recompute_derived` path
//! each tick (which now re-interpolates `ut1_tai_offset` from the EOP
//! table per JEOD's `time_converter_tai_ut1::convert_a_to_b`). The
//! leap-second transition and the per-step UT1 interpolation are
//! bit-identical on both sides by construction.

#![allow(
    clippy::float_cmp,
    reason = "bevy-parity tests assert bit-exact identity between runner and Bevy time fields"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "CSV row counts (≤1442) fit trivially in f64 mantissa and usize"
)]

use std::time::Duration;

use astrodyn::{default_eop_table, default_leap_second_table, SimulationBuilder, SimulationTime};
use astrodyn_bevy::SimulationBuilderBevyExt;
use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use bevy::prelude::*;

mod common;
use common::{assert_simulation_time_bits_eq, bevy_sim_time};

const SECONDS_PER_DAY: f64 = 86400.0;

/// Minimal CSV row holding the columns this parity wrapper consumes.
/// Only `time` (cadence) and a few row-0 anchors are read — the
/// per-tick time-advance is deterministic on both runtimes given
/// identical initialisation, so subsequent rows feed only the loop
/// cadence. The optional `dyn_seconds` / `ude_seconds` / `metveh1_seconds`
/// columns are read from row 0 only, to derive the MET/UDE epochs (the
/// per-tick `assert_simulation_time_bits_eq` then compares the
/// re-derived values against each other, not against the CSV).
struct TimeDockerRow {
    time: f64,
    tai_tjt: Option<f64>,
    ut1_tjt: Option<f64>,
    dyn_seconds: Option<f64>,
    ude_seconds: Option<f64>,
    metveh1_seconds: Option<f64>,
}

/// Parse a time-verification CSV header so missing columns surface as
/// `None` instead of NaN. Mirrors the runner-side
/// `tier3_sim_time_docker.rs::load_time_csv` column-discovery shape
/// (per-column header probe, sim-specific column subsets) while only
/// retaining the fields the parity wrapper actually consumes.
fn load_csv(filename: &str) -> Vec<TimeDockerRow> {
    let path = test_data_path(filename);
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
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

    fn name_of(h: &str) -> &str {
        h.split(|c: char| c.is_whitespace() || c == '{')
            .next()
            .unwrap_or(h)
            .trim()
    }
    let col = |name: &str| -> Option<usize> { headers.iter().position(|h| name_of(h) == name) };

    let i_time = col("sys.exec.out.time").expect("CSV must log sys.exec.out.time");
    let i_tai_tjt = col("jeod_time.time_tai.trunc_julian_time")
        .or_else(|| col("jeod_time.tai.trunc_julian_time"));
    let i_ut1_tjt = col("jeod_time.time_ut1.trunc_julian_time")
        .or_else(|| col("jeod_time.ut1.trunc_julian_time"));
    let i_dyn_s = col("jeod_time.time_manager.dyn_time.seconds")
        .or_else(|| col("jeod_time.manager.dyn_time.seconds"));
    let i_ude_s = col("jeod_time.time_ude.seconds").or_else(|| col("jeod_time.ude.seconds"));
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
        rows.push(TimeDockerRow {
            time: p(i_time),
            tai_tjt: i_tai_tjt.map(p),
            ut1_tjt: i_ut1_tjt.map(p),
            dyn_seconds: i_dyn_s.map(p),
            ude_seconds: i_ude_s.map(p),
            metveh1_seconds: i_met1_s.map(p),
        });
    }
    assert!(!rows.is_empty(), "no data rows in {}", path.display());
    rows
}

/// Best-effort initial TAI TJT for a CSV. SIM_1 / SIM_3 don't log TAI
/// at all (they exercise DynamicTime and UDE only) — those sims have
/// no absolute TAI anchor, so fall back to J2000 the way the runner
/// side does in `tier3_sim_time_docker::initial_tai_tjt`.
fn initial_tai_tjt(first: &TimeDockerRow) -> f64 {
    first.tai_tjt.unwrap_or(astrodyn::J2000_TAI_TJT)
}

/// Per-SIM customization knobs applied on top of the shared builder
/// factory. Each test fills in only the knobs its SIM needs; the rest
/// stay at their defaults so unrelated SIMs aren't affected.
#[derive(Default, Clone, Copy)]
struct TimeDockerSetup {
    /// SIM_5: register a `metveh1` MET with the given TAI-seconds epoch.
    met_epoch_tai_seconds: Option<f64>,
    /// SIM_3: register a UDE with the given parent-seconds epoch.
    ude_epoch_in_parent: Option<f64>,
    /// SIM_4: install the bundled IERS EOP table for per-tick UT1
    /// interpolation. When `false` the constant `ut1_tai_offset` written
    /// from the CSV row-0 `(ut1_tjt - tai_tjt)` is used instead.
    install_eop_table: bool,
}

/// Build a body-less `SimulationBuilder` whose time pipeline is seeded
/// from the supplied CSV row-0 epoch and per-SIM setup. The factory
/// runs twice per test (once per runtime) so each runtime sees
/// bit-identical IC.
fn build_time_docker_builder(
    init: &TimeDockerRow,
    dt: f64,
    setup: TimeDockerSetup,
) -> SimulationBuilder {
    let mut time = SimulationTime::new(initial_tai_tjt(init), default_leap_second_table());
    if setup.install_eop_table {
        // EOP table re-interpolates `ut1_tai_offset` on every advance;
        // do not overwrite with a constant offset afterwards (that
        // would defeat the `with_eop_table` JEOD path that SIM_4 is
        // here to exercise).
        time = time.with_eop_table(default_eop_table());
    } else if let (Some(tai_tjt), Some(ut1_tjt)) = (init.tai_tjt, init.ut1_tjt) {
        let ut1_tai_offset = (ut1_tjt - tai_tjt) * SECONDS_PER_DAY;
        time.set_ut1_tai_offset(ut1_tai_offset);
    }
    if let Some(epoch_in_parent) = setup.ude_epoch_in_parent {
        time.add_ude(epoch_in_parent);
    }
    if let Some(met_epoch_tai_s) = setup.met_epoch_tai_seconds {
        time.add_met(met_epoch_tai_s);
    }
    SimulationBuilder::new(time, dt)
}

/// Derive the integrator timestep from the first two CSV rows. JEOD's
/// time-verification sims log at 1 s cadence for SIM_1/2/3/5/6 and at
/// 60 s for SIM_4. The runner-side test reads the same
/// `rows[1].time - rows[0].time` to drive its `mgr.advance(dt)` loop,
/// so the parity wrapper picks up the same value and stays in
/// lockstep with both the JEOD reference cadence and the runner-side
/// fixture.
fn cadence_dt(rows: &[TimeDockerRow], fallback: f64) -> f64 {
    if rows.len() > 1 {
        rows[1].time - rows[0].time
    } else {
        fallback
    }
}

/// Step both runtimes one CSV-cadence tick forward. Mirrors the
/// timescale parity wrapper's loop body so the two parity tests carry
/// the same shape: `runner.step()` first, then advance Bevy's
/// `Time<Fixed>` by `dt` seconds and run a single `FixedUpdate` pass.
///
/// Asserts bit-identical `SimulationTime` after each tick — the
/// actual divergence detector.
fn step_one_tick(label: &str, t: f64, runner: &mut astrodyn_runner::Simulation, app: &mut App) {
    runner.step().expect("runner step failed");
    let dt = app.world().resource::<astrodyn_bevy::IntegrationDtR>().0;
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .advance_by(Duration::from_secs_f64(dt));
    app.world_mut().run_schedule(FixedUpdate);
    assert_simulation_time_bits_eq(t, label, &runner.time, &bevy_sim_time(app));
}

/// Run a body-less SIM_X case: build both runtimes from the shared
/// factory + per-SIM setup, sanity-check IC alignment, then walk the
/// CSV's rows in lockstep, asserting bit-identical `SimulationTime` at
/// every checkpoint. Per-SIM `#[test]` entries call this with their own
/// CSV filename, cadence-fallback, and setup so a failure diagnostic
/// names the sim.
fn run_sim_parity(label: &str, csv: &str, fallback_dt: f64, setup: TimeDockerSetup) {
    let rows = load_csv(csv);
    assert!(
        rows.len() >= 2,
        "{label}: CSV {csv} must have at least 2 data rows for the parity walk"
    );
    let dt = cadence_dt(&rows, fallback_dt);
    let init = &rows[0];

    // ── Runner side ──
    let mut runner = build_time_docker_builder(init, dt, setup)
        .build()
        .unwrap_or_else(|e| panic!("{label}: runner build failed: {e:?}"));

    // ── Bevy side — same factory, materialised under <Earth> ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let _handles = build_time_docker_builder(init, dt, setup)
        .populate_app::<astrodyn::Earth>(&mut app)
        .unwrap_or_else(|e| panic!("{label}: populate_app failed: {e:?}"));
    // `MinimalPlugins` does not auto-run `Startup`; mirror the
    // timescale parity wrapper and trigger it so Startup-time resource
    // wiring (root frame entity, etc.) lands before the first
    // FixedUpdate pass — the body-less scenario has no per-source
    // frame trees to wire, but the schedule shape stays uniform with
    // every other parity wrapper.
    app.world_mut().run_schedule(Startup);

    // Sanity-check IC alignment before stepping. Cheap to verify and
    // gives a clear failure site if a future refactor of
    // `populate_app` drifts the seeded `SimulationTimeR` from the
    // runner's `Simulation.time`.
    assert_simulation_time_bits_eq(
        0.0,
        &format!("{label} init"),
        &runner.time,
        &bevy_sim_time(&app),
    );

    for (idx, rec) in rows.iter().enumerate().skip(1) {
        step_one_tick(
            &format!("{label} tick {idx} t={t:.3}", t = rec.time),
            rec.time,
            &mut runner,
            &mut app,
        );
    }
}

// ── SIM_1_dyn_only ──────────────────────────────────────────────────────────

/// SIM_1: DynamicTime-only sim. No TAI in the CSV — both runtimes
/// fall back to the J2000 TAI anchor, and the only meaningful
/// time-scale comparison is `simtime` (== DYN at `scale_factor = 1`)
/// plus the derived calendar scales hanging off the synthetic anchor.
#[test]
fn bevy_parity_time_v1_dyn_only() {
    run_sim_parity(
        "SIM_1_dyn_only",
        "time_v1_dyn_only_time_v1.csv",
        1.0,
        TimeDockerSetup::default(),
    );
}

// ── SIM_2_dyn_plus_STD ──────────────────────────────────────────────────────

/// SIM_2 RUN_initialize_by_value: TAI initialised at TJT=10000
/// (early 1968, pre-leap-second era). Verifies TAI seconds + TAI TJT
/// + simtime parity between runtimes at the CSV's 1 s cadence.
#[test]
fn bevy_parity_time_v2_std() {
    run_sim_parity(
        "SIM_2_std",
        "time_v2_std_time_v2.csv",
        1.0,
        TimeDockerSetup::default(),
    );
}

// ── SIM_3_dyn_plus_UDE ──────────────────────────────────────────────────────

/// SIM_3 RUN_init_by_ude: UDE registered with `clock_second = -5`. After
/// initialisation UDE starts at -5 s; DynamicTime starts at 0 and the
/// UDE epoch sits at Dyn = +5 s. At each step JEOD reports
/// `UDE = Dyn - epoch_in_parent`. Both runtimes register the same UDE
/// via `add_ude(init_dyn - init_ude)`, and the bit-identity assertion
/// covers the per-tick `ude[0].seconds` re-derivation as well as the
/// standard calendar scales hanging off the synthetic J2000 anchor.
#[test]
fn bevy_parity_time_v3_ude() {
    let rows = load_csv("time_v3_ude_time_v3.csv");
    let init = &rows[0];
    let init_ude = init.ude_seconds.expect("SIM_3 CSV must log UDE seconds");
    let init_dyn = init.dyn_seconds.expect("SIM_3 CSV must log DYN seconds");
    let epoch_in_parent = init_dyn - init_ude;
    run_sim_parity(
        "SIM_3_ude",
        "time_v3_ude_time_v3.csv",
        1.0,
        TimeDockerSetup {
            ude_epoch_in_parent: Some(epoch_in_parent),
            ..TimeDockerSetup::default()
        },
    );
}

// ── SIM_4_common_usage ──────────────────────────────────────────────────────

/// SIM_4 RUN_JEOD2x: TAI + UTC + UT1 initialised at 1998-12-31 00:00
/// UTC and sampled at 60 s cadence through t=86460 s, crossing the
/// 1999-01-01 leap-second boundary at t=86400 s. Both runtimes install
/// the bundled IERS EOP table so `ut1_tai_offset` interpolates per
/// tick (mirroring the runner-side `with_eop_table(default_eop_table())`
/// path). The leap-second transition stays bit-identical via the shared
/// `default_leap_second_table()` and the same
/// `SimulationTime::recompute_derived` path.
#[test]
fn bevy_parity_time_v4_common() {
    run_sim_parity(
        "SIM_4_common",
        "time_v4_common_time_v4.csv",
        60.0,
        TimeDockerSetup {
            install_eop_table: true,
            ..TimeDockerSetup::default()
        },
    );
}

// ── SIM_5_all_inclusive (RUN_UDE_initialized) ───────────────────────────────

/// SIM_5 RUN_UDE_initialized: exercises every calendar time scale the
/// production `SimulationTime` carries — TAI, TT, TDB, UTC, UT1,
/// GMST, GPS — and additionally registers a `metveh1` MET. JEOD's
/// SIM_5 also runs a second MET (`metveh2`) with a hold/release toggle
/// during the run; our `SimulationTime` currently tracks a single MET
/// at a time, so `metveh2` is out of scope for both the runner-side
/// tier3 test and this parity wrapper. The single-MET path here is
/// load-bearing because it is the only Bevy-side coverage of
/// `MissionElapsedTime::update` running through `time_advance_system`.
///
/// The SIM_5 `RUN_UTC_initialized_tdb` variant is already covered by
/// `bevy_parity_timescale.rs`; this entry covers the complementary
/// `RUN_UDE_initialized` run.
#[test]
fn bevy_parity_time_v5_all() {
    let rows = load_csv("time_v5_all_time_v5.csv");
    let init = &rows[0];
    let init_met1 = init
        .metveh1_seconds
        .expect("SIM_5 CSV must log metveh1 seconds");
    // MET epoch is the TAI-seconds offset that puts the CSV row-0
    // `metveh1.seconds` value on the current TAI clock. Since `tai_seconds`
    // is 0 at construction and `MET = tai_seconds - epoch`, the epoch
    // that produces MET=init_met1 is `-init_met1`.
    let met_epoch = -init_met1;
    run_sim_parity(
        "SIM_5_all",
        "time_v5_all_time_v5.csv",
        1.0,
        TimeDockerSetup {
            met_epoch_tai_seconds: Some(met_epoch),
            ..TimeDockerSetup::default()
        },
    );
}

// ── SIM_6_extension ─────────────────────────────────────────────────────────

/// SIM_6 RUN_tai_initialized: TAI initialised by calendar (2005-12-31
/// 23:59:50 UTC + leap offset). SIM_6 also registers a user-defined
/// "new" time scale that exists only in that sim's verif code — we
/// don't port it. Verifies TAI / simtime / derived calendar-scale
/// parity at 1 s cadence.
#[test]
fn bevy_parity_time_v6_ext() {
    run_sim_parity(
        "SIM_6_ext",
        "time_v6_ext_time_v6.csv",
        1.0,
        TimeDockerSetup::default(),
    );
}
