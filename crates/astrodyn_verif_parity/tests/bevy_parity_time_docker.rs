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
//! Five SIM cases land here — SIM_1 (DynamicTime only), SIM_2
//! (Dyn + TAI), SIM_4 (TAI + UTC + UT1 across the 1999-01-01
//! leap-second boundary), SIM_5 (all calendar scales: TAI/TT/TDB/UTC/
//! UT1/GMST/GPS), SIM_6 (TAI + DYN). SIM_3 (Dyn + UDE) is structurally
//! out of scope here: UDE is a [`astrodyn::TimeManager`] feature that
//! is not surfaced through the per-step `SimulationTimeR` resource the
//! Bevy pipeline propagates, so there is no Bevy-side state to assert
//! parity against. The same reasoning excludes MET-only fields from
//! the SIM_5 case — the calendar-scale fields it shares with the
//! production resource are still checked.
//!
//! The runner-side counterpart is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_time_docker.rs`; this
//! file carries the `bevy ≡ runner` half of the
//! `bevy ≡ runner ≈ JEOD` transitivity argument that the issue-#389
//! superset invariant requires.
//!
//! ## SIM_4 leap-second handling
//!
//! SIM_4's 86460 s window crosses the 1999-01-01 leap-second boundary.
//! Both runtimes consume the same `default_leap_second_table()` through
//! the shared `SimulationBuilder` factory and call the same
//! `SimulationTime::recompute_derived` path each tick, so the
//! leap-second transition is bit-identical on both sides by
//! construction. The runner-side test additionally compares against
//! the JEOD CSV via `with_eop_table(default_eop_table())` (a
//! `TimeManager`-only EOP-interpolation path); the parity wrapper does
//! not need that surface — both sides see the same constant
//! `ut1_tai_offset` written via [`SimulationTime::set_ut1_tai_offset`]
//! at construction.

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

use astrodyn::{default_leap_second_table, SimulationBuilder, SimulationTime};
use astrodyn_bevy::{SimulationBuilderBevyExt, SimulationTimeR};
use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use bevy::prelude::*;

const SECONDS_PER_DAY: f64 = 86400.0;

/// Minimal CSV row holding the columns this parity wrapper consumes.
/// Only `time` (cadence) and the t=0 row's `tai_tjt` / optional
/// `ut1_tjt` (epoch + UT1-TAI offset) are read — the per-tick
/// time-advance is deterministic on both runtimes given identical
/// initialisation, so subsequent rows feed only the loop cadence.
struct TimeDockerRow {
    time: f64,
    tai_tjt: Option<f64>,
    ut1_tjt: Option<f64>,
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

/// Build a body-less `SimulationBuilder` whose time pipeline is seeded
/// from the supplied CSV row-0 epoch. Optionally writes a constant
/// UT1-TAI offset derived from `init.ut1_tjt - init.tai_tjt` (matches
/// the SIM_5 runner path; SIM_4 also has UT1 logged but uses an EOP
/// table on the runner side that is not surfaced through
/// `SimulationTime` — both runtimes share the same constant offset
/// here, so bit-identity holds on both sides). The factory runs twice
/// per test (once per runtime) so each runtime sees bit-identical IC.
fn build_time_docker_builder(init: &TimeDockerRow, dt: f64) -> SimulationBuilder {
    let mut time = SimulationTime::new(initial_tai_tjt(init), default_leap_second_table());
    if let (Some(tai_tjt), Some(ut1_tjt)) = (init.tai_tjt, init.ut1_tjt) {
        let ut1_tai_offset = (ut1_tjt - tai_tjt) * SECONDS_PER_DAY;
        time.set_ut1_tai_offset(ut1_tai_offset);
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
    assert_time_bits_eq(t, label, &runner.time, &bevy_time(app));
}

/// Run a body-less SIM_X case: build both runtimes from the shared
/// factory, sanity-check IC alignment, then walk the CSV's rows in
/// lockstep, asserting bit-identical `SimulationTime` at every
/// checkpoint. Per-SIM `#[test]` entries call this with their own CSV
/// filename + cadence-fallback so a failure diagnostic names the sim.
fn run_sim_parity(label: &str, csv: &str, fallback_dt: f64) {
    let rows = load_csv(csv);
    assert!(
        rows.len() >= 2,
        "{label}: CSV {csv} must have at least 2 data rows for the parity walk"
    );
    let dt = cadence_dt(&rows, fallback_dt);
    let init = &rows[0];

    // ── Runner side ──
    let mut runner = build_time_docker_builder(init, dt)
        .build()
        .unwrap_or_else(|e| panic!("{label}: runner build failed: {e:?}"));

    // ── Bevy side — same factory, materialised under <Earth> ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let _handles = build_time_docker_builder(init, dt)
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
    assert_time_bits_eq(
        0.0,
        &format!("{label} init"),
        &runner.time,
        &bevy_time(&app),
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
    run_sim_parity("SIM_1_dyn_only", "time_v1_dyn_only_time_v1.csv", 1.0);
}

// ── SIM_2_dyn_plus_STD ──────────────────────────────────────────────────────

/// SIM_2 RUN_initialize_by_value: TAI initialised at TJT=10000
/// (early 1968, pre-leap-second era). Verifies TAI seconds + TAI TJT
/// + simtime parity between runtimes at the CSV's 1 s cadence.
#[test]
fn bevy_parity_time_v2_std() {
    run_sim_parity("SIM_2_std", "time_v2_std_time_v2.csv", 1.0);
}

// ── SIM_4_common_usage ──────────────────────────────────────────────────────

/// SIM_4 RUN_JEOD2x: TAI + UTC + UT1 initialised at 1998-12-31 00:00
/// UTC and sampled at 60 s cadence through t=86460 s, crossing the
/// 1999-01-01 leap-second boundary at t=86400 s. The CSV's t=0
/// `ut1_tjt - tai_tjt` seeds a constant `ut1_tai_offset` on both
/// runtimes; the leap-second transition is handled identically on
/// both sides via the shared `default_leap_second_table()` through
/// `SimulationTime::recompute_derived`. See the module-level
/// "SIM_4 leap-second handling" docstring for why this constant-offset
/// scoping is correct for the parity wrapper.
#[test]
fn bevy_parity_time_v4_common() {
    run_sim_parity("SIM_4_common", "time_v4_common_time_v4.csv", 60.0);
}

// ── SIM_5_all_inclusive (RUN_UDE_initialized) ───────────────────────────────

/// SIM_5 RUN_UDE_initialized: exercises every calendar time scale the
/// production `SimulationTime` carries — TAI, TT, TDB, UTC, UT1,
/// GMST, GPS. The runner-side test additionally validates `metveh1`;
/// MET is a `TimeManager`-only feature not surfaced through the per-
/// step `SimulationTimeR` resource, so it is out of scope for this
/// wrapper — the same module-level reasoning that excludes SIM_3
/// entirely.
///
/// The SIM_5 `RUN_UTC_initialized_tdb` variant is already covered by
/// `bevy_parity_timescale.rs`; this entry covers the complementary
/// `RUN_UDE_initialized` run.
#[test]
fn bevy_parity_time_v5_all() {
    run_sim_parity("SIM_5_all", "time_v5_all_time_v5.csv", 1.0);
}

// ── SIM_6_extension ─────────────────────────────────────────────────────────

/// SIM_6 RUN_tai_initialized: TAI initialised by calendar (2005-12-31
/// 23:59:50 UTC + leap offset). SIM_6 also registers a user-defined
/// "new" time scale that exists only in that sim's verif code — we
/// don't port it. Verifies TAI / simtime / derived calendar-scale
/// parity at 1 s cadence.
#[test]
fn bevy_parity_time_v6_ext() {
    run_sim_parity("SIM_6_ext", "time_v6_ext_time_v6.csv", 1.0);
}

/// Snapshot the Bevy app's `SimulationTimeR` resource into a fresh
/// `SimulationTime` clone the assertions can compare field-by-field
/// against the runner's. Cloning avoids holding a long-lived `Res`
/// across the next mutable world access in the loop body.
fn bevy_time(app: &App) -> SimulationTime {
    app.world().resource::<SimulationTimeR>().0.clone()
}

/// Assert every load-bearing `SimulationTime` field matches bit-for-
/// bit between the runner and the Bevy resource. Mirrors the field
/// set asserted by `bevy_parity_timescale.rs::assert_time_bits_eq` so
/// the two time-family wrappers stay in lockstep on what counts as
/// "the time-scale parity surface" — see the docstring there for why
/// `gmst_radians` follows `gmst_seconds` and why
/// `leap_second_table` / `tai_tjt_at_epoch` / `ut1_tai_offset` are
/// seeded once and only their derived scalars need per-tick assertion.
fn assert_time_bits_eq(t: f64, label: &str, runner: &SimulationTime, bevy: &SimulationTime) {
    fn bits_eq(t: f64, label: &str, field: &str, r: f64, b: f64) {
        assert!(
            r.to_bits() == b.to_bits(),
            "bevy_parity_time_docker: {label} at t={t:.6}s diverged on {field}:\n  \
             runner: {r} (bits={:#018x})\n  \
             bevy:   {b} (bits={:#018x})",
            r.to_bits(),
            b.to_bits(),
        );
    }
    bits_eq(
        t,
        label,
        "tai_seconds",
        runner.tai_seconds,
        bevy.tai_seconds,
    );
    bits_eq(t, label, "tai_tjt", runner.tai_tjt, bevy.tai_tjt);
    bits_eq(
        t,
        label,
        "tai_tjt_at_epoch",
        runner.tai_tjt_at_epoch,
        bevy.tai_tjt_at_epoch,
    );
    bits_eq(
        t,
        label,
        "utc_seconds",
        runner.utc_seconds,
        bevy.utc_seconds,
    );
    bits_eq(
        t,
        label,
        "ut1_seconds",
        runner.ut1_seconds,
        bevy.ut1_seconds,
    );
    bits_eq(t, label, "tt_seconds", runner.tt_seconds, bevy.tt_seconds);
    bits_eq(
        t,
        label,
        "tdb_seconds",
        runner.tdb_seconds,
        bevy.tdb_seconds,
    );
    bits_eq(
        t,
        label,
        "gmst_seconds",
        runner.gmst_seconds,
        bevy.gmst_seconds,
    );
    bits_eq(
        t,
        label,
        "gmst_radians",
        runner.gmst_radians,
        bevy.gmst_radians,
    );
    bits_eq(
        t,
        label,
        "gps_seconds",
        runner.gps_seconds,
        bevy.gps_seconds,
    );
    bits_eq(t, label, "simtime", runner.simtime, bevy.simtime);
    bits_eq(
        t,
        label,
        "ut1_tai_offset",
        runner.ut1_tai_offset,
        bevy.ut1_tai_offset,
    );
    bits_eq(
        t,
        label,
        "time_scale_factor",
        runner.time_scale_factor,
        bevy.time_scale_factor,
    );
}
