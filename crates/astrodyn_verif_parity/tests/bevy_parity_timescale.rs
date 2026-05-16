//! Bevy ↔ runner parity for the SIM_5_all_inclusive timescale scenario.
//!
//! Drives both runtimes — [`astrodyn_runner::Simulation`] and the Bevy
//! `populate_app::<Earth>` bridge — through the same body-less
//! [`astrodyn::SimulationBuilder`] for 2 hours at 60 s intervals and
//! asserts bit-identical [`astrodyn::SimulationTime`] field values
//! (`tai_tjt`, `tai_seconds`, `utc_seconds`, `ut1_seconds`, `tt_seconds`,
//! `tdb_seconds`, `gmst_seconds`, `gps_seconds`, `simtime`) at every
//! checkpoint.
//!
//! No bodies and no gravity sources are configured — the JEOD reference
//! sim itself only validates time-scale conversions, so a runner-vs-bevy
//! parity assertion on the integration-pipeline output would have
//! nothing to compare. The pipeline's `time_advance_system` /
//! `Simulation::step()` still advances `SimulationTimeR` on the Bevy
//! side every tick, so the time-scale fields are the load-bearing
//! comparison target.
//!
//! The runner-side counterpart is
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_timescale.rs`; the parity
//! wrapper carries the `bevy ≡ runner` half of the
//! `bevy ≡ runner ≈ JEOD` transitivity argument that the issue-#389
//! superset invariant requires.

#![allow(
    clippy::float_cmp,
    reason = "bevy-parity tests assert bit-exact identity between runner and Bevy time fields"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "checkpoint count (120) fits trivially in f64 mantissa and usize"
)]

use std::time::Duration;

use astrodyn::{SimulationBuilder, SimulationTime};
use astrodyn_bevy::{SimulationBuilderBevyExt, SimulationTimeR};
use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use bevy::prelude::*;

const SECONDS_PER_DAY: f64 = 86400.0;

/// Subset of the SIM_5_all_inclusive timescale CSV consumed by the
/// parity wrapper: only `tai_tjt` and `ut1_tjt` from row 0 are read
/// (epoch + UT1-TAI offset). Subsequent rows are not consumed — the
/// per-tick time advancement is deterministic on both runtimes given
/// identical initialisation, so the row count alone drives the
/// checkpoint cadence.
struct TimescaleEpoch {
    tai_tjt_at_epoch: f64,
    ut1_tai_offset: f64,
}

/// Read the t=0 row of the JEOD timescale CSV and return the epoch
/// inputs the SimulationBuilder needs. The runner-side counterpart at
/// `tier3_sim_timescale.rs` parses the same row through a richer
/// `TimescaleRecord` for tolerance checks against the full per-row
/// reference series; the parity wrapper only needs the IC fields.
fn load_timescale_epoch() -> TimescaleEpoch {
    let csv_path = test_data_path("timescale_tdb_timescale.csv");
    let content = std::fs::read_to_string(&csv_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_5_all_inclusive CSV from {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
            csv_path.display()
        )
    });
    let mut row0 = None;
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 9,
            "timescale CSV line {}: expected >=9 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 {
            f[idx]
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("timescale CSV column {idx} parse failed: {e}"))
        };
        let tai_tjt = p(1);
        let ut1_tjt = p(4);
        let ut1_tai_offset = (ut1_tjt - tai_tjt) * SECONDS_PER_DAY;
        row0 = Some(TimescaleEpoch {
            tai_tjt_at_epoch: tai_tjt,
            ut1_tai_offset,
        });
        break;
    }
    row0.expect("timescale CSV has no data rows after header")
}

/// Build a body-less `SimulationBuilder` whose time pipeline is seeded
/// from the SIM_5_all_inclusive epoch. Used by both runtimes so the
/// per-tick `time_advance_system` / `Simulation::step` sees bit-identical
/// inputs and lands the same `SimulationTime` field values.
fn build_timescale_builder() -> SimulationBuilder {
    let epoch = load_timescale_epoch();
    let mut time = SimulationTime::new(
        epoch.tai_tjt_at_epoch,
        astrodyn::default_leap_second_table(),
    );
    time.set_ut1_tai_offset(epoch.ut1_tai_offset);
    SimulationBuilder::new(time, DT)
}

/// Step size between checkpoints (s). Matches the SIM_5_all_inclusive
/// reference cadence so the parity checkpoints land exactly on the
/// JEOD-logged time scale samples — the same cadence
/// `tier3_simulation_timescale_tdb` uses.
const DT: f64 = 60.0;

/// Total propagation window (s). 2 hours at 60 s per tick = 120
/// checkpoints, mirroring the runner-side tier3 test's full per-row
/// scan against the JEOD reference.
const WINDOW_S: f64 = 2.0 * 60.0 * 60.0;

#[test]
fn bevy_parity_timescale() {
    // ── Runner side ──
    let runner_builder = build_timescale_builder();
    let dt = runner_builder.dt;
    let mut runner = runner_builder
        .build()
        .expect("runner build (body-less timescale)");

    // ── Bevy side — same factory, materialised under <Earth> ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let _handles = build_timescale_builder()
        .populate_app::<astrodyn::Earth>(&mut app)
        .expect("populate_app under <Earth>");
    // `MinimalPlugins` does not auto-run `Startup`; the parity loop
    // drives `FixedUpdate` directly. The body-less scenario has no
    // sources to register frames for, but Startup-time resource wiring
    // (e.g. `RootFrameEntityR`) still happens here so the schedule is
    // in the same "post-Startup" shape mission code observes.
    app.world_mut().run_schedule(Startup);

    // Sanity-check epoch alignment before the first step: both runtimes
    // were constructed from the *same* factory but f64 equality is
    // cheap to verify and gives a clear failure site if a future
    // refactor of `populate_app` drifts the initial `SimulationTimeR`
    // from the runner's `Simulation.time`.
    assert_time_bits_eq(0.0, "init", &runner.time, &bevy_time(&app));

    let n_steps = (WINDOW_S / dt).round() as usize;
    assert!(n_steps >= 1, "window {WINDOW_S}s must be >= dt {dt}s");

    let mut t = 0.0_f64;
    for step_idx in 1..=n_steps {
        runner.step().expect("runner step failed");
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
        t += dt;

        let bevy = bevy_time(&app);
        assert_time_bits_eq(t, &format!("tick {step_idx}"), &runner.time, &bevy);
    }
}

/// Snapshot the Bevy app's `SimulationTimeR` resource into a fresh
/// `SimulationTime` clone the assertions can compare field-by-field
/// against the runner's. Cloning avoids holding a long-lived `Res`
/// across the next mutable world access in the loop body.
fn bevy_time(app: &App) -> SimulationTime {
    app.world().resource::<SimulationTimeR>().0.clone()
}

/// Assert every load-bearing `SimulationTime` field matches bit-for-bit
/// between the runner and the Bevy resource. `gmst_radians` follows
/// `gmst_seconds` through `recompute_derived` so checking the seconds
/// variant covers both; `leap_second_table` is `Copy`-by-value seeded
/// from the same `default_leap_second_table()` on both runtimes, and
/// `ut1_tai_offset` / `tai_tjt_at_epoch` were written from the same
/// JEOD CSV row 0 via `build_timescale_builder`. Asserting the derived
/// scalars at each tick is the actual divergence detector.
fn assert_time_bits_eq(t: f64, label: &str, runner: &SimulationTime, bevy: &SimulationTime) {
    fn bits_eq(t: f64, label: &str, field: &str, r: f64, b: f64) {
        assert!(
            r.to_bits() == b.to_bits(),
            "bevy_parity_timescale: {label} at t={t:.6}s diverged on {field}:\n  \
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
