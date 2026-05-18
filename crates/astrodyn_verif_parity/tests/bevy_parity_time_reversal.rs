//! Bevy ↔ runner parity for the SIM_7_time_reversal RUN_1 scenario:
//! a single Earth-orbiting body propagating spherical-gravity RK4
//! through a forward-then-reverse phase pair with
//! `time_scale_factor = -1.0` flipping at the reversal instant.
//!
//! Mirrors `crates/astrodyn_verif_jeod/tests/tier3_sim_time_reversal.rs`
//! body-for-body: same JEOD CSV-row-0 epoch, same GEM-T1 μ, same
//! LVLH-derived attitude, same dt = 0.03125 s. The reversal-flip is
//! driven through [`SimContext::set_time_scale_factor`] on both
//! runtimes so the trait carries the time-direction mutation alongside
//! the existing `detach_subtree` / `attach_subtree_aligned` /
//! `set_body_external_force` family.
//!
//! ## Comparison cadence
//!
//! The reference CSV samples at 60 s — exactly 1920 ticks of the
//! 0.03125 s integration step, so each comparison checkpoint lands on
//! an integer multiple of `dt`. Bit-identity at the CSV cadence
//! implies bit-identity at every intermediate tick under the
//! monotonic-divergence argument
//! `VerificationCaseParityExt::run_and_assert_parity` uses (once two
//! runtimes drift, they stay drifted).
//!
//! ## Window scope
//!
//! The parity wrapper propagates a [`PARITY_FORWARD_RECORDS`]-record
//! forward window then immediately flips and propagates the same
//! count in reverse, instead of mirroring the JEOD CSV's full
//! 60 000 s × 2 phases. The full window contains ~3.84 M Bevy
//! `FixedUpdate` ticks (1920 per record × 2000 records) which would
//! make this wrapper a multi-hour CI run. The 60-record subset still
//! exercises every load-bearing path (~3-4 orbital revolutions each
//! phase plus the reversal-flip transition) — and the runner-side
//! tier3 sibling already validates physics over the full window
//! against JEOD. Bit-identity between runner and bevy is monotonic,
//! so a 60-record bit-match implies a 1000-record bit-match would
//! also hold against this same pair of runtime configurations.

#![allow(
    clippy::float_cmp,
    reason = "bevy-parity tests assert bit-exact identity between runner and Bevy state fields"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test step counts fit exactly in f64 mantissa and usize"
)]

use std::time::Duration;

use astrodyn::{
    typed_bridge, GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
    GravitySourceEntry, JeodQuat, MassProperties, RotationalState, SimulationBuilder,
    SimulationTime, TranslationalState, VehicleConfig,
};
use astrodyn_bevy::{
    RotationalStateC, SimulationBuilderBevyExt, SimulationTimeR, TranslationalStateC,
};
use astrodyn_runner::{Simulation, SimulationBuilderExt};
use astrodyn_verif_jeod::tier3_csv::test_data_path;
use astrodyn_verif_jeod::verification::SimContext;
use astrodyn_verif_parity::BevySimContext;
use bevy::prelude::*;
use glam::DVec3;

/// One row of `reversal_run1_reversal.csv`. The parity wrapper only
/// consumes `time` (cadence + reversal-point detection) and the t=0
/// row's `tai_tjt` (epoch) + `position` / `velocity` (initial state).
/// JEOD-logged state at t > 0 is **not** read into the parity assertion
/// — only the runner-vs-bevy comparison is performed here, matching the
/// existing `apollo_trajectory` parity wrapper's CSV-times-only pattern.
#[allow(dead_code, reason = "tai_seconds documents reversal-point semantics")]
struct ReversalRow {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    tai_seconds: f64,
    tai_tjt: f64,
}

/// Parse the run1 reversal CSV at the layout used by JEOD's
/// `SIM_7_time_reversal/SET_test/RUN_1` (`time, x, vx, y, vy, z, vz,
/// tai_seconds, tai_tjt`). Interleaved pos/vel columns mirror the
/// JEOD `log_state_ASCII` layout for this sim and differ from the
/// generic `OrbInit` (`time, x, y, z, vx, vy, vz`) — bespoke loader is
/// the simplest fit here.
fn load_reversal_run1() -> Vec<ReversalRow> {
    let csv_path = test_data_path("reversal_run1_reversal.csv");
    let content = std::fs::read_to_string(&csv_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_7_time_reversal RUN_1 CSV from {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
            csv_path.display()
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
            "reversal CSV line {}: expected >=9 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 {
            f[idx]
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("reversal CSV column {idx} parse failed: {e}"))
        };
        records.push(ReversalRow {
            time: p(0),
            position: DVec3::new(p(1), p(3), p(5)),
            velocity: DVec3::new(p(2), p(4), p(6)),
            tai_seconds: p(7),
            tai_tjt: p(8),
        });
    }
    records
}

/// Dynamics timestep from SIM_7_time_reversal RUN_1's S_define
/// (`DYNAMICS = 0.03125 s`, i.e. 32 Hz). Hard-coded here because the
/// JEOD verification sim's S_define value is the source of truth and
/// the parity wrapper reads it implicitly through the CSV cadence
/// (60 s = 1920 × 0.03125 s).
const DT: f64 = 0.03125;

/// Number of forward CSV records the parity wrapper walks before
/// firing the reversal flip. The reverse phase walks the same count.
/// 60 records × 60 s = 3 600 s of sim time per phase = ~3-4 LEO
/// revolutions, enough to exercise the RK4 + spherical-gravity path
/// non-trivially without the full 1 000-record JEOD window (which
/// would put this wrapper at ~3.84 M Bevy schedule runs — a multi-
/// hour CI cost). See the module-level "Window scope" docstring for
/// why the bit-identity argument carries to longer windows under the
/// same runtime pair.
const PARITY_FORWARD_RECORDS: usize = 60;

/// μ for spherical Earth gravity in the SIM_7_time_reversal RUN_1
/// configuration. Matches the runner-side tier3 test's
/// `load_mu_earth_gemt1()`.
fn load_mu_earth_gemt1() -> f64 {
    astrodyn::gravity_fixtures::load_gemt1().mu
}

/// Build the `SimulationBuilder` for SIM_7_time_reversal RUN_1 from
/// the row-0 epoch + LVLH-pitched attitude. The factory runs twice (once
/// per runtime) — both calls take the same `records` slice so each
/// runtime sees bit-identical IC inputs.
///
/// The body is constructed with `rot` populated by JEOD's
/// `LvlhDerivedState` initialisation: yaw=0, pitch=-11.6°, roll=0,
/// `omega=0` — the same triple the tier3 test computes from
/// `compute_body_lvlh_frame(init.position, init.velocity)` and
/// `glam::DMat3::from_rotation_y(-11.6°)`.
fn build_reversal_run1_builder(init: &ReversalRow) -> SimulationBuilder {
    let mu_earth_gemt1 = load_mu_earth_gemt1();
    let leap_table = astrodyn::default_leap_second_table();
    // Epoch: TAI TJT taken from the CSV row 0 (JEOD source data — see
    // `Tier 3 Cross-Validation` in CLAUDE.md). The parity wrapper
    // never reads JEOD-logged state at t > 0 into the comparison.
    let time = SimulationTime::new(init.tai_tjt, leap_table);
    let mut sb = SimulationBuilder::new(time, DT);

    let earth = sb.add_source("Earth", {
        let mut e = GravitySourceEntry::new(
            GravitySource {
                mu: mu_earth_gemt1,
                model: GravityModel::PointMass,
            },
            astrodyn::Position::<astrodyn::RootInertial>::zero(),
            None,
        );
        e.central = true;
        e
    });

    // JEOD initialises attitude in LVLH: yaw=0, pitch=-11.6°, roll=0,
    // omega=0. Compute the LVLH frame and apply the Euler rotation,
    // then convert to JeodQuat (left-transformation convention). Same
    // sequence as the runner-side `tier3_sim_time_reversal_run1`.
    let lvlh = astrodyn::compute_body_lvlh_frame(init.position, init.velocity);
    let t_inertial_lvlh = lvlh.t_parent_this.transpose();
    let pitch = -11.6_f64.to_radians();
    let t_lvlh_body = glam::DMat3::from_rotation_y(pitch);
    let t_inertial_body = t_inertial_lvlh * t_lvlh_body;
    let glam_quat = glam::DQuat::from_mat3(&t_inertial_body);
    let init_quat = JeodQuat::new(glam_quat.w, glam_quat.x, glam_quat.y, glam_quat.z);

    sb.add_body(VehicleConfig {
        trans: typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        rot: Some(typed_bridge::rot_raw_to_self_ref(&RotationalState {
            quaternion: init_quat,
            ang_vel_body: DVec3::ZERO,
        })),
        // Mass irrelevant for spherical gravity; matches runner-side tier3.
        mass: Some(typed_bridge::mass_raw_to_self_ref(&MassProperties::new(
            1.0,
        ))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        ..Default::default()
    });
    sb
}

#[test]
fn bevy_parity_time_reversal() {
    let records = load_reversal_run1();
    assert!(
        records.len() > 2 * PARITY_FORWARD_RECORDS,
        "bevy_parity_time_reversal: reversal_run1_reversal.csv has {} rows; need at least \
         {} for the truncated parity window (forward + reverse phases of \
         {PARITY_FORWARD_RECORDS} records each)",
        records.len(),
        2 * PARITY_FORWARD_RECORDS,
    );
    let init = &records[0];

    // ── Runner side ──
    let mut runner = build_reversal_run1_builder(init)
        .build()
        .expect("runner build for SIM_7_time_reversal RUN_1");

    // ── Bevy side — same factory, materialised under <Earth> ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let handles = build_reversal_run1_builder(init)
        .populate_app::<astrodyn::Earth>(&mut app)
        .expect("populate_app under <Earth> for SIM_7_time_reversal RUN_1");
    let vehicle_entity = handles.body_entities[0];
    let source_entities = handles.source_entities.clone();
    let body_entities = handles.body_entities.clone();
    // `MinimalPlugins` does not auto-run `Startup`; mirror every other
    // parity wrapper and trigger Startup so per-source frame trees are
    // wired before the first FixedUpdate.
    app.world_mut().run_schedule(Startup);

    // Sanity-check IC alignment before stepping: both runtimes were
    // constructed from the same factory but f64 equality is cheap to
    // verify and gives a clear failure site if a future refactor of
    // `populate_app` drifts the seeded state.
    assert_trans_bits_eq("init", 0.0, &runner, &app, vehicle_entity);
    assert_rot_bits_eq("init", 0.0, &runner, &app, vehicle_entity);
    assert_time_scale_bits_eq("init", 0.0, &runner, &app);

    // Phase 1: forward propagation across `PARITY_FORWARD_RECORDS`
    // reference checkpoints, asserting bit-identity at every record.
    for rec in records.iter().take(PARITY_FORWARD_RECORDS + 1).skip(1) {
        step_to_record(rec.time, &mut runner, &mut app);
        assert_trans_bits_eq("fwd", rec.time, &runner, &app, vehicle_entity);
        assert_rot_bits_eq("fwd", rec.time, &runner, &app, vehicle_entity);
        assert_time_scale_bits_eq("fwd", rec.time, &runner, &app);
    }

    // Reversal flip: fire `set_time_scale_factor(-1.0)` through the
    // SimContext trait on both runtimes at the same lockstep tick.
    // Runner forwards to `self.time.time_scale_factor = -1.0`; the
    // Bevy adapter writes the same field on the `SimulationTimeR`
    // resource so the next FixedUpdate's `time_advance_system` reads
    // the post-flip value. Mirrors the runner-side tier3 test's
    // `if i == reversal_idx + 1 { sim.time.time_scale_factor = -1.0; }`
    // guard — we just trigger the flip at our own truncated boundary
    // rather than the JEOD CSV's row 1001.
    let flip_t = records[PARITY_FORWARD_RECORDS].time;
    apply_flip_both::<astrodyn::Earth>(&mut runner, &mut app, &source_entities, &body_entities);
    // Verify the flip landed identically on both sides before the
    // next integration step exercises the reversed dynamic dt.
    assert_time_scale_bits_eq("post-flip", flip_t, &runner, &app);

    // Phase 2: reverse propagation across the same record cadence
    // but walking the CSV's reverse-phase rows (whose `sys.exec.out.time`
    // also increments by 60 s — only the dynamic `tai_seconds` reverses).
    // `step_to_record` uses `runner.elapsed()` (== simtime) which the
    // CSV's column 0 mirrors, so the same step_until shape works for
    // both phases.
    let reverse_phase = records
        .iter()
        .skip(PARITY_FORWARD_RECORDS + 1)
        .take(PARITY_FORWARD_RECORDS);
    for rec in reverse_phase {
        step_to_record(rec.time, &mut runner, &mut app);
        assert_trans_bits_eq("rev", rec.time, &runner, &app, vehicle_entity);
        assert_rot_bits_eq("rev", rec.time, &runner, &app, vehicle_entity);
        assert_time_scale_bits_eq("rev", rec.time, &runner, &app);
    }
}

/// Step both runtimes from their current `simtime` to the next
/// record-cadence checkpoint by the same integer number of `dt`
/// ticks. The runner's `simtime` is the authoritative source of truth
/// — the Bevy adapter reads the bit-exact f64 `dt` from
/// `IntegrationDtR`, so the two runtimes' tick counts match by
/// construction. Asserts the tick count is positive (a record-cadence
/// regression would otherwise silently zero-step and feed stale state
/// into the per-record assertion).
fn step_to_record(target_t: f64, runner: &mut Simulation, app: &mut App) {
    let dt_steps = ((target_t - runner.elapsed()) / DT).round() as usize;
    assert!(
        dt_steps > 0,
        "bevy_parity_time_reversal: target t={target_t} is not strictly after sim time {} \
         (dt={DT}); CSV record cadence must align with the integrator tick.",
        runner.elapsed()
    );
    for _ in 0..dt_steps {
        runner.step().expect("runner step failed");
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }
    // Alignment witness: `simtime` always advances forward regardless
    // of `time_scale_factor`, so the runner's `elapsed()` is the right
    // oracle for the checkpoint boundary. The bound is one milli-dt
    // (matches the parity helper's main loop) — anything larger means
    // the record-time / dt division rounded to the wrong tick count,
    // which is a recipe-construction bug.
    let elapsed_err = (runner.elapsed() - target_t).abs();
    assert!(
        elapsed_err <= 1e-3 * DT,
        "bevy_parity_time_reversal: runner elapsed {:.9}s drifted from target \
         {:.9}s by {:.3e}s after {} ticks of dt={DT}",
        runner.elapsed(),
        target_t,
        elapsed_err,
        dt_steps,
    );
}

/// Fire `set_time_scale_factor(-1.0)` through the `SimContext` trait on
/// both runtimes. Mirrors the `apollo_trajectory` parity wrapper's
/// `apply_event_both` shape — the runner-side implementation forwards
/// to `self.time.time_scale_factor = factor`; the Bevy adapter writes
/// the same field on the `SimulationTimeR` resource so the next
/// FixedUpdate's `time_advance_system` reads the post-flip value.
fn apply_flip_both<P: astrodyn::Planet>(
    runner: &mut Simulation,
    app: &mut App,
    source_entities: &[Entity],
    body_entities: &[Entity],
) {
    {
        let ctx: &mut dyn SimContext = runner;
        ctx.set_time_scale_factor(-1.0);
    }
    {
        let world = app.world_mut();
        let mut ctx = BevySimContext::<P>::new(world, source_entities, body_entities);
        ctx.set_time_scale_factor(-1.0);
    }
}

fn assert_trans_bits_eq(label: &str, t: f64, runner: &Simulation, app: &App, vehicle: Entity) {
    let runner_body = runner.body(0);
    let r_pos = runner_body.trans.position.raw_si();
    let r_vel = runner_body.trans.velocity.raw_si();
    let bevy = app
        .world()
        .get::<TranslationalStateC<astrodyn::Earth>>(vehicle)
        .expect("vehicle carries TranslationalStateC<Earth>")
        .0;
    let b_pos = bevy.position.raw_si();
    let b_vel = bevy.velocity.raw_si();
    for i in 0..3 {
        assert!(
            r_pos[i].to_bits() == b_pos[i].to_bits(),
            "bevy_parity_time_reversal {label} t={t:.6}s position[{i}] diverged: \
             runner={r_v} (bits={r_b:#018x}) bevy={b_v} (bits={b_b:#018x})",
            r_v = r_pos[i],
            r_b = r_pos[i].to_bits(),
            b_v = b_pos[i],
            b_b = b_pos[i].to_bits(),
        );
        assert!(
            r_vel[i].to_bits() == b_vel[i].to_bits(),
            "bevy_parity_time_reversal {label} t={t:.6}s velocity[{i}] diverged: \
             runner={r_v} (bits={r_b:#018x}) bevy={b_v} (bits={b_b:#018x})",
            r_v = r_vel[i],
            r_b = r_vel[i].to_bits(),
            b_v = b_vel[i],
            b_b = b_vel[i].to_bits(),
        );
    }
}

fn assert_rot_bits_eq(label: &str, t: f64, runner: &Simulation, app: &App, vehicle: Entity) {
    let runner_body = runner.body(0);
    let runner_rot = runner_body
        .rot
        .as_ref()
        .expect("runner body carries RotationalState");
    let r_untyped = typed_bridge::rot_typed_to_raw(runner_rot);
    let bevy_rot = app
        .world()
        .get::<RotationalStateC>(vehicle)
        .expect("vehicle carries RotationalStateC")
        .0;
    let b_untyped = typed_bridge::rot_typed_to_raw(&bevy_rot);
    for i in 0..4 {
        let r_v = r_untyped.quaternion.data[i];
        let b_v = b_untyped.quaternion.data[i];
        assert!(
            r_v.to_bits() == b_v.to_bits(),
            "bevy_parity_time_reversal {label} t={t:.6}s quat[{i}] diverged: \
             runner={r_v} (bits={r_b:#018x}) bevy={b_v} (bits={b_b:#018x})",
            r_b = r_v.to_bits(),
            b_b = b_v.to_bits(),
        );
    }
    for i in 0..3 {
        let r_v = r_untyped.ang_vel_body[i];
        let b_v = b_untyped.ang_vel_body[i];
        assert!(
            r_v.to_bits() == b_v.to_bits(),
            "bevy_parity_time_reversal {label} t={t:.6}s ang_vel[{i}] diverged: \
             runner={r_v} (bits={r_b:#018x}) bevy={b_v} (bits={b_b:#018x})",
            r_b = r_v.to_bits(),
            b_b = b_v.to_bits(),
        );
    }
}

/// Bit-identity on the time-scale-related `SimulationTime` fields:
/// `time_scale_factor` (load-bearing for this scenario) plus `tai_tjt`
/// / `tai_seconds` / `simtime` so a Bevy-side time-update regression
/// surfaces here even if the body-state assertions stayed in lockstep
/// (e.g. a tsf write that drifted the dynamic time scales while the
/// integrator picked the right `integ_dt` anyway).
fn assert_time_scale_bits_eq(label: &str, t: f64, runner: &Simulation, app: &App) {
    let bevy = &app.world().resource::<SimulationTimeR>().0;
    let runner_time = &runner.time;
    fn bits_eq(label: &str, t: f64, field: &str, r: f64, b: f64) {
        assert!(
            r.to_bits() == b.to_bits(),
            "bevy_parity_time_reversal {label} t={t:.6}s {field} diverged: \
             runner={r} (bits={r_b:#018x}) bevy={b} (bits={b_b:#018x})",
            r_b = r.to_bits(),
            b_b = b.to_bits(),
        );
    }
    bits_eq(
        label,
        t,
        "scale_factor",
        runner_time.scale_factor(),
        bevy.scale_factor(),
    );
    bits_eq(label, t, "tai_tjt", runner_time.tai_tjt, bevy.tai_tjt);
    bits_eq(
        label,
        t,
        "tai_seconds",
        runner_time.tai_seconds,
        bevy.tai_seconds,
    );
    bits_eq(label, t, "simtime", runner_time.simtime, bevy.simtime);
}
