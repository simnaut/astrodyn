// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: SIM_Apollo trajectory cross-validation through 9 detaches + 2 attaches.

#![allow(
    clippy::float_cmp,
    reason = "Tier 3 tests assert bit-exact recovery of literal-built / analytic state values"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Tier 3 step counts and indices fit exactly in f64 mantissa and usize"
)]
//!
//! Reproduces JEOD's `sims/SIM_Apollo/SET_test/RUN_test` 12-second
//! initialization-only sim and cross-validates `cm_dyn`'s `core_body`
//! trajectory against the reference CSV. The sim has 11 scheduled
//! `add_read` events at integer seconds — 9 detaches and 2 attaches.
//! The full event sequence is applied to our mass tree (so the pipeline
//! exercises all 11 events end-to-end) via the runner's
//! `Simulation::detach_subtree` and `Simulation::attach_subtree_aligned`
//! routed through the shared
//! [`run_verification::sim_apollo_trajectory`](astrodyn_verif_jeod::run_verification::sim_apollo_trajectory)
//! recipe (the same module the Bevy parity wrapper at
//! `crates/astrodyn_verif_parity/tests/bevy_parity_apollo_trajectory.rs`
//! consumes — keeping both runtimes byte-for-byte in lockstep).
//! `attach_subtree_aligned` ports JEOD's `DynBody::attach_child`
//! momentum-conservation algorithm into
//! [`astrodyn_dynamics::attach::combine_states_at_attach`], with full
//! struct↔body-frame distinctions per `MassProperties::t_parent_this`
//! (set per body from `Modified_data/mass/*.py:pt_orientation` —
//! `yaw_180` for CM/LES/DM/LM, identity for SM/S1/S2/S3).
//!
//! Trajectory diffs are asserted through the full 12 s sim — all 11
//! attach/detach events execute and the CSM `core_body` trajectory is
//! compared against JEOD's recorded reference at every 0.1 s sample.
//! Residuals are at numerical-precision limits everywhere: ≲ 7 µm
//! position, ≲ 3 µm/s velocity, ≲ 4 µrad attitude, ≲ 14 µrad/s ang_vel.
//! That level of agreement holds across both the t=6 SM→CM attach
//! (which matches JEOD's logged composite-body angular velocity of
//! −1.7207 rad/s exactly) and the t=9 AttLmCm2 / t=10 DetLm3 sequence
//! that previously produced "larger rotation drift" before the
//! `composite_body`-integration refactor (commit `bd279c2`) and the
//! `step_ballistic` quaternion-multiply-order fix (routed through
//! `BodyAttitude::advance_under_body_rate` after issue #252).
//!
//! ### Scope
//!
//! - Initial state: from `apollo_trajectory.csv` row 0 (= JEOD's
//!   `cm_dyn.dyn_body.core_body.state` after launch_stack assembly,
//!   in Earth.inertial). Equivalent to the LEO LVLH-aligned state
//!   from `Modified_data/state/sv_leo_lvlh.py` shifted by the
//!   structure-to-composite offset.
//! - Epoch: 1969-07-16 13:44:00 UTC (Apollo 11 launch date), with
//!   `Leap_Second.dat` overridden to `TAI-UTC = 4.2 s` (per
//!   `Modified_data/date_n_time/UTC_16Jul1969.py`) and
//!   `UT1-TAI = 0.0115221 - 4.2 s`.
//! - Physics: 8x8 GGM05C Earth (RNP rotation), spherical Moon, spherical
//!   Sun, RK4 at `dt = 0.02 s` (DYNAMICS = 50 Hz from `S_define:72`).
//! - Mass tree: full 8-body Apollo stack (S1, S2, S3, LES, CM, SM, LM,
//!   DM) assembled via launch_stack, then 11 attach/detach events at
//!   `t = 1..11 s` per `RUN_test/input.py`.
//!
//! Mass-tree composite property validation at each phase is covered by
//! `crates/astrodyn_dynamics/tests/tier3_apollo_mass_tree.rs`; this test
//! complements it by exercising the full `Simulation::step()` pipeline
//! end-to-end through the same event sequence.

use astrodyn::JeodQuat;
use astrodyn_runner::{Simulation, SimulationBuilderExt};
use astrodyn_verif_jeod::apollo_truth::{
    load_apollo_attach_truth, nearest_truth_at, ApolloTruthError, ApolloTruthRow,
};
use astrodyn_verif_jeod::crossval::{CrossvalReport, StateLog};
use astrodyn_verif_jeod::run_verification::sim_apollo_trajectory::{
    apollo_trajectory_builder, apply_event, setup_apollo_arena, ApolloTopology, Event, EVENTS,
    SIM_DURATION_S,
};
use astrodyn_verif_jeod::verification::SimContext;
use glam::DVec3;
use std::path::PathBuf;

// JEOD constants and helpers live in the shared recipe; only the
// per-test bits (CSV loader for ALL rows, trajectory-validation window,
// LM diagnostic) stay inline.

const DT: f64 = astrodyn_verif_jeod::run_verification::sim_apollo_trajectory::DT;

/// Trajectory comparison window: full 12 s sim. Asserts every 0.1 s
/// sample through all 11 attach/detach events (5 stage detaches, the
/// t=6 SM→CM attach, the t=7 LM detach, t=8 DM detach, t=9 LM
/// re-attach, t=10 LM detach, t=11 SM detach). See the test header
/// for the residual budget.
const TRAJECTORY_VALIDATION_END_S: f64 = 12.0;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data")
}

/// Apollo CSV reference state at one logged timestamp.
struct ApolloRef {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    quaternion: JeodQuat,
    ang_vel_body: DVec3,
}

fn load_apollo_reference() -> Vec<ApolloRef> {
    let csv_path = test_data_dir().join("apollo_trajectory.csv");
    assert!(
        csv_path.exists(),
        "apollo_trajectory.csv missing at {}. Generate with: cargo xtask regenerate-tier3",
        csv_path.display()
    );
    let content = std::fs::read_to_string(&csv_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", csv_path.display()));

    // Column layout (per APOLLO_SNIPPET in trick/generate_references.sh):
    //   0 time
    //   1 pos[0], 2 vel[0], 3 pos[1], 4 vel[1], 5 pos[2], 6 vel[2]
    //   7 q.scalar, 8-10 q.vec[0..2], 11-13 ang_vel[0..2]
    let mut out = Vec::new();
    // Parse positionally and panic on any column-count or parse error
    // rather than silently skipping rows: this is verification data,
    // and a corrupted reference trajectory should fail loudly, not
    // shift column indices and produce subtly-wrong test results.
    for (row_idx, line) in content.lines().skip(1).enumerate() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        // CSV row index in the source file (1-indexed; +2 to account
        // for skipping the header).
        let csv_row = row_idx + 2;
        assert_eq!(
            fields.len(),
            14,
            "{}:{csv_row} apollo_trajectory.csv: expected 14 columns, got {}: {line:?}",
            csv_path.display(),
            fields.len(),
        );
        let parse = |col: usize| -> f64 {
            fields[col].parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "{}:{csv_row} apollo_trajectory.csv: failed to parse column {col} ({:?}): {e}",
                    csv_path.display(),
                    fields[col],
                )
            })
        };
        out.push(ApolloRef {
            time: parse(0),
            position: DVec3::new(parse(1), parse(3), parse(5)),
            velocity: DVec3::new(parse(2), parse(4), parse(6)),
            // JEOD scalar-first [q0,q1,q2,q3] — store with same convention.
            quaternion: JeodQuat::new(parse(7), parse(8), parse(9), parse(10)),
            ang_vel_body: DVec3::new(parse(11), parse(12), parse(13)),
        });
    }
    out
}

// ── Test setup helper ────────────────────────────────────────────────

fn build_apollo_sim() -> (Simulation, ApolloTopology) {
    let handles = apollo_trajectory_builder();
    let mut sim = handles
        .builder
        .build()
        .expect("apollo simulation must validate");
    // `from_builder` allocates the integrated `cm` body's MassBodyId via
    // `register_in_mass_tree(0, "cm")`. Resolve it now so the arena
    // setup can register the 7 tree-only bodies + 14 mass points + 7
    // launch_stack attaches on top.
    let cm_id = sim
        .body_mass_id(0)
        .expect("cm body must be registered in mass tree by the builder");
    let tree = sim.mass_tree.as_mut().expect("mass tree was just created");
    let topology = setup_apollo_arena(tree, cm_id);

    // Sync the integrated cm body's mass from the fully-assembled tree
    // composite, then flip its `body.trans` from `core_body` to
    // `composite_body` (the integration variable).
    sim.sync_body_mass_from_tree(0);
    sim.convert_body_trans_core_to_composite(0);

    (sim, topology)
}

// ── Test ─────────────────────────────────────────────────────────────

// non-recipe: SIM_Apollo's launch-stack topology, JEOD English-unit
// per-body mass data, and 11-event detach/attach schedule are
// unique to this verification sim and not currently captured in any
// `astrodyn::recipes::scenarios::*` recipe. The JEOD-input.py defect
// (missing `set_vehicle_grav_controls(cm_dyn)` call) is patched at
// reference-CSV-regeneration time inside `trick/generate_references.sh`,
// not via any production-side workaround.
#[test]
fn tier3_sim_apollo_trajectory() {
    let csv = load_apollo_reference();
    assert!(
        (csv.last().unwrap().time - SIM_DURATION_S).abs() < 0.05,
        "apollo_trajectory.csv last row t={} disagrees with SIM_Apollo terminate_time={SIM_DURATION_S}",
        csv.last().unwrap().time
    );

    let (mut sim, topology) = build_apollo_sim();

    // Walk the simulation in 0.1-second log windows, applying the
    // mass-tree event at each integer-second boundary just before the
    // logging step that crosses it (matching Trick's add_read semantics:
    // the event fires at the start of the cycle that begins at t=N,
    // before the data record at t=N is written).
    let mut event_iter = EVENTS.iter().peekable();
    let mut our_log = Vec::with_capacity(csv.len());
    let mut ref_log = Vec::with_capacity(csv.len());
    let mut current_t = 0.0_f64;

    // Skip CSV row 0 (initial state — no integration yet).
    for reference in csv.iter().skip(1) {
        // Apply events in JEOD's order: Trick's `trick.add_read(t, ...)`
        // job fires at the END of the cycle ending at t — after the
        // integrator has advanced state to t. So step up to event_t
        // (current_t == event_t), THEN apply the event. Verified
        // empirically: at t=4 DetachS3, JEOD's lm.vel JUMPS by +0.110 m/s
        // between the t=3.999 sample (= state at end of cycle [3.96,
        // 3.98]) and the t=4.000 sample (= state at end of cycle [3.98,
        // 4.0]). That kick equals one ordinary integration step on cm
        // cascaded through the dyn-tree to lm via JEOD's propagate_state
        // — i.e., the cycle [3.98, 4.0] integrator ran with the PRE-
        // detach mass tree (lm still in cm's tree), THEN the detach
        // fired. See `BUG_A_REPORT.md`.
        while let Some(&&(event_t, event)) = event_iter.peek() {
            if event_t <= reference.time + 1e-9 && event_t > current_t + 1e-9 {
                // Step up to and including event_t, then apply.
                while current_t + 0.5 * DT < event_t {
                    sim.step().expect("step failed");
                    current_t += DT;
                }
                // Route the event through the shared recipe's SimContext
                // dispatch so the runner-vs-Bevy parity wrapper consumes
                // the same event-table and per-event arguments.
                let ctx: &mut dyn SimContext = &mut sim;
                apply_event(ctx, &topology, event);
                event_iter.next();
            } else {
                break;
            }
        }

        // Step up to the reference timestamp.
        while current_t + DT * 0.5 < reference.time {
            sim.step().expect("step failed");
            current_t += DT;
        }

        if reference.time > TRAJECTORY_VALIDATION_END_S + 1e-6 {
            // See module docs / TRAJECTORY_VALIDATION_END_S for why
            // later samples are skipped.
            continue;
        }

        let body = sim.body(0);
        // body.trans is the composite_body inertial integration state;
        // JEOD's reference CSV logs core_body, so derive it via the
        // mass tree (composite and core share body axes — only
        // position+velocity differ).
        let (core_position, core_velocity) = sim.body_core_inertial(0);
        our_log.push(StateLog {
            time: reference.time,
            position: Some(core_position),
            velocity: Some(core_velocity),
            acceleration: Some(body.trans_accel.raw_si()),
            quaternion: body
                .rot
                .as_ref()
                .map(|r| r.q_inertial_body.as_witness().inner().to_glam()),
            ang_vel: body.rot.as_ref().map(|r| r.ang_vel_body.raw_si()),
            ang_accel: body.rot_accel.map(|a| a.raw_si()),
        });
        ref_log.push(StateLog {
            time: reference.time,
            position: Some(reference.position),
            velocity: Some(reference.velocity),
            acceleration: None,
            quaternion: Some(reference.quaternion.to_glam()),
            ang_vel: Some(reference.ang_vel_body),
            ang_accel: None,
        });
    }
    assert!(!our_log.is_empty(), "trajectory log is empty");

    // Tooling-enforced cadence check: dt = 0.02 s and JEOD's CSV
    // samples at 0.1 s, so 0.1 / 0.02 = 5 — every reference row is an
    // integrator-output instant. If a future edit drifts either side
    // off the integer ratio, this fails loudly before the row loop
    // quietly compares against held off-cadence samples.
    CrossvalReport::assert_cadence_matches(&ref_log, DT, 1e-6);

    let report = CrossvalReport::compute("tier3_sim_apollo_trajectory", &our_log, &ref_log);
    report.write();

    // Tolerances per `tests/README.md` (5 % above observed max error).
    //
    // Window: full 12 s sim — every one of the 11 attach/detach events
    // is asserted end-to-end, including the t=6 SM→CM attach (whose
    // composite ang_vel matches JEOD's logged −1.7207 rad/s exactly),
    // the t=9 LM re-attach, and the t=10 LM detach. The closed-form
    // quaternion advance for detached subtrees routes through
    // `BodyAttitude::advance_under_body_rate` (issue #248 / PR #251 +
    // issue #252); fixing the multiply order on `step_ballistic`
    // removed the 1.708 mrad/s S3-attitude drift that had been
    // lever-armed up to
    // 16 mm at LM during the t=4 → t=5 free-fly. Residuals over the
    // full 12 s are now:
    //   - position:    ~7 µm / component
    //   - velocity:    ~2.5 µm/s / component
    //   - quat angle:  ~3.4 µrad
    //   - ang_vel:     ~14 µrad/s worst-component (body-Z, lever-armed
    //                  through the t=6 attach algorithm's ~4 mrad/s
    //                  body-X residue, which is sub-LSB on the input
    //                  cross-products and physically negligible).
    report.assert_position([6.90e-6, 2.50e-6, 5.27e-6]);
    report.assert_velocity([2.58e-6, 1.24e-6, 1.62e-6]);
    report.assert_quat_angle(3.59e-6);
    report.assert_ang_vel([2.29e-6, 1.19e-7, 1.46e-5]);
}

// ─── LM-state-vs-truth diagnostic ────────────────────────────────────
//
// Runs the same sim through the full 12 s and compares LM
// `composite_body` inertial state against `apollo_attach_truth.csv`
// (1 ms cadence) at every integration step plus right after each event.
// Output: stderr table highlighting the first sample to cross 1 mm,
// plus a per-step CSV under `target/tier3_crossval/` for offline
// analysis. Diagnostic only — does not assert tolerances. Marked
// `#[ignore]` because the truth CSV is gitignored and may be missing
// on a fresh clone.

const LM_DIAG_POSITION_TRIP_M: f64 = 1.0e-3;

#[derive(Clone)]
struct LmDiagSample {
    time: f64,
    /// Empty unless this row was captured immediately after an event applied.
    event_label: String,
    // ── LM (always present) ──
    err_pos: DVec3,
    err_vel: DVec3,
    err_quat_angle: f64,
    err_ang_vel: DVec3,
    our_pos: DVec3,
    truth_pos: DVec3,
    /// Raw LM ang_vel from the runner (chain-walked), expressed in body frame.
    our_ang_vel: DVec3,
    /// Raw LM ang_vel from JEOD's truth recorder, body frame.
    truth_ang_vel: DVec3,
    // ── S3 (Some only when truth CSV has s3 columns) ──
    /// `Some` when the truth row exposes `s3`; otherwise the recorder hasn't
    /// been regenerated with the s3 columns yet.
    s3_err_pos: Option<DVec3>,
    s3_err_vel: Option<DVec3>,
    s3_err_quat_angle: Option<f64>,
    s3_err_ang_vel: Option<DVec3>,
}

fn event_short_label(event: Event) -> &'static str {
    match event {
        Event::DetachS1 => "DetS1",
        Event::DetachS2 => "DetS2",
        Event::DetachLes => "DetLes",
        Event::DetachS3 => "DetS3",
        Event::DetachLm => "DetLm",
        Event::AttachLmCm => "AttLmCm",
        Event::DetachLm2 => "DetLm2",
        Event::DetachDm => "DetDm",
        Event::AttachLmCm2 => "AttLmCm2",
        Event::DetachLm3 => "DetLm3",
        Event::DetachSm => "DetSm",
    }
}

/// Walk up from CARGO_MANIFEST_DIR until we find Cargo.lock — that's the
/// workspace root. Mirrors the helper in `astrodyn_verif_jeod::crossval`.
fn workspace_target_tier3_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            break;
        }
        if !dir.pop() {
            dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            break;
        }
    }
    dir.join("target").join("tier3_crossval")
}

/// Quaternion angular distance: `2 · acos(|<q1, q2>|)`. Returns the
/// smaller of the two possible rotations (q and −q represent the same
/// rotation in JEOD's left-quat convention).
fn quat_angle_between(a: JeodQuat, b: JeodQuat) -> f64 {
    let av = a.vector();
    let bv = b.vector();
    let dot = a.scalar() * b.scalar() + av.x * bv.x + av.y * bv.y + av.z * bv.z;
    2.0 * dot.abs().clamp(-1.0, 1.0).acos()
}

fn capture_lm_diag(
    sim: &Simulation,
    topology: &ApolloTopology,
    truth_rows: &[ApolloTruthRow],
    time: f64,
    event_label: &str,
) -> LmDiagSample {
    let our = sim.subtree_composite_inertial(topology.lm);
    let truth = nearest_truth_at(truth_rows, time);
    let truth_quat = truth.lm.quaternion;

    // S3 comparison — only meaningful when the truth recorder logged s3.
    // Even when the truth row has no s3, we still walk our own simulation
    // for s3 so the function is total; the comparison is conditioned on
    // truth.s3 being Some.
    let our_s3 = sim.subtree_composite_inertial(topology.s3);
    let s3_err_pos = truth
        .s3
        .as_ref()
        .map(|s3| our_s3.trans.position - s3.position);
    let s3_err_vel = truth
        .s3
        .as_ref()
        .map(|s3| our_s3.trans.velocity - s3.velocity);
    let s3_err_quat_angle = truth
        .s3
        .as_ref()
        .map(|s3| quat_angle_between(our_s3.rot.q_parent_this, s3.quaternion));
    let s3_err_ang_vel = truth
        .s3
        .as_ref()
        .map(|s3| our_s3.rot.ang_vel_this - s3.ang_vel_body);

    LmDiagSample {
        time,
        event_label: event_label.to_string(),
        err_pos: our.trans.position - truth.lm.position,
        err_vel: our.trans.velocity - truth.lm.velocity,
        err_quat_angle: quat_angle_between(our.rot.q_parent_this, truth_quat),
        err_ang_vel: our.rot.ang_vel_this - truth.lm.ang_vel_body,
        our_pos: our.trans.position,
        truth_pos: truth.lm.position,
        our_ang_vel: our.rot.ang_vel_this,
        truth_ang_vel: truth.lm.ang_vel_body,
        s3_err_pos,
        s3_err_vel,
        s3_err_quat_angle,
        s3_err_ang_vel,
    }
}

/// Diagnostic (ignored by default): runs the full 12 s SIM_Apollo and
/// compares LM `composite_body` inertial state against
/// `apollo_attach_truth.csv` at every integration step and right after
/// each event. Output is a stderr table flagging the first sample whose
/// position error exceeds 1 mm, plus a per-step CSV at
/// `target/tier3_crossval/apollo_lm_state_vs_truth.csv`. The truth CSV
/// is gitignored — regenerate via `cargo xtask regenerate-tier3 --force`.
///
/// Run manually:
///   `cargo nextest run -p astrodyn_runner --test tier3_sim_apollo_trajectory \
///     tier3_sim_apollo_lm_state_vs_truth --run-ignored only`
/// or
///   `cargo test -p astrodyn_runner --test tier3_sim_apollo_trajectory \
///     tier3_sim_apollo_lm_state_vs_truth -- --ignored --nocapture`
#[test]
#[ignore]
fn tier3_sim_apollo_lm_state_vs_truth() {
    let truth_rows = match load_apollo_attach_truth(env!("CARGO_MANIFEST_DIR")) {
        Ok(rows) => rows,
        Err(ApolloTruthError::Missing { path }) => panic!(
            "{} missing — regenerate via `cargo xtask regenerate-tier3 --force` \
             (the attach_truth recorder is in APOLLO_SNIPPET in trick/generate_references.sh)",
            path.display()
        ),
        Err(e) => panic!("failed to load apollo_attach_truth.csv: {e}"),
    };
    eprintln!(
        "loaded {} truth rows spanning t = {:.6} .. {:.6}",
        truth_rows.len(),
        truth_rows.first().unwrap().time,
        truth_rows.last().unwrap().time
    );

    let (mut sim, topology) = build_apollo_sim();

    let mut event_iter = EVENTS.iter().peekable();
    let mut current_t = 0.0_f64;
    let mut samples: Vec<LmDiagSample> = Vec::new();

    samples.push(capture_lm_diag(
        &sim,
        &topology,
        &truth_rows,
        current_t,
        "init",
    ));

    let n_steps = (SIM_DURATION_S / DT).round() as usize;
    for _ in 0..n_steps {
        // Apply any events whose t is at or before current_t (matches
        // the trajectory test's JEOD-order semantics).
        let mut applied = String::new();
        while let Some(&&(event_t, event)) = event_iter.peek() {
            if event_t <= current_t + 1e-9 {
                let ctx: &mut dyn SimContext = &mut sim;
                apply_event(ctx, &topology, event);
                if !applied.is_empty() {
                    applied.push('+');
                }
                applied.push_str(event_short_label(event));
                event_iter.next();
            } else {
                break;
            }
        }
        if !applied.is_empty() {
            samples.push(capture_lm_diag(
                &sim,
                &topology,
                &truth_rows,
                current_t,
                &applied,
            ));
        }
        sim.step().expect("step failed");
        current_t += DT;
        samples.push(capture_lm_diag(&sim, &topology, &truth_rows, current_t, ""));
    }
    // Sweep any trailing events scheduled at current_t (none today, but
    // guard the loop for future schedule edits).
    while let Some(&&(event_t, event)) = event_iter.peek() {
        if event_t <= current_t + 1e-9 {
            let ctx: &mut dyn SimContext = &mut sim;
            apply_event(ctx, &topology, event);
            samples.push(capture_lm_diag(
                &sim,
                &topology,
                &truth_rows,
                current_t,
                event_short_label(event),
            ));
            event_iter.next();
        } else {
            break;
        }
    }

    // ── stderr summary ───────────────────────────────────────────────
    let first_breach = samples
        .iter()
        .find(|s| s.err_pos.length() > LM_DIAG_POSITION_TRIP_M);

    eprintln!();
    eprintln!("==========================================================");
    eprintln!("  LM composite_body vs apollo_attach_truth.csv");
    eprintln!(
        "  position trip threshold = {:.0e} m",
        LM_DIAG_POSITION_TRIP_M
    );
    eprintln!("==========================================================");
    eprintln!(
        "  total samples: {} ({} steps + initial + post-event captures)",
        samples.len(),
        n_steps
    );
    if let Some(s) = first_breach {
        eprintln!();
        eprintln!(
            "  FIRST POSITION BREACH at t = {:.6} s (event_label: {:?})",
            s.time, s.event_label
        );
        eprintln!(
            "    err_pos = [{:>13.6e} {:>13.6e} {:>13.6e}]  |.| = {:.6e} m",
            s.err_pos.x,
            s.err_pos.y,
            s.err_pos.z,
            s.err_pos.length()
        );
        eprintln!(
            "    err_vel = [{:>13.6e} {:>13.6e} {:>13.6e}]  |.| = {:.6e} m/s",
            s.err_vel.x,
            s.err_vel.y,
            s.err_vel.z,
            s.err_vel.length()
        );
        eprintln!("    err_quat_angle = {:.6e} rad", s.err_quat_angle);
        eprintln!(
            "    err_ang_vel = [{:>13.6e} {:>13.6e} {:>13.6e}]  |.| = {:.6e} rad/s",
            s.err_ang_vel.x,
            s.err_ang_vel.y,
            s.err_ang_vel.z,
            s.err_ang_vel.length()
        );
    } else {
        eprintln!();
        eprintln!(
            "  no position breach — max |err_pos| = {:.6e} m",
            samples
                .iter()
                .map(|s| s.err_pos.length())
                .fold(0.0_f64, f64::max)
        );
    }

    // ── per-event-boundary headline (every event, regardless of trip) ─
    let any_s3 = samples.iter().any(|s| s.s3_err_pos.is_some());
    eprintln!();
    eprintln!("  ─── per-event LM error snapshots ─────────────────────");
    eprintln!(
        "  {:>10}  {:>10}  {:>13}  {:>13}  {:>13}  {:>13}",
        "t (s)", "event", "|err_pos| m", "|err_vel| m/s", "dq_ang rad", "|dω| rad/s"
    );
    for s in samples.iter().filter(|s| !s.event_label.is_empty()) {
        eprintln!(
            "  {:>10.6}  {:>10}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}",
            s.time,
            s.event_label,
            s.err_pos.length(),
            s.err_vel.length(),
            s.err_quat_angle,
            s.err_ang_vel.length()
        );
    }

    if any_s3 {
        eprintln!();
        eprintln!("  ─── per-event S3 error snapshots ─────────────────────");
        eprintln!(
            "  {:>10}  {:>10}  {:>13}  {:>13}  {:>13}  {:>13}",
            "t (s)", "event", "|err_pos| m", "|err_vel| m/s", "dq_ang rad", "|dω| rad/s"
        );
        for s in samples.iter().filter(|s| !s.event_label.is_empty()) {
            match (
                s.s3_err_pos,
                s.s3_err_vel,
                s.s3_err_quat_angle,
                s.s3_err_ang_vel,
            ) {
                (Some(ep), Some(ev), Some(eq), Some(ew)) => eprintln!(
                    "  {:>10.6}  {:>10}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}",
                    s.time,
                    s.event_label,
                    ep.length(),
                    ev.length(),
                    eq,
                    ew.length()
                ),
                _ => eprintln!(
                    "  {:>10.6}  {:>10}  (truth row at this time has no s3 columns)",
                    s.time, s.event_label
                ),
            }
        }
    } else {
        eprintln!();
        eprintln!(
            "  S3-vs-truth comparison skipped — truth CSV has no s3_dyn columns. \
             Regenerate via `cargo xtask regenerate-tier3 --force` after pulling \
             the recorder change in `trick/generate_references.sh`."
        );
    }

    // Sanity-check the err_ang_vel = 0 observation by dumping raw values
    // at one mid-window sample. If the bits really are equal, both rows
    // print the same numbers.
    if let Some(probe) = samples
        .iter()
        .find(|s| (s.time - 4.5).abs() < 1e-6 && s.event_label.is_empty())
    {
        eprintln!();
        eprintln!("  ─── ang_vel sanity-check at t = 4.500 ────────────────");
        eprintln!(
            "    our   ang_vel = [{:>22.16} {:>22.16} {:>22.16}]",
            probe.our_ang_vel.x, probe.our_ang_vel.y, probe.our_ang_vel.z
        );
        eprintln!(
            "    truth ang_vel = [{:>22.16} {:>22.16} {:>22.16}]",
            probe.truth_ang_vel.x, probe.truth_ang_vel.y, probe.truth_ang_vel.z
        );
        eprintln!(
            "    raw bit-diff  = [{:>+22.16e} {:>+22.16e} {:>+22.16e}]",
            probe.err_ang_vel.x, probe.err_ang_vel.y, probe.err_ang_vel.z
        );
    }
    eprintln!("==========================================================");

    // ── per-step CSV for offline analysis ────────────────────────────
    let out_dir = workspace_target_tier3_dir();
    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("create_dir_all {}: {e}", out_dir.display()));
    let out_path = out_dir.join("apollo_lm_state_vs_truth.csv");
    let mut out = String::with_capacity(samples.len() * 200);
    out.push_str(
        "time,event,err_pos_norm_m,err_pos_x,err_pos_y,err_pos_z,\
         err_vel_norm_mps,err_vel_x,err_vel_y,err_vel_z,\
         err_quat_angle_rad,err_ang_vel_norm_rps,err_ang_vel_x,err_ang_vel_y,err_ang_vel_z,\
         our_pos_x,our_pos_y,our_pos_z,truth_pos_x,truth_pos_y,truth_pos_z,\
         our_ang_vel_x,our_ang_vel_y,our_ang_vel_z,\
         truth_ang_vel_x,truth_ang_vel_y,truth_ang_vel_z,\
         s3_err_pos_norm_m,s3_err_vel_norm_mps,s3_err_quat_angle_rad,s3_err_ang_vel_norm_rps\n",
    );
    fn fmt_opt_norm(v: Option<DVec3>) -> String {
        v.map(|x| format!("{:.9e}", x.length())).unwrap_or_default()
    }
    fn fmt_opt_f64(v: Option<f64>) -> String {
        v.map(|x| format!("{:.9e}", x)).unwrap_or_default()
    }
    for s in &samples {
        out.push_str(&format!(
            "{:.6},{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {},{},{},{}\n",
            s.time,
            s.event_label,
            s.err_pos.length(),
            s.err_pos.x,
            s.err_pos.y,
            s.err_pos.z,
            s.err_vel.length(),
            s.err_vel.x,
            s.err_vel.y,
            s.err_vel.z,
            s.err_quat_angle,
            s.err_ang_vel.length(),
            s.err_ang_vel.x,
            s.err_ang_vel.y,
            s.err_ang_vel.z,
            s.our_pos.x,
            s.our_pos.y,
            s.our_pos.z,
            s.truth_pos.x,
            s.truth_pos.y,
            s.truth_pos.z,
            s.our_ang_vel.x,
            s.our_ang_vel.y,
            s.our_ang_vel.z,
            s.truth_ang_vel.x,
            s.truth_ang_vel.y,
            s.truth_ang_vel.z,
            fmt_opt_norm(s.s3_err_pos),
            fmt_opt_norm(s.s3_err_vel),
            fmt_opt_f64(s.s3_err_quat_angle),
            fmt_opt_norm(s.s3_err_ang_vel),
        ));
    }
    std::fs::write(&out_path, out).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
    eprintln!(
        "  per-step trace: {} ({} rows)",
        out_path.display(),
        samples.len()
    );
}
