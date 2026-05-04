//! Tier 3: SIM_verif_attach_detach RUN_simple_attach_detach end-to-end
//! through the production [`Simulation::attach`] / [`Simulation::detach`]
//! API (the runner-side equivalent of `bevy_jeod`'s
//! `AttachEvent`/`DetachEvent` surface).
//!
//! Distinct from `tier3_sim_kinematic_propagation` (issue #294 / PR #295)
//! which validates the kernel-vs-CSV invariant during the attached
//! window and the runner-internal kernel self-consistency, but
//! deliberately defers absolute-trajectory cross-validation against
//! JEOD's CSV during the attached and post-detach windows. That defer
//! existed because [`Simulation::attach`] / [`Simulation::detach`] did
//! not yet run JEOD's [`combine_states_at_attach`] kernel — which
//! landed via PR #307 (closing issue #297).
//!
//! With the combine wiring in place, this test now exercises the
//! production attach/detach API end-to-end through `Simulation::step()`
//! and cross-validates **all three windows** of JEOD's
//! `RUN_simple_attach_detach` against the reference CSV
//! (`kinematic_propagation_simple_kinematic_propagation_state.csv`):
//!
//! 1. **Pre-attach** `t ∈ [0, 10)`: veh1 + veh2 + veh3 ballistic
//!    trajectories. Same coverage as the kinematic-propagation test —
//!    included here so the trajectory test stands on its own.
//! 2. **Attached** `t ∈ [10, 20)`: veh1 attached to veh2 via
//!    `Simulation::attach(v1, v2, offset, t_pc)`. The runner's
//!    integrated state on veh2 (the tree root after attach) is
//!    cross-validated against JEOD's `veh2.composite_body` CSV; veh1's
//!    derived kinematic-child state (written through by
//!    `propagate_kinematic_state` each step) is cross-validated against
//!    JEOD's `veh1.composite_body` CSV.
//! 3. **Post-detach** `t ∈ [20, 30)`: veh1 detached via
//!    `Simulation::detach(v1)`. Both veh1 and veh2 carry their own
//!    integrated state again; each is cross-validated against the JEOD
//!    CSV. veh3 (untouched throughout) is also cross-validated end-to-
//!    end as a free-flying ballistic baseline.
//!
//! The validation stops at `t = 30` because past that point JEOD's
//! input.py issues `set_attitude_rate`, `attach_to_frame`, and
//! reference-frame-attach commands that exercise different physics
//! (frame-attach + dynamic-body-action rate changes) outside this PR's
//! scope. See `tier3_sim_kinematic_propagation`'s "What is **not**
//! validated" section for the full deferred-dynamics list — same
//! deferrals apply here.
//!
//! # JEOD source data ingested
//!
//! Initial conditions only — never intermediate-step CSV values. The
//! attach geometry (offset + rotation) and per-vehicle initial state
//! both come from `Modified_data/veh{1,2,3}.py` and the named
//! `BodyAttachAligned` invocation in `attach_detach.py`. The CSV drives
//! only the comparison.
//!
//! [`combine_states_at_attach`]: jeod_dynamics::combine_states_at_attach

use std::path::PathBuf;

use glam::{DMat3, DVec3};
use jeod_dynamics::{IntegratorType, MassProperties};
use jeod_runner::Simulation;
use jeod_sim::{
    GravityControls, JeodQuat, RotationalState, SimulationTime, TranslationalState, VehicleConfig,
};
use jeod_test_data::crossval::CrossvalReport;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data")
}

/// One row of the `kinematic_propagation_state` CSV: time + per-vehicle
/// composite-body state for veh1, veh2, veh3 (39 + 1 columns).
#[derive(Debug, Clone)]
struct StateRow {
    time: f64,
    veh: [VehSnapshot; 3],
}

#[derive(Debug, Clone, Copy)]
struct VehSnapshot {
    position: DVec3,
    velocity: DVec3,
    /// JEOD scalar-first quaternion `[scalar, v0, v1, v2]`.
    quaternion: JeodQuat,
    /// `composite_body.state.rot.ang_vel_this`: angular velocity in
    /// the composite-body frame (rad/s).
    ang_vel_body: DVec3,
}

fn load_csv(filename: &str) -> Vec<StateRow> {
    let path = test_data_dir().join(filename);
    assert!(
        path.exists(),
        "JEOD reference data not found at {}.\n\
         Generate with:\n\
         cargo xtask regenerate-tier3\n\
         (or the equivalent Docker invocation — see CLAUDE.md \"Generating \
         Tier 3 Reference Data (Docker)\"). The CSV is produced by the \
         `kinematic_propagation_state` log group already wired into \
         `trick/generate_references.sh`.",
        path.display()
    );
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

    let mut rows = Vec::new();
    for (idx, line) in content.lines().skip(1).enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<&str> = trimmed.split(',').map(str::trim).collect();
        // 1 (time) + 3 vehicles × 13 fields each = 40
        assert_eq!(
            fields.len(),
            40,
            "CSV {} line {}: expected 40 columns, found {}",
            path.display(),
            idx + 2,
            fields.len(),
        );
        let parse = |col: usize| -> f64 {
            fields[col].parse().unwrap_or_else(|e| {
                panic!(
                    "CSV {} line {} col {}: invalid f64 {:?}: {e}",
                    path.display(),
                    idx + 2,
                    col,
                    fields[col]
                )
            })
        };
        let mut veh = [VehSnapshot {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            quaternion: JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]),
            ang_vel_body: DVec3::ZERO,
        }; 3];
        for (i, snap) in veh.iter_mut().enumerate() {
            let base = 1 + i * 13;
            snap.position = DVec3::new(parse(base), parse(base + 1), parse(base + 2));
            snap.velocity = DVec3::new(parse(base + 3), parse(base + 4), parse(base + 5));
            snap.quaternion = JeodQuat::from_array([
                parse(base + 6),
                parse(base + 7),
                parse(base + 8),
                parse(base + 9),
            ]);
            snap.ang_vel_body = DVec3::new(parse(base + 10), parse(base + 11), parse(base + 12));
        }
        rows.push(StateRow {
            time: parse(0),
            veh,
        });
    }
    assert!(
        !rows.is_empty(),
        "CSV {} contained no data rows",
        path.display()
    );
    rows
}

// ════════════════════════════════════════════════════════════════════
// Initial conditions, all from JEOD source files
// (Modified_data/veh{1,2,3}.py and attach_detach.py).
// ════════════════════════════════════════════════════════════════════

/// Time of `BodyAttachAligned veh1.attach_to_2`, from
/// `SET_test/RUN_simple_attach_detach/input.py:24`.
const ATTACH_TIME: f64 = 10.0;
/// Time of `BodyDetach veh1.detach_from_2`,
/// `SET_test/RUN_simple_attach_detach/input.py:25`.
const DETACH_TIME: f64 = 20.0;
/// Past this time the JEOD run starts firing reference-frame attaches
/// and rate-changes (`set_attitude_rate`, `attach_to_frame`, etc. —
/// `RUN_simple_attach_detach/input.py:27-39`). Reference-frame
/// attaches and dynamic-body-action rate changes are deferred-dynamics
/// scope and are not validated in this test.
const FRAME_ATTACH_PHASE_START: f64 = 30.0;

/// Veh1: mass=1.0, properties.position=(5, 0, 0), inertia=10·I
/// (`Modified_data/veh1.py:12-16`). For an atomic body the mass tree's
/// `composite_properties.position` equals the core's, so the CoM in
/// struct frame is (5, 0, 0).
fn veh1_mass() -> MassProperties {
    MassProperties::with_inertia(
        1.0,
        DMat3::from_diagonal(DVec3::splat(10.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh2_mass() -> MassProperties {
    MassProperties::with_inertia(
        2.0,
        DMat3::from_diagonal(DVec3::splat(20.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh3_mass() -> MassProperties {
    MassProperties::with_inertia(
        3.0,
        DMat3::from_diagonal(DVec3::splat(30.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh1_initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(-5.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 1.0, 0.0),
    }
}

fn veh1_initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]),
        ang_vel_body: DVec3::ZERO,
    }
}

fn veh2_initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(5.0, 10.0, 0.0),
        velocity: DVec3::ZERO,
    }
}

fn veh2_initial_rot() -> RotationalState {
    // JEOD `Modified_data/veh2.py:49`:
    //   `veh2.rot_init.orientation.euler_angles = [ -2.0, 0.0, 0.0 ]`
    // No `trick.attach_units("degree", ...)` wrapper, so the value is
    // in JEOD's default units for euler-angle fields: RADIANS, not
    // degrees. The CSV's `Q_parent_this.scalar=0.5403` at t=0 confirms
    // (cos(1.0)=0.5403 ⇒ half-angle 1.0 rad ⇒ θ=2.0 rad — matches
    // |yaw|=2.0 rad, with the sign flipped by JEOD's quaternion
    // convention `qv = -sin(θ/2)·axis`).
    let yaw_rad = -2.0_f64;
    let q = JeodQuat::left_quat_from_eigen_rotation(yaw_rad, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.0, 0.0, 0.2),
    }
}

fn veh3_initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(0.063, 13.787, -25.0),
        velocity: DVec3::new(0.0, 0.0, 1.0),
    }
}

fn veh3_initial_rot() -> RotationalState {
    // Same convention as `veh2_initial_rot`: euler angles in radians.
    let yaw_rad = -15.8_f64;
    let q = JeodQuat::left_quat_from_eigen_rotation(yaw_rad, DVec3::Z);
    // `Modified_data/veh3.py:50` sets `ang_velocity = [0, 0, 1.0]`,
    // but `RUN_simple_attach_detach/input.py:19` overrides it back to
    // zero before the run starts. Matching the override here keeps
    // initial conditions JEOD-source-faithful.
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::ZERO,
    }
}

/// Build a fresh runner sim configured for the simple attach-detach
/// scenario. Returns the simulation plus the integer body indices for
/// (veh1, veh2, veh3).
///
/// Empty space — no gravity sources, no central body — matching JEOD's
/// `EphemerisMode_EmptySpace` config. RK4 integrator on every body.
fn build_sim() -> (Simulation, usize, usize, usize) {
    let dt = 0.1;
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let v1 = sim.add_body(VehicleConfig {
        trans: veh1_initial_trans(),
        rot: Some(veh1_initial_rot()),
        mass: Some(veh1_mass()),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let v2 = sim.add_body(VehicleConfig {
        trans: veh2_initial_trans(),
        rot: Some(veh2_initial_rot()),
        mass: Some(veh2_mass()),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let v3 = sim.add_body(VehicleConfig {
        trans: veh3_initial_trans(),
        rot: Some(veh3_initial_rot()),
        mass: Some(veh3_mass()),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    sim.add_body_to_tree(v1, "veh1");
    sim.add_body_to_tree(v2, "veh2");
    sim.add_body_to_tree(v3, "veh3");
    (sim, v1, v2, v3)
}

/// JEOD's `BodyAttachAligned veh1.attach_to_2` resolves to a
/// structural-frame `(offset, T_parent_child)` pair via JEOD's
/// named-point attach algorithm
/// (`models/dynamics/mass/src/mass_attach.cc:67-137`). Both arguments
/// below are derived from JEOD source files only:
///
/// - `veh1.node12` lives at `(10, 0, 0)` in `veh1.struct` with identity
///   orientation (`Modified_data/veh1.py:21-23`).
/// - `veh2.node21` lives at `(0, 0, 0)` in `veh2.struct` with
///   YPR(180°, 0, 0) — i.e. `T_pstr_node21 = R(180°, Z)`
///   (`Modified_data/veh2.py:18-26`).
/// - JEOD's `attach_to` inserts a fixed 180° yaw between attach
///   points (`mass_attach.cc:111-117`).
///
/// Composing the chain
/// (veh2.struct → veh2.node21 → 180° flip → veh1.node12 → veh1.struct)
/// collapses to identity rotation and offset `(-10, 0, 0)`:
/// veh1's `node12` (at +10x in veh1) is the attach point, the
/// 180°-yaw-pair flip cancels, and veh2's `node21` is at the origin.
fn simple_attach_offset_and_rotation() -> (DVec3, DMat3) {
    (DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY)
}

/// Quaternion angle error: `2 · acos(|q_a · q_b|)`.
fn quat_angle_err(a: JeodQuat, b: JeodQuat) -> f64 {
    let dot = (a.scalar() * b.scalar() + a.vector().dot(b.vector()))
        .abs()
        .clamp(-1.0, 1.0);
    2.0 * dot.acos()
}

// ════════════════════════════════════════════════════════════════════
// Tolerances. Per CLAUDE.md "Cross-validation tolerances", set to ~5%
// above observed max error per component. Values come from the JSON
// report this test writes to `target/tier3_crossval/
// tier3_sim_attach_detach_trajectory_simple.json`.
// ════════════════════════════════════════════════════════════════════

// veh3 is free-flying across the whole run (never attached). Same
// rigid-body-only RK4 conditions as the kinematic-propagation test.
// Tolerances mirror that test's veh3 row.
const VEH3_POSITION_TOL_M: f64 = 1.5e-12;
const VEH3_VELOCITY_TOL_MPS: f64 = 1e-15;
const VEH3_QUAT_ANGLE_TOL_RAD: f64 = 1e-15;
const VEH3_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

// Pre-attach `t ∈ [0, 10)` veh1 + veh2: same rigid-body-only
// conditions as veh3. Quaternion-angle tolerance absorbs the
// `2 · acos(|q · q'|)` ULP residual on a non-trivially-rotated body
// printed at JEOD's `%g` precision.
const PRE_ATTACH_POSITION_TOL_M: f64 = 1.5e-13;
const PRE_ATTACH_VELOCITY_TOL_MPS: f64 = 1e-15;
const PRE_ATTACH_QUAT_ANGLE_TOL_RAD: f64 = 4.5e-8;
const PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S: f64 = 1e-15;

// Attached window `t ∈ [10, 20)`: veh1 attached to veh2 via the
// production `Simulation::attach` API (which runs JEOD's
// `combine_states_at_attach` per #307). Both bodies'
// `composite_body` state — veh2's integrated, veh1's derived via
// `propagate_kinematic_state` — is cross-validated against JEOD's
// CSV. The position residuals are dominated by f64 round-off in the
// kinematic-kernel composition and the integrator's own ULP-level
// drift; the quaternion residuals are JEOD's `%g`-formatted CSV
// print precision (~3e-8 rad on a non-trivially rotated body).
//
// Observed (JSON report, this test):
//   veh1: pos=1.667e-9, vel=3.065e-11, quat=2.980e-8, ω=5.55e-17
//   veh2: pos=5.555e-10, vel=0, quat=2.980e-8, ω=2.08e-17
const ATTACHED_VEH1_POSITION_TOL_M: f64 = 1.75e-9;
const ATTACHED_VEH1_VELOCITY_TOL_MPS: f64 = 3.22e-11;
const ATTACHED_VEH1_QUAT_ANGLE_TOL_RAD: f64 = 3.13e-8;
const ATTACHED_VEH1_ANG_VEL_TOL_RAD_PER_S: f64 = 5.83e-17;

const ATTACHED_VEH2_POSITION_TOL_M: f64 = 5.84e-10;
const ATTACHED_VEH2_VELOCITY_TOL_MPS: f64 = 1e-15;
const ATTACHED_VEH2_QUAT_ANGLE_TOL_RAD: f64 = 3.13e-8;
const ATTACHED_VEH2_ANG_VEL_TOL_RAD_PER_S: f64 = 2.19e-17;

// Post-detach window `t ∈ [20, 30)`: veh1 detached via the
// production `Simulation::detach` API (which runs the inverse of
// `combine_states_at_attach` per #307). Both bodies are
// independently integrated again. Same dominant-error story as the
// attached window — f64 round-off + JEOD CSV print precision.
//
// Observed (JSON report, this test):
//   veh1: pos=1.651e-9, vel=3.065e-11, quat=2.980e-8, ω=2.08e-17
//   veh2: pos=2.976e-10, vel=1.532e-11, quat=2.980e-8, ω=2.08e-17
const POST_DETACH_VEH1_POSITION_TOL_M: f64 = 1.74e-9;
const POST_DETACH_VEH1_VELOCITY_TOL_MPS: f64 = 3.22e-11;
const POST_DETACH_VEH1_QUAT_ANGLE_TOL_RAD: f64 = 3.13e-8;
const POST_DETACH_VEH1_ANG_VEL_TOL_RAD_PER_S: f64 = 2.19e-17;

const POST_DETACH_VEH2_POSITION_TOL_M: f64 = 3.13e-10;
const POST_DETACH_VEH2_VELOCITY_TOL_MPS: f64 = 1.61e-11;
const POST_DETACH_VEH2_QUAT_ANGLE_TOL_RAD: f64 = 3.13e-8;
const POST_DETACH_VEH2_ANG_VEL_TOL_RAD_PER_S: f64 = 2.19e-17;

/// Per-window per-body max-error accumulator.
#[derive(Default, Clone, Copy)]
struct WindowErrors {
    pos: f64,
    vel: f64,
    quat: f64,
    ang_vel: f64,
}

impl WindowErrors {
    fn update(&mut self, runner: &VehSnapshot, csv: &VehSnapshot) {
        self.pos = self.pos.max((runner.position - csv.position).length());
        self.vel = self.vel.max((runner.velocity - csv.velocity).length());
        self.quat = self
            .quat
            .max(quat_angle_err(runner.quaternion, csv.quaternion));
        self.ang_vel = self
            .ang_vel
            .max((runner.ang_vel_body - csv.ang_vel_body).length());
    }
}

fn body_snapshot(sim: &Simulation, idx: usize) -> VehSnapshot {
    let out = sim.body(idx);
    let rot = out
        .rot
        .expect("attach/detach trajectory test runs every body in 6-DOF");
    VehSnapshot {
        position: out.trans.position,
        velocity: out.trans.velocity,
        quaternion: rot.quaternion,
        ang_vel_body: rot.ang_vel_body,
    }
}

/// Drive the runner end-to-end through `Simulation::step()` from
/// `t = 0` to just past `FRAME_ATTACH_PHASE_START`, firing
/// `Simulation::attach(v1, v2, ...)` at `t ≈ ATTACH_TIME` and
/// `Simulation::detach(v1)` at `t ≈ DETACH_TIME`. Compares the
/// runner's `composite_body` state against JEOD's CSV at every CSV
/// sample, partitioned into the three windows: pre-attach, attached,
/// post-detach.
// non-recipe: SIM_verif_attach_detach exercises a placeholder mass
// tree (1/2/3 kg, three free vehicles in empty space). ISS / Apollo
// recipes don't apply.
#[test]
fn tier3_sim_attach_detach_trajectory_simple() {
    let rows = load_csv("kinematic_propagation_simple_kinematic_propagation_state.csv");
    assert!(rows.len() > 60, "expected >60 rows, got {}", rows.len());

    let (mut sim, v1, v2, v3) = build_sim();

    // Sanity at t=0: CSV's first row must match the JEOD source-file
    // initial conditions we wired into `build_sim`.
    let r0 = &rows[0];
    assert!(
        (r0.veh[0].position - veh1_initial_trans().position).length() < 1e-12
            && (r0.veh[1].position - veh2_initial_trans().position).length() < 1e-12
            && (r0.veh[2].position - veh3_initial_trans().position).length() < 1e-12,
        "CSV t=0 does not match the JEOD source-file initial conditions \
         (Modified_data/veh{{1,2,3}}.py); refresh the fixture or fix the IC \
         constants in this test."
    );

    let mut pre_attach_v1 = WindowErrors::default();
    let mut pre_attach_v2 = WindowErrors::default();
    let mut attached_v1 = WindowErrors::default();
    let mut attached_v2 = WindowErrors::default();
    let mut post_detach_v1 = WindowErrors::default();
    let mut post_detach_v2 = WindowErrors::default();
    let mut veh3_err = WindowErrors::default();

    let mut attach_fired = false;
    let mut detach_fired = false;
    let dt = sim.dt;

    for row in &rows {
        // Stop validation past `FRAME_ATTACH_PHASE_START` — the JEOD
        // run fires `set_attitude_rate` at t=30, then a sequence of
        // reference-frame attach/detach events that aren't modelled
        // here (deferred-dynamics scope).
        if row.time >= FRAME_ATTACH_PHASE_START {
            break;
        }

        // Step until sim.elapsed() ≈ row.time, then fire any due
        // attach / detach event at simtime = T *after* the integration
        // that lands on T (so the t=9.9→10.0 step integrates with the
        // bodies still separate, matching JEOD's read-job semantics
        // where the attach action fires at simtime=T, after the
        // integrator has produced the t=T state for both bodies).
        //
        // Caveat: JEOD's CSV row at exactly `t = T_event` captures
        // veh1's *post-attach + post-propagate-state* snapshot — JEOD's
        // logging cycle runs `propagate_state` once after the read job
        // and before the data-record dump. Our `Simulation::attach`
        // does not invoke `propagate_kinematic_state` directly — that
        // walk runs only inside `step()`. So immediately after attach
        // here, runner.veh1 still carries its pre-attach integrated
        // state. The runner-veh1 vs csv-veh1 comparison therefore skips
        // the row at exactly `t = T_event` (handled below); the next
        // CSV row sees runner.veh1 derived from veh2 by the next
        // `step()`'s pre-integration propagation walk, matching JEOD's
        // CSV bit-faithfully (modulo f64 round-off).
        while sim.elapsed() + 0.5 * dt < row.time {
            sim.step().expect("step() must succeed");
        }
        if !attach_fired && sim.elapsed() + 0.5 * dt >= ATTACH_TIME {
            let (offset, t_pc) = simple_attach_offset_and_rotation();
            sim.attach(v1, v2, offset, t_pc);
            sim.mark_kinematic_only(v1);
            attach_fired = true;
        }
        if !detach_fired && sim.elapsed() + 0.5 * dt >= DETACH_TIME {
            sim.detach(v1);
            detach_fired = true;
        }

        // veh3: free-flying across the entire run.
        let v3_snap = body_snapshot(&sim, v3);
        veh3_err.update(&v3_snap, &row.veh[2]);

        // Window-partitioned veh1/veh2 cross-validation.
        //
        // Attached/post-detach veh1: the row at exactly `t = T_event`
        // captures JEOD's veh1.composite_body *after* the post-event
        // `propagate_state` call runs. Our `Simulation::attach` /
        // `detach` do not propagate kinematic state directly (that walk
        // runs only inside `step()`), so we skip those exact rows for
        // veh1 — the next CSV row sees runner.veh1 derived from veh2
        // by the next `step()`'s pre-integration propagation walk and
        // matches JEOD's CSV at f64-noise floor. veh2 (the integrated
        // tree root after attach, and again itself after detach) is
        // updated directly by `combine_states_at_attach` / its inverse,
        // so its t=T_event snapshot matches JEOD without skipping.
        let v1_snap = body_snapshot(&sim, v1);
        let v2_snap = body_snapshot(&sim, v2);
        let half = 0.5 * dt;
        let is_attach_event_row = (row.time - ATTACH_TIME).abs() < half;
        let is_detach_event_row = (row.time - DETACH_TIME).abs() < half;
        if row.time < ATTACH_TIME {
            pre_attach_v1.update(&v1_snap, &row.veh[0]);
            pre_attach_v2.update(&v2_snap, &row.veh[1]);
        } else if row.time < DETACH_TIME {
            if !is_attach_event_row {
                attached_v1.update(&v1_snap, &row.veh[0]);
            }
            attached_v2.update(&v2_snap, &row.veh[1]);
        } else {
            if !is_detach_event_row {
                post_detach_v1.update(&v1_snap, &row.veh[0]);
            }
            post_detach_v2.update(&v2_snap, &row.veh[1]);
        }
    }

    // Emit the cross-validation report.
    let mut report = CrossvalReport::compute("tier3_sim_attach_detach_trajectory_simple", &[], &[]);
    let push = |report: &mut CrossvalReport, label: &str, w: &WindowErrors| {
        report.add_extra(&format!("{label}_max_position_err"), w.pos, "m");
        report.add_extra(&format!("{label}_max_velocity_err"), w.vel, "m/s");
        report.add_extra(&format!("{label}_max_quat_angle_err"), w.quat, "rad");
        report.add_extra(&format!("{label}_max_ang_vel_err"), w.ang_vel, "rad/s");
    };
    push(&mut report, "pre_attach_veh1", &pre_attach_v1);
    push(&mut report, "pre_attach_veh2", &pre_attach_v2);
    push(&mut report, "attached_veh1", &attached_v1);
    push(&mut report, "attached_veh2", &attached_v2);
    push(&mut report, "post_detach_veh1", &post_detach_v1);
    push(&mut report, "post_detach_veh2", &post_detach_v2);
    push(&mut report, "veh3", &veh3_err);
    report.write();

    // ── Assertions ──

    // Pre-attach: rigid-body-only RK4 floor.
    assert!(
        pre_attach_v1.pos < PRE_ATTACH_POSITION_TOL_M,
        "veh1 pre-attach position {:.3e} m exceeds {PRE_ATTACH_POSITION_TOL_M:.1e}",
        pre_attach_v1.pos
    );
    assert!(
        pre_attach_v1.vel < PRE_ATTACH_VELOCITY_TOL_MPS,
        "veh1 pre-attach velocity {:.3e} m/s exceeds {PRE_ATTACH_VELOCITY_TOL_MPS:.1e}",
        pre_attach_v1.vel
    );
    assert!(
        pre_attach_v1.quat < PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
        "veh1 pre-attach quat-angle {:.3e} rad exceeds {PRE_ATTACH_QUAT_ANGLE_TOL_RAD:.1e}",
        pre_attach_v1.quat
    );
    assert!(
        pre_attach_v1.ang_vel < PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
        "veh1 pre-attach ang-vel {:.3e} rad/s exceeds {PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S:.1e}",
        pre_attach_v1.ang_vel
    );
    assert!(
        pre_attach_v2.pos < PRE_ATTACH_POSITION_TOL_M,
        "veh2 pre-attach position {:.3e} m exceeds {PRE_ATTACH_POSITION_TOL_M:.1e}",
        pre_attach_v2.pos
    );
    assert!(
        pre_attach_v2.vel < PRE_ATTACH_VELOCITY_TOL_MPS,
        "veh2 pre-attach velocity {:.3e} m/s exceeds {PRE_ATTACH_VELOCITY_TOL_MPS:.1e}",
        pre_attach_v2.vel
    );
    assert!(
        pre_attach_v2.quat < PRE_ATTACH_QUAT_ANGLE_TOL_RAD,
        "veh2 pre-attach quat-angle {:.3e} rad exceeds {PRE_ATTACH_QUAT_ANGLE_TOL_RAD:.1e}",
        pre_attach_v2.quat
    );
    assert!(
        pre_attach_v2.ang_vel < PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S,
        "veh2 pre-attach ang-vel {:.3e} rad/s exceeds {PRE_ATTACH_ANG_VEL_TOL_RAD_PER_S:.1e}",
        pre_attach_v2.ang_vel
    );

    // Attached window.
    assert!(
        attached_v1.pos < ATTACHED_VEH1_POSITION_TOL_M,
        "veh1 attached position {:.3e} m exceeds {ATTACHED_VEH1_POSITION_TOL_M:.1e}",
        attached_v1.pos
    );
    assert!(
        attached_v1.vel < ATTACHED_VEH1_VELOCITY_TOL_MPS,
        "veh1 attached velocity {:.3e} m/s exceeds {ATTACHED_VEH1_VELOCITY_TOL_MPS:.1e}",
        attached_v1.vel
    );
    assert!(
        attached_v1.quat < ATTACHED_VEH1_QUAT_ANGLE_TOL_RAD,
        "veh1 attached quat-angle {:.3e} rad exceeds {ATTACHED_VEH1_QUAT_ANGLE_TOL_RAD:.1e}",
        attached_v1.quat
    );
    assert!(
        attached_v1.ang_vel < ATTACHED_VEH1_ANG_VEL_TOL_RAD_PER_S,
        "veh1 attached ang-vel {:.3e} rad/s exceeds {ATTACHED_VEH1_ANG_VEL_TOL_RAD_PER_S:.1e}",
        attached_v1.ang_vel
    );
    assert!(
        attached_v2.pos < ATTACHED_VEH2_POSITION_TOL_M,
        "veh2 attached position {:.3e} m exceeds {ATTACHED_VEH2_POSITION_TOL_M:.1e}",
        attached_v2.pos
    );
    assert!(
        attached_v2.vel < ATTACHED_VEH2_VELOCITY_TOL_MPS,
        "veh2 attached velocity {:.3e} m/s exceeds {ATTACHED_VEH2_VELOCITY_TOL_MPS:.1e}",
        attached_v2.vel
    );
    assert!(
        attached_v2.quat < ATTACHED_VEH2_QUAT_ANGLE_TOL_RAD,
        "veh2 attached quat-angle {:.3e} rad exceeds {ATTACHED_VEH2_QUAT_ANGLE_TOL_RAD:.1e}",
        attached_v2.quat
    );
    assert!(
        attached_v2.ang_vel < ATTACHED_VEH2_ANG_VEL_TOL_RAD_PER_S,
        "veh2 attached ang-vel {:.3e} rad/s exceeds {ATTACHED_VEH2_ANG_VEL_TOL_RAD_PER_S:.1e}",
        attached_v2.ang_vel
    );

    // Post-detach window.
    assert!(
        post_detach_v1.pos < POST_DETACH_VEH1_POSITION_TOL_M,
        "veh1 post-detach position {:.3e} m exceeds {POST_DETACH_VEH1_POSITION_TOL_M:.1e}",
        post_detach_v1.pos
    );
    assert!(
        post_detach_v1.vel < POST_DETACH_VEH1_VELOCITY_TOL_MPS,
        "veh1 post-detach velocity {:.3e} m/s exceeds {POST_DETACH_VEH1_VELOCITY_TOL_MPS:.1e}",
        post_detach_v1.vel
    );
    assert!(
        post_detach_v1.quat < POST_DETACH_VEH1_QUAT_ANGLE_TOL_RAD,
        "veh1 post-detach quat-angle {:.3e} rad exceeds {POST_DETACH_VEH1_QUAT_ANGLE_TOL_RAD:.1e}",
        post_detach_v1.quat
    );
    assert!(
        post_detach_v1.ang_vel < POST_DETACH_VEH1_ANG_VEL_TOL_RAD_PER_S,
        "veh1 post-detach ang-vel {:.3e} rad/s exceeds {POST_DETACH_VEH1_ANG_VEL_TOL_RAD_PER_S:.1e}",
        post_detach_v1.ang_vel
    );
    assert!(
        post_detach_v2.pos < POST_DETACH_VEH2_POSITION_TOL_M,
        "veh2 post-detach position {:.3e} m exceeds {POST_DETACH_VEH2_POSITION_TOL_M:.1e}",
        post_detach_v2.pos
    );
    assert!(
        post_detach_v2.vel < POST_DETACH_VEH2_VELOCITY_TOL_MPS,
        "veh2 post-detach velocity {:.3e} m/s exceeds {POST_DETACH_VEH2_VELOCITY_TOL_MPS:.1e}",
        post_detach_v2.vel
    );
    assert!(
        post_detach_v2.quat < POST_DETACH_VEH2_QUAT_ANGLE_TOL_RAD,
        "veh2 post-detach quat-angle {:.3e} rad exceeds {POST_DETACH_VEH2_QUAT_ANGLE_TOL_RAD:.1e}",
        post_detach_v2.quat
    );
    assert!(
        post_detach_v2.ang_vel < POST_DETACH_VEH2_ANG_VEL_TOL_RAD_PER_S,
        "veh2 post-detach ang-vel {:.3e} rad/s exceeds {POST_DETACH_VEH2_ANG_VEL_TOL_RAD_PER_S:.1e}",
        post_detach_v2.ang_vel
    );

    // veh3: free-flying baseline across the entire run.
    assert!(
        veh3_err.pos < VEH3_POSITION_TOL_M,
        "veh3 position {:.3e} m exceeds {VEH3_POSITION_TOL_M:.1e}",
        veh3_err.pos
    );
    assert!(
        veh3_err.vel < VEH3_VELOCITY_TOL_MPS,
        "veh3 velocity {:.3e} m/s exceeds {VEH3_VELOCITY_TOL_MPS:.1e}",
        veh3_err.vel
    );
    assert!(
        veh3_err.quat < VEH3_QUAT_ANGLE_TOL_RAD,
        "veh3 quaternion {:.3e} rad exceeds {VEH3_QUAT_ANGLE_TOL_RAD:.1e}",
        veh3_err.quat
    );
    assert!(
        veh3_err.ang_vel < VEH3_ANG_VEL_TOL_RAD_PER_S,
        "veh3 ang-vel {:.3e} rad/s exceeds {VEH3_ANG_VEL_TOL_RAD_PER_S:.1e}",
        veh3_err.ang_vel
    );
}
