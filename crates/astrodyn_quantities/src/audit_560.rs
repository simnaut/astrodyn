//! #560 operation-level FP-parity dump infrastructure.
//!
//! Shared by every crate in the contact RK4 path. Emits one stderr line
//! per instrumented FP intermediate, gated by the `ASTRODYN_560_FULL_DUMP`
//! environment variable so production runs see zero overhead.
//!
//! Line format:
//!
//! ```text
//! [#560/FULL] step=N stage=K body=B op=<name> k1=v1 k2=v2 ...
//! ```
//!
//! where step/stage are thread-local counters set by the integrator
//! driver before each `eval_stage` call, body is the per-body index
//! (0..n), op is a stable identifier (e.g., `rel_pos`, `spring_force`,
//! `t_inertial_body`), and `kI=vI` are 17-digit f64 dumps of each
//! component. Tools that diff JEOD's matching stream and ours align on
//! the `(step, stage, body, op)` key.
//!
//! All entry points are no-ops when the env var is unset. Remove this
//! module and every `dump_*` call site once #560 lands.
//!
//! See `/home/user/git/2astrodyn/.claude/plans/switch-to-main-pull-composed-tower.md`
//! for the Phase A / B / C plan this infrastructure supports.

use glam::{DMat3, DVec3};
use std::cell::Cell;

thread_local! {
    /// Step counter — incremented at the top of each
    /// `integrate_bodies_contact_coupled` call.
    pub static DUMP_STEP: Cell<usize> = const { Cell::new(0) };
    /// Stage counter — set before each RK4 stage's `eval_stage` call
    /// (1, 2, 3, 4). Returns to 0 in end-of-step composition.
    pub static DUMP_STAGE: Cell<usize> = const { Cell::new(0) };
    /// Are we inside `integrate_bodies_contact_coupled`? Dumps fire only
    /// when true — this excludes the `assert_contact_force_torque`
    /// post-sim assertion path which would otherwise add ~6 extra
    /// occurrences per op and break diff_streams alignment with JEOD's
    /// strictly in-sim instrumentation.
    pub static DUMP_IN_INTEG: Cell<bool> = const { Cell::new(false) };
}

/// Returns `true` when `ASTRODYN_560_FULL_DUMP` is set **AND** the
/// integrator's `enter_integration()` has been called without a matching
/// `exit_integration()`.
///
/// The env-var check is cached per thread; the integration gate is the
/// thread-local `DUMP_IN_INTEG` flag.
#[inline]
pub fn enabled() -> bool {
    thread_local! {
        static CACHED: Cell<Option<bool>> = const { Cell::new(None) };
    }
    let env_ok = CACHED.with(|c| {
        if let Some(v) = c.get() {
            v
        } else {
            let v = std::env::var_os("ASTRODYN_560_FULL_DUMP").is_some();
            c.set(Some(v));
            v
        }
    });
    env_ok && DUMP_IN_INTEG.with(|f| f.get())
}

/// Mark entry into the integrator's hot path. Dumps fire from this point
/// until [`exit_integration`] is called.
#[inline]
pub fn enter_integration() {
    DUMP_IN_INTEG.with(|f| f.set(true));
}

/// Mark exit from the integrator's hot path. Dumps cease until the next
/// [`enter_integration`] call.
#[inline]
pub fn exit_integration() {
    DUMP_IN_INTEG.with(|f| f.set(false));
}

/// Reset the step counter to 0. Called from the test harness if it
/// re-enters the integrator for a fresh trajectory.
pub fn reset_step() {
    DUMP_STEP.with(|s| s.set(0));
    DUMP_STAGE.with(|s| s.set(0));
}

/// Increment the step counter and return the new value. Called at the
/// top of `integrate_bodies_contact_coupled`.
#[inline]
pub fn advance_step() -> usize {
    DUMP_STEP.with(|s| {
        let next = s.get() + 1;
        s.set(next);
        next
    })
}

/// Set the RK4 stage counter. Called by the integrator driver before
/// `eval_stage`.
#[inline]
pub fn set_stage(k: usize) {
    DUMP_STAGE.with(|s| s.set(k));
}

/// Get the current (step, stage) — used by the dump emitters.
#[inline]
pub fn step_stage() -> (usize, usize) {
    let step = DUMP_STEP.with(|s| s.get());
    let stage = DUMP_STAGE.with(|s| s.get());
    (step, stage)
}

/// Internal: format a (step, stage, body, op) line header.
#[inline]
fn header(body: usize, op: &str) -> String {
    let (step, stage) = step_stage();
    format!("[#560/FULL] step={step} stage={stage} body={body} op={op}")
}

/// Dump a single scalar.
pub fn dump_scalar(op: &str, body: usize, value: f64) {
    if !enabled() {
        return;
    }
    eprintln!("{} v={:.17e}", header(body, op), value);
}

/// Dump a 3-vector (`x`, `y`, `z`).
pub fn dump_vec3(op: &str, body: usize, v: DVec3) {
    if !enabled() {
        return;
    }
    eprintln!(
        "{} x={:.17e} y={:.17e} z={:.17e}",
        header(body, op),
        v.x,
        v.y,
        v.z,
    );
}

/// Dump a 4-element quaternion (`q0`, `q1`, `q2`, `q3` — JEOD
/// scalar-first ordering).
pub fn dump_quat(op: &str, body: usize, q: [f64; 4]) {
    if !enabled() {
        return;
    }
    eprintln!(
        "{} q0={:.17e} q1={:.17e} q2={:.17e} q3={:.17e}",
        header(body, op),
        q[0],
        q[1],
        q[2],
        q[3],
    );
}

/// Dump a 3×3 matrix as 9 components in row-major order
/// (`m00..m22`).
pub fn dump_mat3(op: &str, body: usize, m: DMat3) {
    if !enabled() {
        return;
    }
    // glam DMat3 is column-major in memory; we emit row-major for
    // legibility against JEOD's `Vector3::transform` convention which
    // reads `M[i][j]` as row-major.
    let r0 = m.row(0);
    let r1 = m.row(1);
    let r2 = m.row(2);
    eprintln!(
        "{} m00={:.17e} m01={:.17e} m02={:.17e} m10={:.17e} m11={:.17e} m12={:.17e} m20={:.17e} m21={:.17e} m22={:.17e}",
        header(body, op),
        r0.x, r0.y, r0.z,
        r1.x, r1.y, r1.z,
        r2.x, r2.y, r2.z,
    );
}
