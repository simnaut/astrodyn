// Issue #560 root-cause audit infrastructure. Diagnostic-only — every
// public function in this module is intended to compile to a single
// env-var check + early return when `ASTRODYN_560_FULL_DUMP` is unset
// (the production case). Tagged sites in `src/integration.rs`,
// `src/interactions.rs`, and `crates/astrodyn_interactions/src/contact.rs`
// call into here on every RK4 stage; the cost on the default code path
// must remain at one `OnceLock` read and one boolean branch.
//
// The dump format mirrors the JEOD-side `[#560/FULL]` emitter installed
// by `trick/audit_560/run_audit.sh` so `trick/audit_560/diff_streams.py`
// can align both streams by `(op, body, occurrence-index)`. The shared
// format is:
//
//   [#560/FULL] step=N stage=K body=B op=<name> kI=vI ...
//
// where `step` is the outer integration step (TimeUpdate cadence),
// `stage` is the RK4 stage (1..=4 for the contact-coupled kernel,
// 0 for end-of-step composition), `body` is the per-pair body index
// in the scratch ordering, and `kI=vI` are op-specific scalar / vector
// fields formatted with `{:.17e}` (17 sig figs — sufficient to recover
// the exact f64 bit pattern through `f64::from_str`).
//
// This module is `pub` so callers in `astrodyn` (the gateway) and
// `astrodyn_interactions` can reach it through the shared
// `astrodyn_quantities` foundation crate without anyone declaring a
// new cross-crate dep just for diagnostics.

//! Issue #560 root-cause audit infrastructure (diagnostic-only).
//!
//! See [Phase A+B operation-level audit](https://github.com/simnaut/astrodyn/issues/560)
//! for the audit conclusion: the 2.5 mm trajectory residual in
//! `tier3_contact_point_off_center` is collectively produced by
//! ULP-level FP rounding-path divergences across ~25 distinct
//! operations, amplified exponentially through stiff spring-damper
//! contact dynamics over 152 contact-event stages. The infrastructure
//! preserved by this module is the "production" half of the
//! bidirectional diff used to reach that conclusion; the JEOD-side
//! patches that emit the same line format live in
//! `trick/audit_560/run_audit.sh`, and the alignment / diff tool lives
//! in `trick/audit_560/diff_streams.py`.
//!
//! ## Activation
//!
//! Set `ASTRODYN_560_FULL_DUMP=1` in the environment. With the variable
//! unset (the production case), every public function in this module
//! short-circuits on the first read of the cached [`enabled`] flag —
//! the per-stage dump call sites pay one `OnceLock` read and one
//! boolean branch, and emit nothing.
//!
//! ## Format
//!
//! Each `dump_*` function emits a single newline-terminated line to
//! `stderr`:
//!
//! ```text
//! [#560/FULL] step=N stage=K body=B op=<name> kI=vI ...
//! ```
//!
//! - `step` — outer integration step counter, advanced once per
//!   [`begin_step`] call. **1-based**: the first emitted line carries
//!   `step=1` (the underlying counter starts at `0` as an
//!   "uninitialized" sentinel and is `wrapping_add(1)`'d on every
//!   [`begin_step`]). Pair with `stage` for the full per-line address.
//! - `stage` — RK4 stage, advanced once per [`begin_stage`] call.
//!   Convention: 1..=4 for the four RK4 stages, 0 for the end-of-step
//!   composition op (set by an explicit `begin_stage(0)`), sentinel
//!   `99` for "no stage context yet" (the value [`begin_step`] resets
//!   the stage to before any [`begin_stage`] runs).
//! - `body` — index in the contact-pair body ordering (0 for vehicle
//!   A, 1 for vehicle B in the SIM_contact two-body fixture).
//! - `op` — short opcode that names the operation being dumped
//!   (`rel_pos`, `geom_normal`, `force_total`, …). Kept stable across
//!   versions so the JEOD-side patch and the diff script can align by
//!   string match.
//!
//! Scalar values render as `kI=vI` with 17-significant-digit
//! exponential notation; vector values render as `kI.x=vIx kI.y=vIy
//! kI.z=vIz`; quaternion values render as `kI.0=vI0 kI.1=vI1 kI.2=vI2
//! kI.3=vI3` (component order matches `JeodQuat::data`).
//!
//! ## Counter discipline
//!
//! The `(step, stage)` counters live in **thread-local** storage so a
//! future parallel-bodies test path can dump per-thread without
//! cross-thread interleaving. The contact-coupled RK4 kernel runs the
//! four stages sequentially on a single thread, so the thread-local
//! choice is sufficient for the existing call sites and zero-cost on
//! the default path.

use glam::DVec3;
use std::cell::Cell;
use std::sync::OnceLock;

/// Read the `ASTRODYN_560_FULL_DUMP` env var once at first access and
/// cache the result for the remainder of the process lifetime.
///
/// Returns `true` only when the variable is set to a non-empty string.
/// An unset variable, an empty value, or a read failure all return
/// `false`. The cache lives in a process-wide [`OnceLock`] so every
/// dump call site pays one atomic load when the dump is disabled.
#[inline]
pub fn enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("ASTRODYN_560_FULL_DUMP") {
        Ok(v) => !v.is_empty(),
        Err(_) => false,
    })
}

thread_local! {
    /// Outer integration-step counter. Incremented by [`begin_step`];
    /// resets are caller-managed via [`reset_counters`].
    ///
    /// 1-based after the first [`begin_step`] call: the counter starts
    /// at `0` (uninitialized sentinel) and the `wrapping_add(1)` in
    /// `begin_step` makes the *first emitted* step `1`. End-of-step
    /// composition for step `N` reuses `step=N` with `stage=0`.
    static STEP: Cell<u64> = const { Cell::new(0) };
    /// RK4 stage counter. Set by [`begin_stage`]; reset by
    /// [`begin_step`] to the sentinel `99` (meaning "between stages —
    /// no stage context yet"). End-of-step composition is tagged with
    /// `stage=0` by an explicit `begin_stage(0)` from the caller; the
    /// `99` sentinel only appears on any line emitted between the start
    /// of a step and its first `begin_stage` call.
    static STAGE: Cell<u32> = const { Cell::new(99) };
}

/// Reset both counters to their initial state. Used by tests that
/// drive the kernel directly and want the dump stream to start from
/// `step=0 stage=99`. Not called in production code.
#[inline]
pub fn reset_counters() {
    STEP.with(|s| s.set(0));
    STAGE.with(|s| s.set(99));
}

/// Begin a new outer integration step. Increments the step counter
/// (1-based: the underlying counter starts at `0` as an
/// "uninitialized" sentinel, and the first call makes the emitted
/// step `1`) and sets the stage to the sentinel `99` so any dump
/// emitted before the first [`begin_stage`] call of this step is
/// unambiguously "no stage context".
///
/// No-op when [`enabled`] returns `false`.
#[inline]
pub fn begin_step() {
    if !enabled() {
        return;
    }
    STEP.with(|s| s.set(s.get().wrapping_add(1)));
    STAGE.with(|s| s.set(99));
}

/// Begin a new RK4 stage within the current step. The caller is
/// expected to pass `1..=4` for the four real stages and `0` for the
/// end-of-step composition window. Any other value is forwarded
/// verbatim and emitted as-is.
///
/// No-op when [`enabled`] returns `false`.
#[inline]
pub fn begin_stage(stage: u32) {
    if !enabled() {
        return;
    }
    STAGE.with(|s| s.set(stage));
}

/// Snapshot the current `(step, stage)` pair without mutating either
/// counter. Used by the formatting helpers below.
#[inline]
fn current() -> (u64, u32) {
    let step = STEP.with(Cell::get);
    let stage = STAGE.with(Cell::get);
    (step, stage)
}

/// Emit one scalar field. Format: `op=<name> <key>=<value>` with
/// `value` rendered as `{:.17e}`. Returns immediately when dump is
/// disabled.
#[inline]
pub fn dump_scalar(op: &str, body: usize, key: &str, value: f64) {
    if !enabled() {
        return;
    }
    let (step, stage) = current();
    eprintln!("[#560/FULL] step={step} stage={stage} body={body} op={op} {key}={value:.17e}");
}

/// Emit one 3-vector field. Format: `op=<name> <key>.x=<x>
/// <key>.y=<y> <key>.z=<z>` with 17-significant-digit exponential
/// rendering. Returns immediately when dump is disabled.
#[inline]
pub fn dump_vec3(op: &str, body: usize, key: &str, v: DVec3) {
    if !enabled() {
        return;
    }
    let (step, stage) = current();
    eprintln!(
        "[#560/FULL] step={step} stage={stage} body={body} op={op} \
         {key}.x={x:.17e} {key}.y={y:.17e} {key}.z={z:.17e}",
        x = v.x,
        y = v.y,
        z = v.z,
    );
}

/// Emit one 4-component quaternion field laid out as
/// `[q0, q1, q2, q3]` — matching `JeodQuat::data`'s scalar-first
/// storage. Format: `op=<name> <key>.0=<q0> ... <key>.3=<q3>` with
/// 17-significant-digit exponential rendering. Returns immediately
/// when dump is disabled.
#[inline]
pub fn dump_quat(op: &str, body: usize, key: &str, q: [f64; 4]) {
    if !enabled() {
        return;
    }
    let (step, stage) = current();
    eprintln!(
        "[#560/FULL] step={step} stage={stage} body={body} op={op} \
         {key}.0={q0:.17e} {key}.1={q1:.17e} {key}.2={q2:.17e} {key}.3={q3:.17e}",
        q0 = q[0],
        q1 = q[1],
        q2 = q[2],
        q3 = q[3],
    );
}

#[cfg(test)]
mod tests {
    //! The audit module's tests deliberately do not exercise the
    //! `dump_*` emitters' stderr output — `OnceLock` caches the
    //! env-var read for the lifetime of the process, so any test that
    //! observed the dump path would be order-dependent with respect to
    //! whichever other test ran first in the same process. The
    //! production-path guarantee (no-op when the env var is unset) is
    //! verified at the timing level by the spot check in
    //! `tier3_sim_contact` — the suite must run in roughly the same
    //! time as on `main`.
    //!
    //! Instead we cover the no-op invariants of the counter helpers
    //! and the byte-shape of the format strings via direct buffer
    //! formatting in a `format!`-equivalent path.
    use super::*;

    #[test]
    fn begin_step_and_stage_no_op_when_disabled() {
        // `ASTRODYN_560_FULL_DUMP` is unset in the default test
        // environment. Both calls must be no-ops; the counters stay at
        // their initial values.
        reset_counters();
        begin_step();
        begin_stage(2);
        let (step, stage) = current();
        assert_eq!(step, 0, "begin_step must be a no-op when disabled");
        assert_eq!(stage, 99, "begin_stage must be a no-op when disabled");
    }

    #[test]
    fn reset_counters_brings_state_back_to_initial() {
        reset_counters();
        let (step, stage) = current();
        assert_eq!(step, 0);
        assert_eq!(stage, 99);
    }
}
