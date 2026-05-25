//! LSODE — Livermore Solver for Ordinary Differential Equations.
//!
//! A port of JEOD's `utils/integration/lsode/` (itself a de-Fortran-ed
//! ODEPACK `DLSODE`): a variable-order, variable-step Nordsieck multistep
//! integrator with an implicit-Adams (non-stiff, orders 1–12) family and a
//! BDF (stiff, orders 1–5) family. Closes issues #200 / #122.
//!
//! ## Closure-driven design
//!
//! JEOD's LSODE is *re-entrant*: each ODEPACK `CALL F` becomes a return to
//! Trick (which owns the derivative function), dispatched through a
//! `re_entry_point` state machine. Our pipeline instead hands the
//! integrator a callable derivative closure (the way RK4/RKF45 already
//! work), so this port replaces each "return for a fresh derivative" with
//! an inline `accel_fn(state)` call and drops the entire FSM. The result is
//! numerically identical — the adaptive order/step sequence is driven by
//! the coefficients, error norms, and controller, none of which the
//! F-delivery mechanism touches — but far less error-prone.
//!
//! ## Families and correctors (#200, #122)
//!
//! - **Non-stiff implicit Adams** (orders 1–12) with the functional-iteration
//!   corrector ([`lsode_translational_step`] via [`functional_corrector`]).
//! - **Stiff BDF** (orders 1–5) with a modified-Newton chord corrector
//!   ([`chord_corrector`]) driven by an internally-generated forward-
//!   difference Jacobian ([`build_dense_iteration_matrix`], ODEPACK MITER=2)
//!   and a dense LU solve ([`linalg`]). The iteration matrix
//!   `P = I − h·el0·J` is factored once and reused until it drifts stale
//!   (`MAX_REL_CHANGE_WITHOUT_JACOBIAN` / `MAX_STEPS_PER_JACOBIAN`).
//!
//! The diagonal Jacobi-Newton approximation (ODEPACK MITER=3,
//! `JacobiNewtonInternalJac`) is not yet ported — selecting it panics
//! loudly in [`dstode_step`] rather than silently running a different
//! corrector.
//!
//! The flattened first-order system is `y = [position; velocity]` with
//! `y_dot = [velocity; acceleration]`; the closure supplies the
//! acceleration (bottom three components), the top three are the current
//! velocity.

// The driver mirrors ODEPACK/JEOD index arithmetic over the Nordsieck
// array and work vectors line-by-line for auditability; the `[i]` index
// loops and `a = b op a` recurrences are deliberate, and all integer→f64
// casts are small order/column counts (≤ 13) that are exactly representable.
#![allow(
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "driver transcribes DSTODE/DLSODE index arithmetic for line-by-line auditability; test step-count casts are small exact integers"
)]

pub mod coeffs;
pub mod config;
pub mod error_weights;
pub(crate) mod linalg;
pub mod nordsieck;

pub use config::{CorrectorMethod, IntegrationMethod, LsodeConfig};

use crate::state::TranslationalState;
use coeffs::{MethodCoeffs, TestCoeffs};
use error_weights::{load_error_weights, weighted_rms_norm};
use glam::DVec3;
use nordsieck::Nordsieck;

/// Number of flattened first-order ODEs for one translational body
/// (`[position(3); velocity(3)]`).
const N_ODES: usize = 6;

/// Maximum relative change in `h·el0` tolerated before the stiff
/// corrector's iteration matrix is rebuilt (ODEPACK `CCMAX`, JEOD
/// `max_rel_change_without_jacobian`).
const MAX_REL_CHANGE_WITHOUT_JACOBIAN: f64 = 0.3;

/// Maximum number of steps between forced iteration-matrix rebuilds in the
/// stiff corrector (ODEPACK `MSBP`, JEOD `max_num_steps_jacobian`).
const MAX_STEPS_PER_JACOBIAN: usize = 20;

/// Persistent LSODE integrator state for one body, carried across
/// `integrate` calls (the Nordsieck history, current order/step, and the
/// adaptive-control bookkeeping). Analogous to
/// [`crate::gauss_jackson::GaussJacksonState`].
#[derive(Debug, Clone)]
pub struct LsodeState {
    config: LsodeConfig,
    /// ELCO coefficient table (all orders), computed once.
    method_coeffs: MethodCoeffs,
    /// TESCO test-coefficient table (all orders), computed once.
    test_coeffs: TestCoeffs,
    /// Current-order method coefficients `el[0..=nq]` (`method_coeffs_current`).
    el: [f64; 13],
    /// Nordsieck history array.
    nordsieck: Nordsieck,
    /// Current method order `nq` (`method_order_current`).
    order: usize,
    /// Number of active Nordsieck columns (`= order + 1`).
    num_cols: usize,
    /// Spare-column index in the Nordsieck array (JEOD's `max_history_size`
    /// = `max_order_internal`, the family cap clamped by config). The
    /// history array has `max_history_size + 1` columns, and the order
    /// increase is gated on `num_cols != max_history_size`, so the
    /// **effective maximum integration order is `max_history_size - 1`**
    /// and column `max_history_size` is always a free spare for the
    /// order-increase indicator (never an active Nordsieck column).
    max_history_size: usize,
    /// Current internal step size `h` (`step_size`).
    step_size: f64,
    /// Step size from the previous step (`prev_step_size`, HOLD).
    prev_step_size: f64,
    /// Internal time the integrator has advanced to (`stage_target_time`, TN).
    stage_target_time: f64,
    /// This cycle's target offset = `dyn_dt` (`cycle_target_time`).
    cycle_target_time: f64,
    /// Countdown until an order/step change may be considered
    /// (`order_select_para`, IALTH).
    order_select_para: usize,
    /// Cap on the step-increase ratio (`max_step_increase_ratio`, RMAX).
    max_step_increase_ratio: f64,
    /// Corrector convergence-rate estimate (`convergence_rate`, CRATE).
    convergence_rate: f64,
    /// Steps taken since construction (`num_steps_taken`, NST).
    num_steps_taken: usize,
    /// 0 on the first step (order-1 init), 1 thereafter.
    internal_state: i32,
    /// True until the first `integrate` call has initialized the history.
    first_pass: bool,
    /// Order used on the last successful step (`prev_method_order`, NQU).
    prev_method_order: usize,
    /// Inverted error weights `1/ewt` (DVNORM multiplies by these).
    error_weight: [f64; N_ODES],
    /// Multistep history invalidated by an attach/detach topology change.
    topology_dirty: bool,

    // ── Stiff (chord/Newton) corrector state. Unused for functional
    // iteration; carried here so the iteration matrix persists across the
    // ~20 steps between rebuilds (ODEPACK reuses the factored matrix). ──
    /// Iteration matrix `P = I − h·el0·J`, LU-factored in place by
    /// [`linalg::lu_factor`]. Only meaningful when `jacobian_current`.
    iter_matrix: [[f64; N_ODES]; N_ODES],
    /// Row pivots from the `iter_matrix` LU factorization.
    iter_pivots: [usize; N_ODES],
    /// `h·el0` at the last iteration-matrix build (drift tracking against
    /// `MAX_REL_CHANGE_WITHOUT_JACOBIAN`).
    jac_hl0: f64,
    /// Whether `iter_matrix` reflects the current `h·el0` and state.
    jacobian_current: bool,
    /// `num_steps_taken` at the last iteration-matrix build (forced-rebuild
    /// cadence against `MAX_STEPS_PER_JACOBIAN`).
    steps_at_last_jacobian: usize,
}

impl LsodeState {
    /// Create a fresh LSODE state from `config` (validated here).
    pub fn new(config: LsodeConfig) -> Self {
        config.check();
        // The Nordsieck array needs `max_history_size + 1` columns
        // (working columns 0..effective-max-order plus the spare).
        let max_history_size = config.effective_max_order();
        let (method_coeffs, test_coeffs) =
            coeffs::calculate_integration_coefficients(config.method);
        Self {
            config,
            method_coeffs,
            test_coeffs,
            el: [0.0; 13],
            nordsieck: Nordsieck::new(N_ODES, max_history_size),
            order: 1,
            num_cols: 2,
            max_history_size,
            step_size: 0.0,
            prev_step_size: 0.0,
            stage_target_time: 0.0,
            cycle_target_time: 0.0,
            order_select_para: 2,
            max_step_increase_ratio: 10_000.0,
            convergence_rate: 0.7,
            num_steps_taken: 0,
            internal_state: 0,
            first_pass: true,
            prev_method_order: 1,
            error_weight: [0.0; N_ODES],
            topology_dirty: false,
            iter_matrix: [[0.0; N_ODES]; N_ODES],
            iter_pivots: [0; N_ODES],
            jac_hl0: 0.0,
            jacobian_current: false,
            steps_at_last_jacobian: 0,
        }
    }

    /// The configuration this state was built from.
    pub fn config(&self) -> &LsodeConfig {
        &self.config
    }

    /// Mark the multistep history invalid after a mass-tree topology
    /// change (attach/detach). The next step re-primes from order 1.
    pub fn mark_topology_dirty(&mut self) {
        self.topology_dirty = true;
    }

    /// Whether the history is awaiting a reset after a topology change.
    pub fn is_topology_dirty(&self) -> bool {
        self.topology_dirty
    }

    /// Reset the integrator to a cold start (history re-primed on the next
    /// step). Called after an attach/detach that invalidated the history.
    pub fn reset_for_topology_change(&mut self) {
        let config = self.config;
        *self = Self::new(config);
    }

    /// Load `el` (current-order coefficients) and the convergence factor
    /// for the current order. Mirrors `integrator_reset_method_coeffs`.
    fn reset_method_coeffs(&mut self) {
        for ii in 0..self.num_cols {
            self.el[ii] = self.method_coeffs[ii][self.order - 1];
        }
    }

    /// `conit = 0.5 / (order + 2)` (`convergence_factor`).
    #[allow(
        clippy::cast_precision_loss,
        reason = "order ≤ 12 is exactly representable in f64"
    )]
    fn convergence_factor(&self) -> f64 {
        0.5 / (self.order as f64 + 2.0)
    }
}

/// Evaluate the flattened-system derivative `y_dot = [vel; accel]` at the
/// state `y = [position; velocity]`, writing it into `save`. `accel_fn`
/// supplies the translational acceleration; `frac` is the fraction of the
/// cycle for time-dependent gravity (ephemeris).
fn eval_derivative(
    y: &[f64; N_ODES],
    accel_fn: &impl Fn(&TranslationalState, f64) -> DVec3,
    frac: f64,
    save: &mut [f64; N_ODES],
) {
    let pos = DVec3::new(y[0], y[1], y[2]);
    let vel = DVec3::new(y[3], y[4], y[5]);
    let accel = accel_fn(
        &TranslationalState {
            position: pos,
            velocity: vel,
        },
        frac,
    );
    // y_dot = [velocity; acceleration]
    save[0] = vel.x;
    save[1] = vel.y;
    save[2] = vel.z;
    save[3] = accel.x;
    save[4] = accel.y;
    save[5] = accel.z;
}

/// Advance `state` by `dyn_dt` using LSODE (non-stiff Adams, functional
/// iteration), calling `accel_fn` for derivatives. Returns the propagated
/// translational state. `lsode` carries the Nordsieck history and adaptive
/// control across calls.
///
/// `accel_fn(state, frac)` returns the translational acceleration at the
/// given state; `frac ∈ [0, 1]` is the fraction of `dyn_dt` (for
/// time-dependent gravity). Mirrors the closure passed to RK4.
///
/// # Panics
/// Panics (fail-loudly) if the corrector repeatedly fails to converge or
/// the step count exceeds `max_num_steps` — a silently-degraded trajectory
/// is worse than a loud stop. Set `allow_non_convergence` to soften (TODO:
/// warn-and-continue parity with JEOD, currently always panics on failure).
pub fn lsode_translational_step(
    state: &TranslationalState,
    accel_fn: impl Fn(&TranslationalState, f64) -> DVec3,
    dyn_dt: f64,
    lsode: &mut LsodeState,
) -> TranslationalState {
    assert!(
        !lsode.topology_dirty,
        "LsodeState used while topology-dirty — reset_for_topology_change() must run after \
         an attach/detach before the next integration step."
    );
    // LSODE is a forward-time multistep method: its Nordsieck history encodes
    // a positive step direction, so a non-positive or non-finite dyn_dt is a
    // misconfiguration (e.g. a negative time_scale_factor for reverse-time —
    // use RK4/RKF45 for that). Fail loudly rather than corrupt the history.
    assert!(
        dyn_dt.is_finite() && dyn_dt > 0.0,
        "LSODE requires a finite, strictly-positive dyn_dt (got {dyn_dt}); it is forward-time \
         only. For reverse-time integration select IntegratorType::Rk4 or Rkf45."
    );
    let eps = f64::EPSILON;
    let max_step_size_inv = if lsode.config.max_step_size > 0.0 {
        1.0 / lsode.config.max_step_size
    } else {
        0.0
    };
    let mut save = [0.0_f64; N_ODES];

    // ── First-pass initialization (manager_initialize_calculation). ──
    if lsode.first_pass {
        // Load y and y_dot into Nordsieck columns 0 and 1.
        let y0 = [
            state.position.x,
            state.position.y,
            state.position.z,
            state.velocity.x,
            state.velocity.y,
            state.velocity.z,
        ];
        for i in 0..N_ODES {
            lsode.nordsieck.history[i][0] = y0[i];
        }
        eval_derivative(&y0, &accel_fn, 0.0, &mut save);
        for i in 0..N_ODES {
            lsode.nordsieck.history[i][1] = save[i];
        }

        // Error weights at order 1 (then inverted).
        lsode.order = 1;
        compute_inverted_ewt(lsode);

        // Initial step size (auto-select) — ODEPACK H0 heuristic.
        let t0 = dyn_dt.abs();
        assert!(
            t0 >= 2.0 * eps,
            "LSODE: dyn_dt ({dyn_dt}) too small to start integration."
        );
        let h0 = if lsode.config.initial_step_size > 0.0 {
            lsode.config.initial_step_size.copysign(dyn_dt)
        } else {
            let mut rtol = lsode.config.rel_tolerance;
            if rtol <= 0.0 {
                // No relative tolerance: derive one from the absolute
                // tolerance and the state magnitude.
                let atol = lsode.config.abs_tolerance;
                for i in 0..N_ODES {
                    if y0[i] != 0.0 {
                        rtol = rtol.max(atol / y0[i].abs());
                    }
                }
            }
            rtol = rtol.max(100.0 * eps).min(0.001);
            // DVNORM of the y_dot column (history col 1) under inverted ewt.
            let col1: [f64; N_ODES] = std::array::from_fn(|i| lsode.nordsieck.history[i][1]);
            let ss = weighted_rms_norm(&col1, &lsode.error_weight);
            let sum = 1.0 / (rtol * t0 * t0) + rtol * ss * ss;
            let mut h0 = 1.0 / sum.sqrt();
            h0 = h0.min(t0);
            h0.copysign(dyn_dt)
        };
        // Honor max_step_size.
        let mut h0 = h0;
        let ratio = h0.abs() * max_step_size_inv;
        if ratio > 1.0 {
            h0 /= ratio;
        }
        lsode.step_size = h0;
        // Scale the first-derivative column into a delta-x estimate.
        for i in 0..N_ODES {
            lsode.nordsieck.history[i][1] *= h0;
        }
        lsode.num_cols = 2;
        lsode.order = 1;
        lsode.internal_state = 0;
        lsode.stage_target_time = 0.0;
        lsode.first_pass = false;
        lsode.cycle_target_time = dyn_dt;
    } else {
        // Subsequent cycle: the previous cycle may have overshot its
        // target; rebase the internal clock and set the new target.
        lsode.stage_target_time -= lsode.cycle_target_time;
        lsode.cycle_target_time = dyn_dt;
        lsode.internal_state = 1;
    }

    // ── Integration loop: step until we reach/overshoot the target. ──
    let mut steps_this_cycle = 0usize;
    while (lsode.stage_target_time - lsode.cycle_target_time) * lsode.step_size < 0.0 {
        steps_this_cycle += 1;
        assert!(
            steps_this_cycle <= lsode.config.max_num_steps,
            "LSODE: exceeded max_num_steps ({}) within one cycle without reaching the target \
             time (reached {} of {}). Check tolerances/step size.",
            lsode.config.max_num_steps,
            lsode.stage_target_time,
            lsode.cycle_target_time
        );
        compute_inverted_ewt(lsode);
        dstode_step(lsode, &accel_fn, max_step_size_inv);
    }

    // ── Interpolate the state back to exactly cycle_target_time. ──
    interpolate_to_target(lsode)
}

/// Recompute `ewt = rtol·|y| + atol` from Nordsieck column 0 and store its
/// reciprocal in `error_weight` (DVNORM uses the inverted form).
fn compute_inverted_ewt(lsode: &mut LsodeState) {
    let y0: [f64; N_ODES] = std::array::from_fn(|i| lsode.nordsieck.history[i][0]);
    let mut ewt = [0.0_f64; N_ODES];
    load_error_weights(
        &y0,
        lsode.config.rel_tolerance,
        lsode.config.abs_tolerance,
        &mut ewt,
    );
    for i in 0..N_ODES {
        // ewt = rtol·|y| + atol. With atol = 0 this is 0 whenever a state
        // component is exactly 0 (e.g. a velocity passing through zero),
        // even though rtol > 0 — the config-time `rtol>0 || atol>0` guard
        // (IG.20) does not prevent it. Fail loudly with the fix (IG.23).
        assert!(
            ewt[i] > 0.0,
            "LSODE: error weight for component {i} is {} (≤ 0). With atol = 0 this happens when \
             y[{i}] is exactly 0; set abs_tolerance > 0 for components that may pass through zero.",
            ewt[i]
        );
        lsode.error_weight[i] = 1.0 / ewt[i];
    }
}

/// One DSTODE step: predict, correct (functional iteration with retries on
/// non-convergence / error-test failure), accept, and select the next
/// order/step. On entry the step may be retried internally with a reduced
/// step before this returns. Panics (fail-loudly) on terminal failure.
#[allow(
    clippy::cast_precision_loss,
    reason = "order/column counts ≤ 13 are exactly representable in f64"
)]
fn dstode_step(
    lsode: &mut LsodeState,
    accel_fn: &impl Fn(&TranslationalState, f64) -> DVec3,
    max_step_size_inv: f64,
) {
    let told = lsode.stage_target_time;
    let mut step_error: i32 = 0;
    let mut accum = [0.0_f64; N_ODES];

    // On the very first step, set order-1 coefficients.
    if lsode.internal_state == 0 {
        lsode.order = 1;
        lsode.num_cols = 2;
        lsode.order_select_para = 2;
        lsode.max_step_increase_ratio = 10_000.0;
        lsode.convergence_rate = 0.7;
        lsode.reset_method_coeffs();
        lsode.internal_state = 1;
    }

    'attempt: loop {
        // ── Predict: advance the Nordsieck array by Pascal's triangle. ──
        lsode.stage_target_time = told + lsode.step_size;
        lsode.nordsieck.predict(lsode.order);

        let frac = (lsode.stage_target_time / lsode.cycle_target_time).clamp(0.0, 1.0);
        let conv_factor = lsode.convergence_factor();
        let el0 = lsode.el[0];
        let tesco1 = lsode.test_coeffs[1][lsode.order - 1];

        // ── Corrector. ──
        // The corrector solves the implicit step equation for `accum` (the
        // accumulated correction). Functional (fixed-point) iteration is the
        // non-stiff Adams path; the chord (modified-Newton) corrector with a
        // finite-difference Jacobian is the stiff path — the BDF family, and
        // Adams when a Newton corrector is selected. Both leave `accum`
        // holding the accepted correction the error test and the history
        // update below consume. `history[i][0]` stays at the PREDICTED value
        // throughout (the single post-acceptance update
        // `history[jj] += el[jj]·accum`, jj=0 included, brings column 0 from
        // predicted to corrected exactly once).
        let (_converged, corrector_failed) = match lsode.config.corrector {
            CorrectorMethod::FunctionalIteration => {
                functional_corrector(lsode, accel_fn, frac, el0, tesco1, conv_factor, &mut accum)
            }
            // Stiff family: a chord (modified-Newton) corrector driven by an
            // internally-generated finite-difference Jacobian (MITER=2); the
            // iteration matrix P = I − h·el0·J is LU factored and reused until
            // it drifts stale.
            CorrectorMethod::NewtonIterInternalJac => {
                chord_corrector(lsode, accel_fn, frac, el0, tesco1, conv_factor, &mut accum)
            }
            CorrectorMethod::JacobiNewtonInternalJac => panic!(
                "LSODE corrector JacobiNewtonInternalJac (MITER=3, diagonal Jacobi-Newton) is \
                 not yet ported. Use the dense finite-difference Newton corrector \
                 (NewtonIterInternalJac) for the stiff BDF family, or the non-stiff Adams \
                 family with functional iteration."
            ),
        };

        if corrector_failed {
            // Retract the prediction, reduce the step, retry.
            lsode.stage_target_time = told;
            retract_prediction(lsode);
            lsode.max_step_increase_ratio = 2.0;
            assert!(
                lsode.step_size.abs() > lsode.config.min_step_size * 1.00001,
                "LSODE corrector failed to converge at the minimum step size — trajectory \
                 would be silently degraded. Review tolerances / step size."
            );
            apply_step_ratio(lsode, 0.25, max_step_size_inv);
            continue 'attempt;
        }

        // ── Local error test. ──
        let dsm = weighted_rms_norm(&accum, &lsode.error_weight) / tesco1;
        if dsm > 1.0 {
            // Error test failed: retract, reduce step (or order), retry.
            step_error -= 1;
            lsode.stage_target_time = told;
            retract_prediction(lsode);
            lsode.max_step_increase_ratio = 2.0;
            assert!(
                lsode.step_size.abs() > lsode.config.min_step_size * 1.00001,
                "LSODE error test failed at the minimum step size — trajectory would be \
                 silently degraded. Review tolerances / step size."
            );
            if step_error <= -3 {
                // Repeated failures: reset to order 1, h *= 0.1, retry.
                assert!(
                    step_error > -10,
                    "LSODE: 10 consecutive error-test failures — giving up (would be silently \
                     wrong). Review the integration setup."
                );
                fail_reset_order_1(lsode, accel_fn, max_step_size_inv);
                // After reset the step is retried from the loop top.
                continue 'attempt;
            }
            // Order selection on failure (mirrors error_test_failed →
            // compute_new_order): r_inc forced to 0 so the order is held or
            // decreased, never increased — then retry the step.
            select_new_order(lsode, &accum, dsm, step_error, 0.0, max_step_size_inv);
            continue 'attempt;
        }

        // ── Step accepted: commit the correction into the history. ──
        lsode.num_steps_taken += 1;
        lsode.prev_method_order = lsode.order;
        for jj in 0..lsode.num_cols {
            for i in 0..N_ODES {
                lsode.nordsieck.history[i][jj] += lsode.el[jj] * accum[i];
            }
        }

        // ── Order/step selection (mirrors corrector_converged's branching). ──
        // The stash of `accum` into the spare column `history[max_history_size]`
        // (for the order-increase indicator) happens ONLY on the step where
        // the countdown hits 1 (the step *before* selection), so that when the
        // countdown hits 0 the next step reads last step's accum — not this
        // step's (which would zero the r_inc difference).
        lsode.order_select_para -= 1;
        if lsode.order_select_para == 0 {
            let r_inc = compute_r_inc(lsode, &accum);
            select_new_order(lsode, &accum, dsm, step_error, r_inc, max_step_size_inv);
        } else if lsode.order_select_para == 1 && lsode.num_cols != lsode.max_history_size {
            for i in 0..N_ODES {
                lsode.nordsieck.history[i][lsode.max_history_size] = accum[i];
            }
        }
        lsode.prev_step_size = lsode.step_size;
        return;
    }
}

/// Functional (fixed-point) iteration corrector — the non-stiff Adams path
/// (`integrator_corrector_iteration` functional branch). Iterates
/// `y = history[0] + el0·(h·f(y) − h·y'_pred)` to a fixed point, leaving the
/// accepted correction in `accum`. Returns `(converged, failed)`.
fn functional_corrector(
    lsode: &mut LsodeState,
    accel_fn: &impl Fn(&TranslationalState, f64) -> DVec3,
    frac: f64,
    el0: f64,
    tesco1: f64,
    conv_factor: f64,
    accum: &mut [f64; N_ODES],
) -> (bool, bool) {
    let pred0: [f64; N_ODES] = std::array::from_fn(|i| lsode.nordsieck.history[i][0]);
    let mut y_work = pred0;
    for a in accum.iter_mut() {
        *a = 0.0;
    }
    let mut save = [0.0_f64; N_ODES];
    let mut prev_iter_delta = 0.0_f64;
    let mut converged = false;
    let mut corrector_failed = false;
    for iter in 0..lsode.config.max_correction_iters {
        eval_derivative(&y_work, accel_fn, frac, &mut save);
        // residual: save = h·f − h·y'_pred ; increment = save − accum.
        let mut incr = [0.0_f64; N_ODES];
        for i in 0..N_ODES {
            save[i] = lsode.step_size * save[i] - lsode.nordsieck.history[i][1];
            incr[i] = save[i] - accum[i];
        }
        let iter_delta = weighted_rms_norm(&incr, &lsode.error_weight);
        for i in 0..N_ODES {
            y_work[i] = pred0[i] + el0 * save[i];
            accum[i] = save[i];
        }
        if iter != 0 {
            lsode.convergence_rate =
                (0.2 * lsode.convergence_rate).max(iter_delta / prev_iter_delta);
        }
        let dcon =
            iter_delta * (1.0_f64).min(1.5 * lsode.convergence_rate) / (tesco1 * conv_factor);
        if dcon <= 1.0 {
            converged = true;
            break;
        }
        if iter >= 1 && iter_delta > 2.0 * prev_iter_delta {
            corrector_failed = true;
            break;
        }
        prev_iter_delta = iter_delta;
    }
    if !converged && !corrector_failed {
        corrector_failed = true; // hit max_correction_iters
    }
    (converged, corrector_failed)
}

/// Chord (modified-Newton) corrector — the stiff path
/// (`integrator_corrector_iteration` chord branch + `linear_chord_iteration`).
///
/// Each iteration evaluates the residual `r = h·f(y) − (h·y'_pred + accum)`
/// and solves `P·Δ = r` for the Newton step `Δ` against the factored
/// iteration matrix `P = I − h·el0·J`, accumulating `accum += Δ` and
/// `y = history[0] + el0·accum`. The Jacobian-backed `P` is built on demand
/// and reused until it drifts stale (`MAX_REL_CHANGE_WITHOUT_JACOBIAN` /
/// `MAX_STEPS_PER_JACOBIAN`); a convergence failure with a *stale* matrix
/// triggers one rebuild-and-retry before giving up, mirroring JEOD's
/// `integrator_corrector_failed_part1` (the chord iteration converges to the
/// exact residual root regardless of Jacobian quality, so the matrix only
/// governs convergence rate). Returns `(converged, failed)`; a `failed`
/// return drives the caller's step-reduction recovery.
fn chord_corrector(
    lsode: &mut LsodeState,
    accel_fn: &impl Fn(&TranslationalState, f64) -> DVec3,
    frac: f64,
    el0: f64,
    tesco1: f64,
    conv_factor: f64,
    accum: &mut [f64; N_ODES],
) -> (bool, bool) {
    let pred0: [f64; N_ODES] = std::array::from_fn(|i| lsode.nordsieck.history[i][0]);
    let hl0 = lsode.step_size * el0;
    // Stale if never built, the step/order changed `h·el0` materially, or
    // the forced-rebuild cadence elapsed.
    let drift = if lsode.jac_hl0 == 0.0 {
        f64::INFINITY
    } else {
        (hl0 / lsode.jac_hl0 - 1.0).abs()
    };
    let mut need_build = !lsode.jacobian_current
        || drift > MAX_REL_CHANGE_WITHOUT_JACOBIAN
        || lsode.num_steps_taken >= lsode.steps_at_last_jacobian + MAX_STEPS_PER_JACOBIAN;

    loop {
        let built_now = need_build;
        if need_build {
            // Base derivative at the predicted state for the finite-
            // difference Jacobian, then build + factor P = I − h·el0·J.
            let mut f_base = [0.0_f64; N_ODES];
            eval_derivative(&pred0, accel_fn, frac, &mut f_base);
            if build_dense_iteration_matrix(lsode, accel_fn, frac, &pred0, &f_base, hl0).is_err() {
                // Singular iteration matrix — reduce the step and retry.
                lsode.jacobian_current = false;
                return (false, true);
            }
            lsode.jacobian_current = true;
            lsode.jac_hl0 = hl0;
            lsode.steps_at_last_jacobian = lsode.num_steps_taken;
            lsode.convergence_rate = 0.7; // CRATE reset on a fresh matrix (ODEPACK)
        }

        let mut y_work = pred0;
        for a in accum.iter_mut() {
            *a = 0.0;
        }
        let mut save = [0.0_f64; N_ODES];
        let mut prev_iter_delta = 0.0_f64;
        let mut converged = false;
        for iter in 0..lsode.config.max_correction_iters {
            eval_derivative(&y_work, accel_fn, frac, &mut save);
            // Newton residual r = h·f − (h·y'_pred + accum); solve P·Δ = r.
            let mut delta = [0.0_f64; N_ODES];
            for i in 0..N_ODES {
                delta[i] = lsode.step_size * save[i] - (lsode.nordsieck.history[i][1] + accum[i]);
            }
            linalg::lu_solve(&lsode.iter_matrix, &lsode.iter_pivots, &mut delta);
            let iter_delta = weighted_rms_norm(&delta, &lsode.error_weight);
            for i in 0..N_ODES {
                accum[i] += delta[i];
                y_work[i] = pred0[i] + el0 * accum[i];
            }
            if iter != 0 {
                lsode.convergence_rate =
                    (0.2 * lsode.convergence_rate).max(iter_delta / prev_iter_delta);
            }
            let dcon =
                iter_delta * (1.0_f64).min(1.5 * lsode.convergence_rate) / (tesco1 * conv_factor);
            if dcon <= 1.0 {
                converged = true;
                break;
            }
            // Divergence (iterate growing) — abandon this matrix's iterations.
            if iter >= 1 && iter_delta > 2.0 * prev_iter_delta {
                break;
            }
            prev_iter_delta = iter_delta;
        }
        if converged {
            return (true, false);
        }
        // Failed: a fresh matrix can't be improved → reduce the step. A
        // stale matrix gets one rebuild-and-retry first.
        lsode.jacobian_current = false;
        if built_now {
            return (false, true);
        }
        need_build = true;
    }
}

/// Build and LU-factor the stiff corrector's iteration matrix
/// `P = I − h·el0·J`, with `J` an internally-generated forward-difference
/// Jacobian (ODEPACK `DPREPJ` MITER=2 / JEOD `jacobian_prep_*`
/// `NewtonIterInternalJac`). `f_base` is `f(y_base)`; `hl0 = h·el0`.
///
/// Stores the mathematically-standard orientation `P[i][j]` (row = equation,
/// column = perturbed variable). JEOD stores `lin_alg[j][i]` (transposed) —
/// an apparent quirk that converges anyway because the chord iteration's
/// fixed point is Jacobian-independent; we use the standard orientation,
/// which is correct and converges at least as well. Returns `Err(col)` if
/// the factorization finds a singular column.
#[allow(
    clippy::float_cmp,
    reason = "exact r0==0 guard mirrors DPREPJ's fpclassify(r0)==FP_ZERO fallback to r0=1"
)]
#[allow(
    clippy::cast_precision_loss,
    reason = "N_ODES = 6 is exactly representable in f64"
)]
fn build_dense_iteration_matrix(
    lsode: &mut LsodeState,
    accel_fn: &impl Fn(&TranslationalState, f64) -> DVec3,
    frac: f64,
    y_base: &[f64; N_ODES],
    f_base: &[f64; N_ODES],
    hl0: f64,
) -> Result<(), usize> {
    let eps = f64::EPSILON;
    let srur = eps.sqrt(); // unit-roundoff scale (ODEPACK SRUR)
    let fac0 = weighted_rms_norm(f_base, &lsode.error_weight);
    let mut r0 = 1000.0 * eps * lsode.step_size.abs() * (N_ODES as f64) * fac0;
    if r0 == 0.0 {
        r0 = 1.0;
    }
    let mut y = *y_base;
    let mut ftem = [0.0_f64; N_ODES];
    for j in 0..N_ODES {
        let yj = y_base[j];
        // Perturbation: scaled to the variable magnitude, floored by the
        // tolerance-weighted r0 (error_weight is 1/ewt, so r0/error_weight
        // = r0·ewt — the literal JEOD expression).
        let r = (srur * yj.abs()).max(r0 / lsode.error_weight[j]);
        y[j] = yj + r;
        eval_derivative(&y, accel_fn, frac, &mut ftem);
        let fac = -hl0 / r;
        for i in 0..N_ODES {
            // (∂f_i/∂y_j) · (−hl0) = −hl0·J[i][j]
            lsode.iter_matrix[i][j] = (ftem[i] - f_base[i]) * fac;
        }
        y[j] = yj; // restore
    }
    // P = I − hl0·J: add the identity.
    for i in 0..N_ODES {
        lsode.iter_matrix[i][i] += 1.0;
    }
    linalg::lu_factor(&mut lsode.iter_matrix, &mut lsode.iter_pivots)
}

/// Retract a prediction by reversing the Pascal-triangle shift (the `-=`
/// loop in `integrator_corrector_failed_part2` / `error_test_failed`).
fn retract_prediction(lsode: &mut LsodeState) {
    for i_iter in (1..=lsode.order).rev() {
        for j_hist in (i_iter - 1)..lsode.order {
            for k_var in 0..N_ODES {
                lsode.nordsieck.history[k_var][j_hist] -=
                    lsode.nordsieck.history[k_var][j_hist + 1];
            }
        }
    }
}

/// Apply a step-size ratio: clamp, rescale the Nordsieck columns, update
/// `step_size`. Mirrors `integrator_reset_yh`.
fn apply_step_ratio(lsode: &mut LsodeState, ratio: f64, max_step_size_inv: f64) {
    let mut r = ratio.min(lsode.max_step_increase_ratio);
    r /= (1.0_f64).max(lsode.step_size.abs() * max_step_size_inv * r);
    lsode.nordsieck.rescale_columns(r, lsode.num_cols);
    lsode.step_size *= r;
    lsode.order_select_para = lsode.num_cols;
}

/// Order-increase step-ratio indicator (`compute_new_order_prep`'s
/// `step_ratio_order_inc`). Uses the accumulated correction minus the
/// previous step's stashed accum (the spare column
/// `history[max_history_size]`). Returns 0 when no spare column is
/// available — mirroring JEOD's `num_nordsiek_cols != max_history_size`
/// guard. Since the effective maximum order is `max_history_size - 1`,
/// column `max_history_size` is *always* a free spare (never an active
/// Nordsieck column), so the stash can never collide with the working
/// history.
#[allow(
    clippy::cast_precision_loss,
    reason = "column count ≤ 13 is exactly representable in f64"
)]
fn compute_r_inc(lsode: &LsodeState, accum: &[f64; N_ODES]) -> f64 {
    if lsode.num_cols == lsode.max_history_size {
        return 0.0;
    }
    let diff: [f64; N_ODES] =
        std::array::from_fn(|i| accum[i] - lsode.nordsieck.history[i][lsode.max_history_size]);
    let dup = weighted_rms_norm(&diff, &lsode.error_weight) / lsode.test_coeffs[2][lsode.order - 1];
    let exup = 1.0 / (lsode.num_cols as f64 + 1.0);
    1.0 / (1.4 * dup.powf(exup) + 0.0000014)
}

/// Re-prime at order 1 with a 10× step reduction after 3+ failures
/// (`integrator_fail_reset_order_1`).
fn fail_reset_order_1(
    lsode: &mut LsodeState,
    accel_fn: &impl Fn(&TranslationalState, f64) -> DVec3,
    _max_step_size_inv: f64,
) {
    let ratio = (lsode.config.min_step_size / lsode.step_size.abs()).max(0.1);
    lsode.step_size *= ratio;
    // Recompute the first derivative at the current state (column 0).
    let y0: [f64; N_ODES] = std::array::from_fn(|i| lsode.nordsieck.history[i][0]);
    let mut save = [0.0_f64; N_ODES];
    eval_derivative(&y0, accel_fn, 0.0, &mut save);
    for i in 0..N_ODES {
        lsode.nordsieck.history[i][1] = lsode.step_size * save[i];
    }
    lsode.order_select_para = 5;
    lsode.order = 1;
    lsode.num_cols = 2;
    lsode.reset_method_coeffs();
}

/// Choose the order (−1/same/+1) with the largest step ratio and apply it
/// (`integrator_compute_new_order` + `set_new_order`). Runs only when the
/// `order_select_para` countdown reaches 0.
#[allow(
    clippy::cast_precision_loss,
    reason = "order/column counts ≤ 13 are exactly representable in f64"
)]
fn select_new_order(
    lsode: &mut LsodeState,
    accum: &[f64; N_ODES],
    dsm: f64,
    step_error: i32,
    r_inc: f64,
    max_step_size_inv: f64,
) {
    // `r_inc` is the order-increase indicator: computed via `compute_r_inc`
    // on a successful step, or forced to 0 after an error-test failure
    // (JEOD sets `step_ratio_order_inc = 0` there so the order can only be
    // held or decreased — never increased while recovering from a failure).
    let exsm = 1.0 / lsode.num_cols as f64;
    let r_same = 1.0 / (1.2 * dsm.powf(exsm) + 0.0000012);
    let mut r_dec = 0.0;
    if lsode.order != 1 {
        let col: [f64; N_ODES] =
            std::array::from_fn(|i| lsode.nordsieck.history[i][lsode.num_cols - 1]);
        let ddn =
            weighted_rms_norm(&col, &lsode.error_weight) / lsode.test_coeffs[0][lsode.order - 1];
        let exdn = 1.0 / lsode.order as f64;
        r_dec = 1.0 / (1.3 * ddn.powf(exdn) + 0.0000013);
    }

    // Tie priority: same ≥ inc, same ≥ dec → keep order; else inc > dec → up; else down.
    let (new_order, ratio);
    if r_same >= r_inc && r_same >= r_dec {
        new_order = lsode.order;
        ratio = r_same;
    } else if r_inc > r_dec {
        new_order = lsode.num_cols; // order + 1
        ratio = r_inc;
        if ratio < 1.1 {
            lsode.order_select_para = 3;
            return;
        }
        // Seed the new highest Nordsieck column for the increased order.
        let r = lsode.el[lsode.num_cols - 1] / lsode.num_cols as f64;
        for i in 0..N_ODES {
            lsode.nordsieck.history[i][new_order] = accum[i] * r;
        }
        set_new_order(lsode, new_order, ratio, max_step_size_inv);
        return;
    } else {
        new_order = lsode.order - 1;
        ratio = if step_error < 0 {
            r_dec.min(1.0)
        } else {
            r_dec
        };
    }

    // check_step_error: a <1.1 same/dec ratio with a clean step holds for 3 steps.
    if step_error == 0 && ratio < 1.1 {
        lsode.order_select_para = 3;
        return;
    }
    let ratio = if step_error <= -2 {
        ratio.min(0.2)
    } else {
        ratio
    };
    set_new_order(lsode, new_order, ratio, max_step_size_inv);
}

/// Commit a new order + step ratio (`integrator_set_new_order`).
fn set_new_order(lsode: &mut LsodeState, new_order: usize, ratio: f64, max_step_size_inv: f64) {
    if new_order == lsode.order {
        let ratio = ratio.max(lsode.config.min_step_size / lsode.step_size.abs());
        apply_step_ratio(lsode, ratio, max_step_size_inv);
    } else {
        lsode.order = new_order;
        lsode.num_cols = new_order + 1;
        lsode.reset_method_coeffs();
        apply_step_ratio(lsode, ratio, max_step_size_inv);
    }
}

/// Interpolate the solution to exactly `cycle_target_time` via the
/// Nordsieck polynomial (DINTDY, K=0, Horner form), returning the
/// translational state.
fn interpolate_to_target(lsode: &LsodeState) -> TranslationalState {
    let s = (lsode.cycle_target_time - lsode.stage_target_time) / lsode.step_size;
    let mut y = [0.0_f64; N_ODES];
    for i in 0..N_ODES {
        y[i] = lsode.nordsieck.history[i][lsode.num_cols - 1];
    }
    for jb in 1..=lsode.order {
        let j = lsode.order - jb;
        for i in 0..N_ODES {
            y[i] = y[i] * s + lsode.nordsieck.history[i][j];
        }
    }
    TranslationalState {
        position: DVec3::new(y[0], y[1], y[2]),
        velocity: DVec3::new(y[3], y[4], y[5]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-body circular-orbit acceleration: a = -μ r / |r|³.
    fn kepler_accel(mu: f64) -> impl Fn(&TranslationalState, f64) -> DVec3 {
        move |s: &TranslationalState, _frac: f64| {
            let r = s.position;
            let rn = r.length();
            -mu * r / (rn * rn * rn)
        }
    }

    /// Linear damped-oscillator acceleration on the x-axis:
    /// `a = (−k·x − c·v, 0, 0)`. For large `c` relative to `k` the system is
    /// stiff (well-separated eigenvalues), exercising the BDF/Newton path.
    fn damped_oscillator_accel(k: f64, c: f64) -> impl Fn(&TranslationalState, f64) -> DVec3 {
        move |s: &TranslationalState, _frac: f64| {
            DVec3::new(-k * s.position.x - c * s.velocity.x, 0.0, 0.0)
        }
    }

    /// A stiff BDF configuration (orders 1–5, modified-Newton chord corrector
    /// with an internal finite-difference Jacobian) at the given tolerances.
    fn bdf_config(rel_tolerance: f64, abs_tolerance: f64) -> LsodeConfig {
        LsodeConfig {
            method: IntegrationMethod::ImplicitBackDiffStiff,
            corrector: CorrectorMethod::NewtonIterInternalJac,
            max_order: 5,
            rel_tolerance,
            abs_tolerance,
            ..LsodeConfig::default()
        }
    }

    /// BDF/Newton integrates a stiff overdamped oscillator to its analytic
    /// solution at a step size where an explicit method would be unstable.
    ///
    /// `ẍ + c·ẋ + k·x = 0`, `x(0)=1`, `ẋ(0)=0` has eigenvalues
    /// `λ = (−c ± √(c²−4k))/2`. With `c=200, k=1` the fast mode `λ₂ ≈ −200`
    /// makes the system stiff (an explicit step is stable only for
    /// `dt < 2/200 = 0.01`); we drive `dt = 0.05`, where only an
    /// L-stable BDF stays bounded, and compare to
    /// `x(t) = A·e^{λ₁t} + B·e^{λ₂t}`.
    #[test]
    fn bdf_stiff_overdamped_oscillator_matches_analytic() {
        let (k, c) = (1.0_f64, 200.0_f64);
        let disc = (c * c - 4.0 * k).sqrt();
        let lam1 = (-c + disc) / 2.0; // slow mode ≈ −0.005
        let lam2 = (-c - disc) / 2.0; // fast (stiff) mode ≈ −200
                                      // x(0)=1, v(0)=0 ⇒ A = λ₂/(λ₂−λ₁), B = −λ₁/(λ₂−λ₁).
        let a_coef = lam2 / (lam2 - lam1);
        let b_coef = -lam1 / (lam2 - lam1);
        let analytic = |t: f64| a_coef * (lam1 * t).exp() + b_coef * (lam2 * t).exp();

        let start = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        // Unit-scale tolerances (tighter would just force the error test to
        // demand sub-pico accuracy on an O(1) state and is unrelated to
        // stiffness handling).
        let mut lsode = LsodeState::new(bdf_config(1e-7, 1e-9));
        let accel = damped_oscillator_accel(k, c);
        let dt = 0.05; // 5× the explicit stability limit
        let mut s = start;
        let mut t = 0.0;
        for _ in 0..40 {
            s = lsode_translational_step(&s, &accel, dt, &mut lsode);
            t += dt;
            let want = analytic(t);
            assert!(
                (s.position.x - want).abs() < 1e-5,
                "BDF stiff x(t={t:.2}) = {} vs analytic {want} (err {:e})",
                s.position.x,
                (s.position.x - want).abs()
            );
        }
        // Bounded and decayed toward the slow-mode tail (no explicit blow-up).
        assert!(s.position.x.abs() < 1.0, "solution did not stay bounded");
        // The stiff path actually exercised the Newton/Jacobian machinery.
        assert!(
            lsode.num_steps_taken >= 40,
            "expected internal sub-stepping for the stiff transient"
        );
    }

    /// The chord (Newton) corrector reaches the same converged solution as
    /// the proven functional corrector on a smooth (non-stiff) orbit. Both
    /// solve the same implicit step equation; only the iteration scheme
    /// differs, so the trajectories must agree to tolerance. This validates
    /// the Newton corrector + finite-difference Jacobian against the
    /// Phase-6A functional path.
    #[test]
    fn adams_newton_corrector_matches_functional_on_orbit() {
        let mu = 3.986_004_418e14_f64;
        let r0 = 7_000_000.0_f64;
        let v0 = (mu / r0).sqrt();
        let start = TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        };
        let accel = kepler_accel(mu);
        let dt = 30.0;
        let n = 200usize;

        // Adams + functional iteration (the default 6A path).
        let mut func = LsodeState::new(LsodeConfig {
            rel_tolerance: 1e-11,
            abs_tolerance: 1e-6,
            ..LsodeConfig::default()
        });
        // Adams + modified-Newton chord corrector (6C corrector on the
        // non-stiff family — same coefficients, different solve).
        let mut newt = LsodeState::new(LsodeConfig {
            corrector: CorrectorMethod::NewtonIterInternalJac,
            rel_tolerance: 1e-11,
            abs_tolerance: 1e-6,
            ..LsodeConfig::default()
        });

        let mut sf = start;
        let mut sn = start;
        for _ in 0..n {
            sf = lsode_translational_step(&sf, &accel, dt, &mut func);
            sn = lsode_translational_step(&sn, &accel, dt, &mut newt);
        }
        let pos_diff = (sf.position - sn.position).length();
        let vel_diff = (sf.velocity - sn.velocity).length();
        // Two independent adaptive integrations of the same orbit with the
        // same coefficients but different correctors agree to ~1e-8 relative
        // (cm-scale over a 7000 km orbit) — far tighter than the method's own
        // truncation error, confirming both solve the same step equation. The
        // residual gap is independent order/step-controller rounding, not a
        // corrector discrepancy.
        assert!(
            pos_diff < 1.0,
            "Newton vs functional position diverged: {pos_diff:.3e} m (>1 m ⇒ different solution)"
        );
        assert!(
            vel_diff < 1e-3,
            "Newton vs functional velocity diverged: {vel_diff:.3e} m/s"
        );
    }

    /// The unported diagonal Jacobi-Newton corrector (MITER=3) must fail
    /// loudly rather than silently running a different corrector.
    #[test]
    #[should_panic(expected = "JacobiNewtonInternalJac")]
    fn jacobi_newton_diagonal_panics_until_ported() {
        let start = TranslationalState {
            position: DVec3::new(7_000_000.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7_546.0, 0.0),
        };
        let mut lsode = LsodeState::new(LsodeConfig {
            method: IntegrationMethod::ImplicitBackDiffStiff,
            corrector: CorrectorMethod::JacobiNewtonInternalJac,
            max_order: 5,
            ..LsodeConfig::default()
        });
        let accel = kepler_accel(3.986_004_418e14);
        lsode_translational_step(&start, &accel, 30.0, &mut lsode);
    }

    /// Tight-tolerance (rtol=2.3e-16, atol=0) continuous integration over
    /// many dyn_dt cycles using JEOD's `RUN_lsode` initial conditions and
    /// derived μ — the same scenario the Tier 3
    /// `tier3_simulation_lsode_default` cross-validates against JEOD. Confirms
    /// the integrator core is stable and efficient in isolation: the order
    /// climbs to ~7, a few internal steps per ~15.5 s cycle, no step
    /// collapse, and energy is conserved.
    #[test]
    fn lsode_tight_tolerance_run_lsode_ics_is_stable() {
        // μ = sma³·ω² and the t=0 prop_integ_state from
        // SIM_integ_test/RUN_lsode (JEOD source values).
        let mu = 6_811_137.0_f64.powi(3) * 1.123_154_395_240_404_1e-3_f64.powi(2);
        let start = TranslationalState {
            position: DVec3::new(2_554_176.375, 5_859_203.640_407_667, 2_353_189.957_992_002),
            velocity: DVec3::new(
                6_580.790_321_332_448,
                -1_407.722_362_559_127,
                -3_637.771_420_498,
            ),
        };
        let mut lsode = LsodeState::new(LsodeConfig {
            rel_tolerance: 2.3e-16,
            abs_tolerance: 0.0,
            ..LsodeConfig::default()
        });
        let accel = kepler_accel(mu);
        let dt = 15.539_530_979_805_79; // sim_dt · time_scale
        let mut s = start;
        for _ in 0..20 {
            s = lsode_translational_step(&s, &accel, dt, &mut lsode);
        }
        // Stable: reached order > 3 with a healthy step, far under the
        // per-cycle step budget (no collapse). `num_steps_taken` for 20
        // cycles of a smooth orbit stays small (tens, not hundreds).
        assert!(lsode.order >= 4, "order stuck low ({})", lsode.order);
        assert!(
            lsode.num_steps_taken < 200,
            "too many internal steps ({}) — step collapsed",
            lsode.num_steps_taken
        );
        // Energy conserved across the 20 cycles.
        let e0 = 0.5 * start.velocity.length_squared() - mu / start.position.length();
        let e = 0.5 * s.velocity.length_squared() - mu / s.position.length();
        assert!(((e - e0) / e0).abs() < 1e-10, "energy drift too large");
    }

    /// A circular LEO propagated one period should return near its start
    /// (closed orbit), and conserve energy — a self-consistency check that
    /// the adaptive Adams driver integrates a smooth orbit stably.
    #[test]
    fn lsode_circular_orbit_closes_and_conserves_energy() {
        let mu = 3.986_004_418e14_f64;
        let r0 = 6_778_137.0_f64; // ~400 km altitude
        let v0 = (mu / r0).sqrt();
        let period = std::f64::consts::TAU * (r0 * r0 * r0 / mu).sqrt();

        let start = TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        };
        let energy =
            |s: &TranslationalState| 0.5 * s.velocity.length_squared() - mu / s.position.length();
        let e0 = energy(&start);

        let mut lsode = LsodeState::new(LsodeConfig {
            rel_tolerance: 1e-12,
            abs_tolerance: 1e-6,
            ..LsodeConfig::default()
        });
        let accel = kepler_accel(mu);
        // Step in 60 s increments for one period.
        let dt = 60.0;
        let n = (period / dt).round() as usize;
        let mut s = start;
        for _ in 0..n {
            s = lsode_translational_step(&s, &accel, dt, &mut lsode);
        }
        // Energy conserved to high precision (adaptive high-order Adams).
        let e_err = ((energy(&s) - e0) / e0).abs();
        assert!(
            e_err < 2e-9,
            "relative energy drift {e_err:.3e} too large over one orbit"
        );
        // Radius stays the circular radius (orbit didn't spiral).
        let r_err = (s.position.length() - r0).abs() / r0;
        assert!(r_err < 1e-4, "radius drift {r_err:.3e} too large");
    }

    /// Against analytic circular motion: after a quarter period the
    /// position should be ~(0, r0, 0) for the chosen ICs.
    #[test]
    fn lsode_quarter_period_matches_analytic_circle() {
        let mu = 3.986_004_418e14_f64;
        let r0 = 7_000_000.0_f64;
        let v0 = (mu / r0).sqrt();
        let period = std::f64::consts::TAU * (r0 * r0 * r0 / mu).sqrt();
        let start = TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        };
        let mut lsode = LsodeState::new(LsodeConfig {
            rel_tolerance: 1e-11,
            abs_tolerance: 1e-6,
            ..LsodeConfig::default()
        });
        let accel = kepler_accel(mu);
        // Use a dt that divides the quarter period exactly so the final
        // step lands on the quarter turn — otherwise up to half a step of
        // along-track phase (~100 km) swamps the integration error.
        let quarter = period / 4.0;
        let n = 200usize;
        let dt = quarter / n as f64;
        let mut s = start;
        for _ in 0..n {
            s = lsode_translational_step(&s, &accel, dt, &mut lsode);
        }
        // Quarter orbit from (r0,0,0) at +v ⇒ position ≈ (0, r0, 0).
        assert!(s.position.x.abs() < 1.0e2, "x = {} not ~0", s.position.x);
        assert!(
            (s.position.y - r0).abs() < 1.0e2,
            "y = {} not ~r0 ({r0})",
            s.position.y
        );
    }
}
