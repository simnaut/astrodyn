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
//! ## Phase status (#200)
//!
//! Phase 6A — non-stiff implicit Adams with functional-iteration corrector
//! — is implemented here ([`lsode_translational_step`]). The stiff BDF
//! family (Jacobian + chord corrector) is deferred (Phase 6C); selecting it
//! panics in [`LsodeConfig::check`] unless paired with a Newton corrector,
//! and the Newton path is not yet built.
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
    /// Maximum order (clamped to the family cap).
    max_order: usize,
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
}

impl LsodeState {
    /// Create a fresh LSODE state from `config` (validated here).
    pub fn new(config: LsodeConfig) -> Self {
        config.check();
        let max_order = config.effective_max_order();
        let (method_coeffs, test_coeffs) =
            coeffs::calculate_integration_coefficients(config.method);
        Self {
            config,
            method_coeffs,
            test_coeffs,
            el: [0.0; 13],
            nordsieck: Nordsieck::new(N_ODES, max_order),
            order: 1,
            num_cols: 2,
            max_order,
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
/// state currently stored in Nordsieck column 0, writing it into `save`.
/// `accel_fn` supplies the translational acceleration; `frac` is the
/// fraction of the cycle for time-dependent gravity (ephemeris).
fn eval_derivative(
    nordsieck: &Nordsieck,
    accel_fn: &impl Fn(&TranslationalState, f64) -> DVec3,
    frac: f64,
    save: &mut [f64; N_ODES],
) {
    let pos = DVec3::new(
        nordsieck.history[0][0],
        nordsieck.history[1][0],
        nordsieck.history[2][0],
    );
    let vel = DVec3::new(
        nordsieck.history[3][0],
        nordsieck.history[4][0],
        nordsieck.history[5][0],
    );
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
        eval_derivative(&lsode.nordsieck, &accel_fn, 0.0, &mut save);
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
        assert!(
            ewt[i] > 0.0,
            "LSODE: error weight {i} fell to <= 0 ({}). atol must be > 0.",
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
    let mut save = [0.0_f64; N_ODES];
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

        // ── Functional-iteration corrector. ──
        for i in 0..N_ODES {
            accum[i] = 0.0;
        }
        let mut prev_iter_delta = 0.0_f64;
        let mut converged = false;
        let mut corrector_failed = false;
        for iter in 0..lsode.config.max_correction_iters {
            // y currently sits in Nordsieck column 0; evaluate derivative.
            eval_derivative(&lsode.nordsieck, accel_fn, frac, &mut save);
            // residual: save = h·f − h·y'_pred ; increment = save − accum.
            let mut incr = [0.0_f64; N_ODES];
            for i in 0..N_ODES {
                save[i] = lsode.step_size * save[i] - lsode.nordsieck.history[i][1];
                incr[i] = save[i] - accum[i];
            }
            let iter_delta = weighted_rms_norm(&incr, &lsode.error_weight);
            for i in 0..N_ODES {
                lsode.nordsieck.history[i][0] = lsode.nordsieck.history[i][0] + el0 * incr[i];
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
            let ratio = error_test_step_ratio(lsode, dsm, step_error);
            apply_step_ratio(lsode, ratio, max_step_size_inv);
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
        // `max_order` is the spare-column index (`max_history_size`); the
        // stash of `accum` for the order-increase indicator happens ONLY on
        // the step where the countdown hits 1 (the step *before* selection),
        // so that when the countdown hits 0 the next step reads last step's
        // accum — not this step's (which would zero the r_inc difference).
        lsode.order_select_para -= 1;
        if lsode.order_select_para == 0 {
            select_new_order(lsode, &accum, dsm, step_error, max_step_size_inv);
        } else if lsode.order_select_para == 1 && lsode.num_cols != lsode.max_order + 1 {
            for i in 0..N_ODES {
                lsode.nordsieck.history[i][lsode.max_order] = accum[i];
            }
        }
        lsode.prev_step_size = lsode.step_size;
        return;
    }
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

/// Step ratio after an error-test failure (order maintained): the
/// `step_ratio_order_same` formula, capped after repeated failures.
#[allow(
    clippy::cast_precision_loss,
    reason = "column count ≤ 13 is exactly representable in f64"
)]
fn error_test_step_ratio(lsode: &LsodeState, dsm: f64, step_error: i32) -> f64 {
    let exsm = 1.0 / lsode.num_cols as f64;
    let mut ratio = 1.0 / (1.2 * dsm.powf(exsm) + 0.0000012);
    if step_error <= -2 {
        ratio = ratio.min(0.2);
    }
    ratio
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
    // Recompute the first derivative at the current state.
    let mut save = [0.0_f64; N_ODES];
    eval_derivative(&lsode.nordsieck, accel_fn, 0.0, &mut save);
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
    max_step_size_inv: f64,
) {
    // r_inc requires the stashed accum from the previous step.
    let mut r_inc = 0.0;
    if lsode.num_cols != lsode.max_order + 1 {
        let diff: [f64; N_ODES] =
            std::array::from_fn(|i| accum[i] - lsode.nordsieck.history[i][lsode.max_order]);
        let dup =
            weighted_rms_norm(&diff, &lsode.error_weight) / lsode.test_coeffs[2][lsode.order - 1];
        let exup = 1.0 / (lsode.num_cols as f64 + 1.0);
        r_inc = 1.0 / (1.4 * dup.powf(exup) + 0.0000014);
    }
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
            e_err < 1e-9,
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
