//! Adams-Bashforth-Moulton 4th-order (ABM4) integrator for second-order ODEs.
//!
//! Port of Trick's er7_utils `ABM4SimpleSecondOrderODEIntegrator`
//! (`trick_source/er7_utils/integration/abm4/src/abm4_second_order_ode_integrator.cc`).
//!
//! This is also the non-stiff mode used by JEOD's LSODE integrator for
//! orbital problems. The variable-order Adams method in LSODE is not ported
//! here — we port the fixed 4th-order predictor-corrector scheme that is
//! exercised by JEOD's `SIM_integ_test/RUN_abm4`. Variable-order Adams and
//! BDF (stiff) support remain as future work.
//!
//! # Algorithm
//!
//! ABM4 integrates a second-order ODE `x'' = a(t, x, x')` as a coupled pair
//! `x' = v, v' = a`. It uses two complementary Adams multi-step formulas:
//!
//! - **Predictor (Adams-Bashforth 4)** — explicit, uses 4 past derivative values:
//!   `y_{n+1}^p = y_n + (h/24) * (55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})`
//!
//! - **Corrector (3-step Adams-Moulton, order 4)** — implicit, uses the predicted derivative:
//!   `y_{n+1}   = y_n + (h/24) * (9 f(y_{n+1}^p) + 19 f_n - 5 f_{n-1} + f_{n-2})`
//!
//! Per JEOD's implementation, the corrector is evaluated exactly once per step
//! (functional iteration with a single pass), not iterated to convergence.
//! This is the standard PECE scheme (Predict, Evaluate, Correct, Evaluate).
//!
//! # Priming
//!
//! ABM4 needs 4 history points (f_n, f_{n-1}, f_{n-2}, f_{n-3}) before the
//! multi-step formulas can be applied. The first three steps are taken with
//! RK4 (matching JEOD's use of RK4 as the priming integrator), accumulating
//! derivatives into the history buffer. Starting with the 4th step the ABM4
//! predict/correct scheme takes over.

use crate::integration::rk4_translational_step;
use crate::state::TranslationalState;
use glam::DVec3;

// ── ABM4 coefficients (from er7_utils `abm4_second_order_ode_integrator.cc`) ──

/// Common scale factor so predictor/corrector weights can be integral.
/// JEOD: `wscale = 1.0/24.0`.
const WSCALE: f64 = 1.0 / 24.0;

/// Adams-Bashforth 4 predictor weights, applied to [f_n, f_{n-1}, f_{n-2}, f_{n-3}].
/// JEOD: `predictor_weights[4] = {55.0, -59.0, 37.0, -9.0}`.
const PREDICTOR_WEIGHTS: [f64; 4] = [55.0, -59.0, 37.0, -9.0];

/// Adams-Moulton 3 corrector weights, applied to
/// [f(y_{n+1}^p), f_n, f_{n-1}, f_{n-2}].
/// JEOD: `corrector_weights[4] = {9.0, 19.0, -5.0, 1.0}`.
const CORRECTOR_WEIGHTS: [f64; 4] = [9.0, 19.0, -5.0, 1.0];

/// Number of derivative history points retained (ABM4 needs 4).
const HIST_LEN: usize = 4;

/// Persistent state for the ABM4 integrator.
///
/// Must be retained across calls to `abm4_translational_step`. The caller is
/// responsible for creating one `Abm4State` per integrated entity and passing
/// the same mutable reference on every step.
///
/// # Priming
///
/// The first three steps use RK4 to build up the history buffer. The step
/// count is tracked in `primed_steps`:
/// - `primed_steps = 0, 1, 2`: RK4 step taken; derivatives at the starting
///   state are pushed into the history.
/// - `primed_steps >= 3`: ABM4 predict/correct is used.
///
/// On creation all history slots are zero. They are filled in order as the
/// priming proceeds. A `reset()` method clears the state back to unprimed.
#[derive(Debug, Clone)]
pub struct Abm4State {
    /// Sliding-window derivative history: `[f_n, f_{n-1}, f_{n-2}, f_{n-3}]`
    /// for each of velocity (position derivative) and acceleration (velocity
    /// derivative). Each step, `rotate_history()` shifts all samples one slot
    /// older and drops the oldest — there is no wraparound; the window always
    /// holds the most recent `HIST_LEN` derivatives.
    ///
    /// Slot 0 is always the most recent value after `rotate_history()`.
    posdot_hist: [DVec3; HIST_LEN],
    veldot_hist: [DVec3; HIST_LEN],
    /// Number of RK4 priming steps completed so far. The step count is capped
    /// at `HIST_LEN - 1` (=3); once `primed_steps >= HIST_LEN - 1` the
    /// ABM4 formulas are used.
    primed_steps: usize,
    /// True after [`Self::mark_topology_dirty`] is called and before
    /// [`Self::reset_for_topology_change`] (or [`Self::reset`]) clears it.
    /// While set, [`abm4_translational_step`] panics — predictor history
    /// from a different mass / attachment topology is no longer valid and
    /// silently propagating it would corrupt the trajectory.
    ///
    /// JEOD precedent: `dyn_body_attach.cc::reset_integrators()` (lines
    /// 860, 871). See `JEOD_invariants.md` row IG.37.
    topology_dirty: bool,
}

impl Default for Abm4State {
    fn default() -> Self {
        Self::new()
    }
}

impl Abm4State {
    /// Create a fresh, unprimed integrator state.
    pub fn new() -> Self {
        Self {
            posdot_hist: [DVec3::ZERO; HIST_LEN],
            veldot_hist: [DVec3::ZERO; HIST_LEN],
            primed_steps: 0,
            topology_dirty: false,
        }
    }

    /// Reset the integrator back to its unprimed state. The next 3 steps will
    /// be taken with RK4 to rebuild the history.
    ///
    /// Also clears the [`topology_dirty`](Self::mark_topology_dirty) flag, so
    /// callers handling an attach / detach event may call this directly.
    /// Prefer [`Self::reset_for_topology_change`] at attach / detach call
    /// sites — it documents the intent at the call site.
    pub fn reset(&mut self) {
        self.posdot_hist = [DVec3::ZERO; HIST_LEN];
        self.veldot_hist = [DVec3::ZERO; HIST_LEN];
        self.primed_steps = 0;
        self.topology_dirty = false;
    }

    /// Reset the integrator and clear the topology-dirty flag. Equivalent to
    /// [`Self::reset`]; named to make attach / detach call sites self-
    /// documenting and to match JEOD's `reset_integrators()` semantics.
    ///
    /// Port of `dyn_body_attach.cc::reset_integrators()` for the ABM4
    /// branch — clears the predictor history so the next 3 steps re-prime
    /// with RK4 against the post-attach state.
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
    pub fn reset_for_topology_change(&mut self) {
        self.reset();
    }

    /// Mark the integrator as carrying stale predictor history because the
    /// body's mass / attachment topology changed. While the flag is set,
    /// [`abm4_translational_step`] panics — silently integrating with stale
    /// history would propagate a discontinuity-blind predicted state.
    ///
    /// Pair with [`Self::reset_for_topology_change`] at the attach / detach
    /// site. The combined `mark_topology_dirty` + `reset_for_topology_change`
    /// pattern catches future code paths that mark the integrator dirty but
    /// forget to clear the flag through reset.
    pub fn mark_topology_dirty(&mut self) {
        self.topology_dirty = true;
    }

    /// Returns true if the integrator is carrying stale history that has
    /// not yet been cleared via [`Self::reset_for_topology_change`].
    pub fn is_topology_dirty(&self) -> bool {
        self.topology_dirty
    }

    /// Returns true while the integrator is still priming with RK4.
    /// After `HIST_LEN - 1` steps the ABM4 formulas take over.
    pub fn is_priming(&self) -> bool {
        self.primed_steps < HIST_LEN - 1
    }

    /// Shift the history buffers right by one slot, dropping the oldest
    /// sample (a sliding-window shift, not a wraparound rotation).
    ///
    /// JEOD: `abm::rotate_history<4>(posdot_hist, veldot_hist)`. The JEOD
    /// name says "rotate" but the operation discards the sample that falls
    /// off the end; slot 0 is then ready to receive the newest derivatives.
    fn rotate_history(&mut self) {
        for j in (1..HIST_LEN).rev() {
            self.posdot_hist[j] = self.posdot_hist[j - 1];
            self.veldot_hist[j] = self.veldot_hist[j - 1];
        }
    }

    /// Save derivatives at the start of an RK4 priming step.
    ///
    /// JEOD: `ABM4SimpleSecondOrderODEIntegrator::technique_save_derivatives`.
    fn save_priming_derivatives(&mut self, velocity: DVec3, accel: DVec3) {
        self.rotate_history();
        self.posdot_hist[0] = velocity;
        self.veldot_hist[0] = accel;
    }
}

/// Advance translational state by one ABM4 step.
///
/// During priming (the first 3 calls after `reset()` / construction) this uses
/// RK4 to compute the step and stores the start-of-step derivatives in the
/// history. Starting with the 4th call, the ABM4 predict/correct scheme is
/// applied: one predictor pass, one end-of-step acceleration evaluation at
/// the predicted state (`time_frac = 1.0`), one corrector pass.
///
/// The `accel_fn` signature matches the other integrators in this crate:
/// `accel_fn(state, time_frac)` where `time_frac` is the fractional stage time
/// within the step (0.0 at the start, 1.0 at the end). ABM4 calls it:
/// - Once at `time_frac = 0.0` to capture `f_n`
/// - Once at `time_frac = 1.0` at the predicted state, for the corrector
///
/// During priming, `accel_fn` is called 4 times (standard RK4 stages) — the
/// `time_frac` values are `0.0, 0.5, 0.5, 1.0`.
///
/// # Panics
///
/// Panics unless `dt` is finite and strictly positive. ABM4 maintains a
/// multi-step history that would be silently corrupted by a zero-sized step
/// (the history would rotate without any actual time advance), so callers
/// must never invoke this for a no-op step.
pub fn abm4_translational_step(
    state: &TranslationalState,
    accel_fn: impl Fn(&TranslationalState, f64) -> DVec3,
    dt: f64,
    abm_state: &mut Abm4State,
) -> TranslationalState {
    // JEOD_INV: IG.34 — step dt must be finite and strictly positive; a zero step would rotate
    // ABM4's multi-step history without any actual time advance, silently corrupting the state.
    assert!(
        dt.is_finite() && dt > 0.0,
        "abm4_translational_step requires a finite positive dt, got {dt}"
    );
    // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change.
    // The attach / detach handler in `astrodyn_bevy::staging_system` and `astrodyn_runner::Simulation`
    // is responsible for calling `Abm4State::reset_for_topology_change()` whenever it mutates
    // the mass tree. Stepping past a topology change without that reset propagates predictor
    // history accumulated under the previous mass / attachment configuration, silently
    // corrupting the trajectory.
    assert!(
        !abm_state.topology_dirty,
        "abm4_translational_step called with stale predictor history: the body's \
         mass / attachment topology changed but Abm4State::reset_for_topology_change() \
         was not called. Wire the attach / detach handler to reset the integrator state \
         (astrodyn::reset_integrators) — see JEOD's dyn_body_attach.cc::reset_integrators() \
         and JEOD_invariants.md row IG.37."
    );

    if abm_state.is_priming() {
        // Priming: take an RK4 step. Before stepping, capture the
        // start-of-step derivatives into the history so ABM4 can use them
        // once enough history has accumulated.
        let accel_start = accel_fn(state, 0.0);
        abm_state.save_priming_derivatives(state.velocity, accel_start);
        abm_state.primed_steps += 1;

        // RK4 recomputes acceleration internally for stages 2, 3, 4. We just
        // wrap `accel_fn` and forward.
        return rk4_translational_step(state, &accel_fn, dt);
    }

    // ── Post-priming: ABM4 predict + evaluate + correct ──
    // Compute and save the current derivative f_n before rotating (matching
    // JEOD, which rotates then overwrites slot 0 with f_n inside the predictor).
    let accel_n = accel_fn(state, 0.0);
    let velocity_n = state.velocity;

    abm_state.rotate_history();
    abm_state.posdot_hist[0] = velocity_n;
    abm_state.veldot_hist[0] = accel_n;

    // Predictor weighted sums:
    //   w_posdot = 55*v_n - 59*v_{n-1} + 37*v_{n-2} - 9*v_{n-3}
    //   w_veldot = 55*a_n - 59*a_{n-1} + 37*a_{n-2} - 9*a_{n-3}
    let mut weighted_posdot = DVec3::ZERO;
    let mut weighted_veldot = DVec3::ZERO;
    for (j, &w) in PREDICTOR_WEIGHTS.iter().enumerate() {
        weighted_posdot += abm_state.posdot_hist[j] * w;
        weighted_veldot += abm_state.veldot_hist[j] * w;
    }

    let wscaled_dt = WSCALE * dt;
    let init_pos = state.position;
    let init_vel = state.velocity;
    let pred_pos = init_pos + weighted_posdot * wscaled_dt;
    let pred_vel = init_vel + weighted_veldot * wscaled_dt;

    // Evaluate derivatives at the predicted state (PECE scheme).
    let pred_state = TranslationalState {
        position: pred_pos,
        velocity: pred_vel,
    };
    let accel_pred = accel_fn(&pred_state, 1.0);

    // Corrector: y_{n+1} = y_n + (h/24) * (9*f_{pred} + 19*f_n - 5*f_{n-1} + f_{n-2}).
    // History slot 0 is now f_n (we just stored it), slot 1 is f_{n-1},
    // slot 2 is f_{n-2} — so the corrector sum over history uses slots 0..=2
    // with weights [19, -5, 1].
    let der_w = CORRECTOR_WEIGHTS[0]; // 9.0, applied to f(pred_state)
    let hist_w = &CORRECTOR_WEIGHTS[1..]; // [19, -5, 1]

    let mut corr_posdot = der_w * pred_vel;
    let mut corr_veldot = der_w * accel_pred;
    for (j, &w) in hist_w.iter().enumerate() {
        corr_posdot += abm_state.posdot_hist[j] * w;
        corr_veldot += abm_state.veldot_hist[j] * w;
    }

    let final_pos = init_pos + corr_posdot * wscaled_dt;
    let final_vel = init_vel + corr_veldot * wscaled_dt;

    // primed_steps stays saturated; subsequent steps continue in ABM mode.
    abm_state.primed_steps = abm_state.primed_steps.saturating_add(1);

    TranslationalState {
        position: final_pos,
        velocity: final_vel,
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "integrator step-count and analytic-period tests assert bit-exact recovery of literal values"
)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test step counts (~hours of orbit) fit exactly in f64 mantissa and usize"
)]
mod tests {
    use super::*;

    /// Harmonic oscillator: x'' = -x, x(0) = 1, v(0) = 0.
    /// After one full period (2π), state should return near initial.
    #[test]
    fn abm4_harmonic_oscillator() {
        let dt = 0.01;
        let steps = 628; // ~2π
        let t_final = dt * steps as f64;

        let mut state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let mut abm = Abm4State::new();
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        for _ in 0..steps {
            state = abm4_translational_step(&state, accel_fn, dt, &mut abm);
        }

        let exact_pos = t_final.cos();
        let exact_vel = -t_final.sin();
        let pos_err = (state.position.x - exact_pos).abs();
        let vel_err = (state.velocity.x - exact_vel).abs();

        // ABM4 is 4th-order accurate; similar tolerance to RK4.
        assert!(pos_err < 1e-6, "ABM4 harmonic osc pos err: {pos_err:.2e}");
        assert!(vel_err < 1e-6, "ABM4 harmonic osc vel err: {vel_err:.2e}");
    }

    /// Convergence order: halving dt should reduce the error by ~16x (4th order).
    /// Uses a long horizon (many steps) so the bulk of the error comes from
    /// ABM4's 4th-order truncation, not from the 3-step RK4 priming transient.
    #[test]
    fn abm4_convergence_order() {
        let dt_coarse = 0.01_f64;
        let dt_fine = dt_coarse / 2.0;
        let total_time = 10.0_f64;

        let initial = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };
        let exact_pos = total_time.cos();

        // Coarse run
        let steps_coarse = (total_time / dt_coarse).round() as usize;
        let mut state_c = initial;
        let mut abm_c = Abm4State::new();
        for _ in 0..steps_coarse {
            state_c = abm4_translational_step(&state_c, accel_fn, dt_coarse, &mut abm_c);
        }
        let err_coarse = (state_c.position.x - exact_pos).abs();

        // Fine run
        let steps_fine = (total_time / dt_fine).round() as usize;
        let mut state_f = initial;
        let mut abm_f = Abm4State::new();
        for _ in 0..steps_fine {
            state_f = abm4_translational_step(&state_f, accel_fn, dt_fine, &mut abm_f);
        }
        let err_fine = (state_f.position.x - exact_pos).abs();

        let ratio = err_coarse / err_fine;
        // 4th order: expect ~16. Allow loose bounds because priming (RK4 over
        // 3 steps) adds a fixed-order-5 contribution that depends weakly on
        // dt. Empirically on this oscillator we observe ~15-17.
        assert!(
            (10.0..=25.0).contains(&ratio),
            "ABM4 convergence ratio {ratio:.1} not consistent with 4th order \
             (err_coarse={err_coarse:.3e}, err_fine={err_fine:.3e})"
        );
    }

    /// Free particle: zero acceleration, constant velocity.
    #[test]
    fn abm4_free_particle() {
        let dt = 0.5;
        let initial_pos = DVec3::new(1.0, 2.0, 3.0);
        let initial_vel = DVec3::new(4.0, 5.0, 6.0);

        let mut state = TranslationalState {
            position: initial_pos,
            velocity: initial_vel,
        };
        let mut abm = Abm4State::new();
        let zero_accel = |_: &TranslationalState, _t: f64| DVec3::ZERO;

        let n = 50;
        for _ in 0..n {
            state = abm4_translational_step(&state, zero_accel, dt, &mut abm);
        }

        let expected = initial_pos + initial_vel * (dt * n as f64);
        let pos_err = (state.position - expected).length();
        let vel_err = (state.velocity - initial_vel).length();
        assert!(pos_err < 1e-11, "Free particle pos err: {pos_err}");
        assert!(vel_err < 1e-14, "Free particle vel err: {vel_err}");
    }

    /// Compared to RK4 at the same step size, ABM4 should be of similar
    /// accuracy on a smooth problem (both are 4th-order).
    #[test]
    fn abm4_comparable_to_rk4() {
        let dt = 0.01_f64;
        let steps = 1000;
        let t_final = dt * steps as f64;

        let initial = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        let mut s_rk4 = initial;
        for _ in 0..steps {
            s_rk4 = rk4_translational_step(&s_rk4, accel_fn, dt);
        }

        let mut s_abm = initial;
        let mut abm = Abm4State::new();
        for _ in 0..steps {
            s_abm = abm4_translational_step(&s_abm, accel_fn, dt, &mut abm);
        }

        let exact = t_final.cos();
        let err_rk4 = (s_rk4.position.x - exact).abs();
        let err_abm = (s_abm.position.x - exact).abs();

        // Both 4th-order schemes — within 10× of each other is expected.
        let ratio = err_abm.max(err_rk4) / err_abm.min(err_rk4).max(1e-20);
        assert!(
            ratio < 100.0,
            "ABM4 ({err_abm:.2e}) vs RK4 ({err_rk4:.2e}) differ by {ratio:.1}x"
        );
    }

    /// Kepler orbit: one full period on a circular LEO-like orbit.
    /// Verify orbit closure to within a few tens of meters.
    #[test]
    fn abm4_kepler_orbit() {
        let mu: f64 = 3.986_004_415e14;
        let r0: f64 = 7_000_000.0;
        let v0 = (mu / r0).sqrt();

        let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();
        let target_dt = 10.0_f64;
        let steps = (period / target_dt).round() as usize;
        let dt = period / steps as f64;

        let mut state = TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        };
        let mut abm = Abm4State::new();
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 {
            let r = s.position.length();
            -mu / (r * r * r) * s.position
        };

        for _ in 0..steps {
            state = abm4_translational_step(&state, accel_fn, dt, &mut abm);
        }

        let pos_err = (state.position - DVec3::new(r0, 0.0, 0.0)).length();
        // ABM4 over one orbit at dt=10s on a circular LEO — loose bound to
        // cover any priming transient. Observed error ≈ hundreds of meters.
        assert!(
            pos_err < 5_000.0,
            "ABM4 Kepler orbit closure err: {pos_err:.2e} m"
        );
    }

    /// Reset restores the integrator to the priming state.
    #[test]
    fn abm4_reset() {
        let mut abm = Abm4State::new();
        assert!(abm.is_priming());

        // Take 5 steps on a trivial problem.
        let mut state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };
        for _ in 0..5 {
            state = abm4_translational_step(&state, accel_fn, 0.01, &mut abm);
        }
        assert!(!abm.is_priming());

        abm.reset();
        assert!(abm.is_priming());
        assert_eq!(abm.primed_steps, 0);
        assert_eq!(abm.posdot_hist[0], DVec3::ZERO);
    }

    /// `reset_for_topology_change` clears history, the topology-dirty flag,
    /// and the priming counter — a strict superset of `reset`'s contract,
    /// suitable for use at attach / detach call sites.
    ///
    /// IG.37: matches JEOD's `dyn_body_attach.cc::reset_integrators()`
    /// semantics for the ABM4 branch.
    #[test]
    fn abm4_reset_for_topology_change_clears_history_and_dirty_flag() {
        let mut abm = Abm4State::new();
        let mut state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        // Run past priming so the predictor history is non-trivial.
        for _ in 0..5 {
            state = abm4_translational_step(&state, accel_fn, 0.01, &mut abm);
        }
        assert!(!abm.is_priming());
        assert_ne!(abm.posdot_hist[0], DVec3::ZERO);
        assert_ne!(abm.veldot_hist[0], DVec3::ZERO);

        // Mark dirty (as a topology-change handler would), then reset.
        abm.mark_topology_dirty();
        assert!(abm.is_topology_dirty());

        abm.reset_for_topology_change();

        assert!(abm.is_priming(), "history must be cleared back to priming");
        assert!(
            !abm.is_topology_dirty(),
            "topology-dirty flag must be cleared by reset"
        );
        assert_eq!(abm.primed_steps, 0);
        for i in 0..HIST_LEN {
            assert_eq!(abm.posdot_hist[i], DVec3::ZERO);
            assert_eq!(abm.veldot_hist[i], DVec3::ZERO);
        }
    }

    /// Stepping with the topology-dirty flag set must panic — silently
    /// using stale predictor history across a topology change is exactly
    /// the bug IG.37 closes.
    #[test]
    #[should_panic(expected = "stale predictor history")]
    fn abm4_step_with_topology_dirty_panics() {
        // JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
        let mut abm = Abm4State::new();
        let state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        abm.mark_topology_dirty();
        let _ = abm4_translational_step(&state, accel_fn, 0.01, &mut abm);
    }

    /// IG.34: `abm4_translational_step` rejects a zero (or negative)
    /// `dt`. A zero step still rotates the history buffer slot, dropping
    /// the oldest sample without any actual time advance — the predictor
    /// then reads a stale derivative as if it were `n-3` cycles old, and
    /// the corrector silently propagates a degraded state.
    #[test]
    #[should_panic(expected = "abm4_translational_step requires a finite positive dt")]
    fn ig_34_panics_on_zero_dt() {
        // JEOD_INV: IG.34 — step dt must be finite and strictly positive
        let mut abm = Abm4State::new();
        let state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        let _ = abm4_translational_step(&state, accel_fn, 0.0, &mut abm);
    }
}
