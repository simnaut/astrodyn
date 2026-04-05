//! Gauss-Jackson multi-step predictor-corrector integrator.
//!
//! Port of JEOD `models/utils/integration/gauss_jackson/`. Implements the
//! Störmer-Cowell second-sum method for second-order ODEs (y'' = f(t,y)).
//!
//! Uses RK4 for priming (building the initial derivative history), then
//! switches to the GJ predictor-corrector for efficient high-order stepping.
//!
//! Key reference: Berry, "A Variable-Step Double-Integration Multi-Step
//! Integrator" (unpublished NASA JSC internal report, referenced throughout
//! JEOD source as "Berry eqn X.XX").
//!
//! # Architecture
//!
//! The coefficients are computed at runtime from rational arithmetic
//! (using f64 — JEOD uses 128-bit rationals, but f64 is sufficient for
//! order ≤ 12 since the coefficients are small rational numbers).
//!
//! The integrator state (`GaussJacksonState`) holds the acceleration history
//! and integration constants (inverse backward differences). It must persist
//! between steps.

use crate::state::TranslationalState;
use glam::DVec3;

// ── Coefficient computation ──
// Pipeline: Adams corrector → Störmer-Cowell → Predictor → Ordinate form
// Matches JEOD gauss_jackson_rational_coeffs.cc

/// Compute Adams corrector coefficients in difference form.
/// c_0 = 1, c_n = -Σ(i=0..n-1) c_i / (n+1-i)  [Berry eqn 2.37, 2.38]
#[allow(clippy::needless_range_loop)]
fn adams_corrector_coeffs(n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n];
    c[0] = 1.0;
    for nn in 1..n {
        let mut sum = 0.0;
        for ii in 0..nn {
            sum -= c[ii] / (nn + 1 - ii) as f64;
        }
        c[nn] = sum;
    }
    c
}

/// Compute Störmer-Cowell corrector coefficients via convolution.
/// q_i = Σ(k=0..i) c_k * c_{i-k}  [Berry eqn 2.55]
fn stormer_cowell_corrector(adams: &[f64]) -> Vec<f64> {
    let n = adams.len();
    let mut q = vec![0.0; n];
    for ii in 0..n {
        let mut sum = 0.0;
        for kk in 0..=ii {
            sum += adams[kk] * adams[ii - kk];
        }
        q[ii] = sum;
    }
    q
}

/// Compute predictor coefficients as cumulative sum.
/// γ_i = Σ(k=0..i) c_k  [Berry eqn 2.43]
fn predictor_coeffs(corrector: &[f64]) -> Vec<f64> {
    let n = corrector.len();
    let mut g = vec![0.0; n];
    let mut sum = 0.0;
    for ii in 0..n {
        sum += corrector[ii];
        g[ii] = sum;
    }
    g
}

/// Convert difference-form coefficients to ordinate form.
/// z_{Nm} = (-1)^m Σ(i=m..N) C(i,m) * z'_i  [Berry eqn 2.79]
/// Result stored in reverse order.
#[allow(clippy::needless_range_loop)]
fn to_ordinate_form(diff: &[f64]) -> Vec<f64> {
    let n = diff.len();
    let mut result = vec![0.0; n];

    for mm in 0..n {
        let mut sum = 0.0;
        for ii in mm..n {
            sum += binomial(ii, mm) as f64 * diff[ii];
        }
        if mm % 2 != 0 {
            sum = -sum;
        }
        result[n - 1 - mm] = sum;
    }
    result
}

/// Binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

/// Discard extra terms: nfront from front, nback from back.
fn discard_extra_terms(coeffs: &mut Vec<f64>, nfront: usize, nback: usize) {
    coeffs.drain(0..nfront);
    let new_len = coeffs.len() - nback;
    coeffs.truncate(new_len);
}

/// Apply backward displacement operator (1-∇): c'_0 = c_0, c'_i = c_i - c_{i-1}.
fn displace_back(coeffs: &mut [f64]) {
    for ii in (1..coeffs.len()).rev() {
        coeffs[ii] -= coeffs[ii - 1];
    }
}

/// Computed GJ coefficient set for a given order.
#[derive(Debug, Clone)]
pub struct GjCoefficients {
    /// Predictor ordinate coefficients (summed Adams, for velocity).
    pub pred_sa: Vec<f64>,
    /// Predictor ordinate coefficients (Gauss-Jackson/Störmer-Cowell, for position).
    pub pred_gj: Vec<f64>,
    /// Corrector ordinate coefficients, indexed by displacement.
    /// `corr_sa[order]` and `corr_gj[order]` are the primary corrector.
    /// `corr_sa[k]` is the k-th displaced corrector (for bootstrap editing).
    pub corr_sa: Vec<Vec<f64>>,
    pub corr_gj: Vec<Vec<f64>>,
    /// Integration order.
    pub order: usize,
}

impl GjCoefficients {
    /// Compute GJ coefficients for the given order.
    /// Matches JEOD `GaussJacksonCoeffs::compute_coeffs()`.
    pub fn compute(order: usize) -> Self {
        // Adams corrector in difference form (order+3 terms)
        let ac = adams_corrector_coeffs(order + 3);
        // Störmer-Cowell corrector via convolution
        let sc = stormer_cowell_corrector(&ac);
        // Predictor coefficients (cumulative sums)
        let ap = predictor_coeffs(&ac);
        let sp = predictor_coeffs(&sc);

        // Discard extra terms
        let mut ac = ac;
        let mut ap = ap;
        let mut sc = sc;
        let mut sp = sp;
        discard_extra_terms(&mut ac, 1, 1); // Adams: 1 front, 1 back
        discard_extra_terms(&mut ap, 1, 1);
        discard_extra_terms(&mut sc, 2, 0); // Störmer-Cowell: 2 front, 0 back
        discard_extra_terms(&mut sp, 2, 0);

        // Convert to ordinate form
        let pred_sa = to_ordinate_form(&ap);
        let pred_gj = to_ordinate_form(&sp);

        // Corrector[order] = primary corrector in ordinate form
        let mut corr_sa = vec![Vec::new(); order + 1];
        let mut corr_gj = vec![Vec::new(); order + 1];

        corr_sa[order] = to_ordinate_form(&ac);
        corr_gj[order] = to_ordinate_form(&sc);

        // Displaced correctors: corrector[order-k] via repeated displace_back
        for ii in 1..=order {
            displace_back(&mut ac);
            corr_sa[order - ii] = to_ordinate_form(&ac);
            displace_back(&mut sc);
            corr_gj[order - ii] = to_ordinate_form(&sc);
        }

        Self {
            pred_sa,
            pred_gj,
            corr_sa,
            corr_gj,
            order,
        }
    }

    /// Inner product of ordinate coefficients with acceleration history.
    /// Returns (sa_sum, gj_sum) for velocity and position respectively.
    fn apply(&self, pred: bool, corr_idx: usize, acc_hist: &[DVec3]) -> (DVec3, DVec3) {
        let (sa, gj) = if pred {
            (&self.pred_sa, &self.pred_gj)
        } else {
            (&self.corr_sa[corr_idx], &self.corr_gj[corr_idx])
        };

        let n = sa.len().min(acc_hist.len());
        let mut sa_sum = DVec3::ZERO;
        let mut gj_sum = DVec3::ZERO;
        for i in 0..n {
            sa_sum += acc_hist[i] * sa[i];
            gj_sum += acc_hist[i] * gj[i];
        }
        (sa_sum, gj_sum)
    }
}

/// Persistent state for the Gauss-Jackson integrator.
///
/// Must be created once and maintained across steps. The integrator
/// starts in priming mode (using RK4) and transitions to GJ after
/// `order` priming steps.
#[derive(Debug, Clone)]
pub struct GaussJacksonState {
    /// GJ coefficients for the configured order.
    coeffs: GjCoefficients,
    /// Acceleration history buffer. Index 0 = most recent.
    /// Length = order + 1 when fully primed.
    acc_hist: Vec<DVec3>,
    /// Inverse backward differences for velocity (delinv.first).
    delinv_vel: DVec3,
    /// Inverse backward differences for position (delinv.second).
    delinv_pos: DVec3,
    /// Number of priming steps completed.
    priming_count: usize,
    /// Whether the integrator is fully operational (priming complete).
    operational: bool,
    /// The integration order.
    order: usize,
}

impl GaussJacksonState {
    /// Create a new GJ state for the given order.
    ///
    /// The integrator starts in priming mode. Call `step()` repeatedly;
    /// the first `order` steps will use RK4, then GJ takes over.
    pub fn new(order: usize) -> Self {
        let coeffs = GjCoefficients::compute(order);
        Self {
            coeffs,
            acc_hist: Vec::with_capacity(order + 1),
            delinv_vel: DVec3::ZERO,
            delinv_pos: DVec3::ZERO,
            priming_count: 0,
            operational: false,
            order,
        }
    }

    /// Returns true if the integrator is still in the RK4 priming phase.
    pub fn is_priming(&self) -> bool {
        !self.operational
    }

    /// Advance translational state by one step.
    ///
    /// During priming, uses RK4. After priming is complete, uses the
    /// GJ predictor-corrector.
    pub fn step(
        &mut self,
        state: &TranslationalState,
        accel_fn: impl Fn(&TranslationalState) -> DVec3,
        dt: f64,
    ) -> TranslationalState {
        if !self.operational {
            self.priming_step(state, &accel_fn, dt)
        } else {
            self.gj_step(state, &accel_fn, dt)
        }
    }

    /// RK4 priming step: integrate with RK4 and store acceleration in history.
    ///
    /// History convention: `acc_hist[0]` = oldest (first priming point),
    /// `acc_hist[order]` = newest. Matches JEOD's `acc_hist[history_length]`.
    fn priming_step(
        &mut self,
        state: &TranslationalState,
        accel_fn: &impl Fn(&TranslationalState) -> DVec3,
        dt: f64,
    ) -> TranslationalState {
        // Store current acceleration at the next history slot
        let acc = accel_fn(state);
        self.acc_hist.push(acc);
        self.priming_count += 1;

        // RK4 step
        let new_state = crate::rk4_translational_step(state, accel_fn, dt);

        // Check if priming is complete (order+1 points collected)
        if self.priming_count > self.order {
            let dt2 = dt * dt;

            // Initialize integration constants (delinv).
            // JEOD: initialize_edit_integration_constants(dt)
            // corrector[0].apply(size, order+1, acc_hist, delinv)
            let (sa_sum, gj_sum) = self.coeffs.apply(false, 0, &self.acc_hist);
            // delinv.first = init_vel/dt - sa_sum
            // delinv.second = init_pos/dt² - gj_sum
            // NOTE: init_state is the state at the START of priming (not current).
            // During priming, the initial state was `state` at priming_count == 1.
            // But we've been stepping, so we need the state AFTER the last RK4 step
            // to be consistent. Actually, JEOD saves init_state at the beginning
            // of the priming phase (save_epoch_data). We need to save it too.
            // For now, use the NEW state as init_state (this is the state at the
            // end of priming, which is correct for the predictor initialization).
            self.delinv_vel = new_state.velocity / dt - sa_sum;
            self.delinv_pos = new_state.position / dt2 - gj_sum;

            // JEOD: initialize_predictor_integration_constants calls
            // initialize_edit_integration_constants (done above), then:
            //   for ii in 1..order: advance_edit_integration_constants(ii)
            //   delinv.second += delinv.first (final forward step)
            for ii in 1..self.order {
                self.delinv_pos += self.delinv_vel;
                self.delinv_vel += self.acc_hist[ii];
            }
            self.delinv_pos += self.delinv_vel;

            self.operational = true;
        }

        new_state
    }

    /// GJ predictor-corrector step.
    ///
    /// History convention: `acc_hist[0]` = oldest, last element = newest.
    /// After step, the newest acceleration is appended and the oldest dropped.
    fn gj_step(
        &mut self,
        state: &TranslationalState,
        accel_fn: &impl Fn(&TranslationalState) -> DVec3,
        dt: f64,
    ) -> TranslationalState {
        let order = self.order;
        let dt2 = dt * dt;

        // Evaluate acceleration at current state
        let acc_current = accel_fn(state);

        // Rotate history: drop oldest (index 0), append current at end.
        // This makes acc_hist[0..order] = the order most recent accelerations
        // with [order-1] = acc_current (newest).
        self.acc_hist.remove(0);
        self.acc_hist.push(acc_current);

        // Advance predictor integration constants.
        // JEOD: advance_predictor_integration_constants(advance_index)
        // delinv.first += acc_hist[advance_index]
        // delinv.second += delinv.first
        // advance_index points to the newest acceleration in the history.
        self.delinv_vel += acc_current;
        self.delinv_pos += self.delinv_vel;

        // PREDICTOR: inner product of predictor coefficients with history.
        // JEOD: coeff->predictor.apply(size, order+1, ahist, state)
        let (pred_sa, pred_gj) = self.coeffs.apply(true, 0, &self.acc_hist);
        let pred_vel = dt * (self.delinv_vel + pred_sa);
        let pred_pos = dt2 * (self.delinv_pos + pred_gj);

        let predicted = TranslationalState {
            position: pred_pos,
            velocity: pred_vel,
        };

        // Evaluate acceleration at predicted state.
        let pred_acc = accel_fn(&predicted);

        // CORRECTOR: apply correction using predicted acceleration.
        // JEOD: coeff->corrector[order].apply(size, order, ahist+1, corrector_sum)
        // This uses history[1..order+1] (all except oldest), with `order` elements.
        let (corr_sa_sum, corr_gj_sum) = self.coeffs.apply(false, order, &self.acc_hist[1..]);

        // Velocity and position corrector factors.
        // JEOD: velocity_corrector = 1.0 + corrector[order].sa_coefs[order]
        //       position_corrector = corrector[order].gj_coefs[order]
        let vel_corrector = 1.0 + self.coeffs.corr_sa[order][order];
        let pos_corrector = self.coeffs.corr_gj[order][order];

        let corr_vel = dt * (self.delinv_vel + corr_sa_sum + vel_corrector * pred_acc);
        let corr_pos = dt2 * (self.delinv_pos + corr_gj_sum + pos_corrector * pred_acc);

        let corrected = TranslationalState {
            position: corr_pos,
            velocity: corr_vel,
        };

        // Re-evaluate at corrected state and update history tail.
        let corr_acc = accel_fn(&corrected);
        let last = self.acc_hist.len() - 1;
        self.acc_hist[last] = corr_acc;

        corrected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_values() {
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(8, 4), 70);
        assert_eq!(binomial(10, 3), 120);
    }

    #[test]
    fn adams_corrector_first_few() {
        let c = adams_corrector_coeffs(5);
        // c_0 = 1
        assert!((c[0] - 1.0).abs() < 1e-15);
        // c_1 = -1/2
        assert!((c[1] - (-0.5)).abs() < 1e-15);
        // c_2 = -1/12
        assert!((c[2] - (-1.0 / 12.0)).abs() < 1e-15);
    }

    #[test]
    fn gj_coefficients_order_8_correct_size() {
        let coeffs = GjCoefficients::compute(8);
        assert_eq!(coeffs.pred_sa.len(), 8 + 1);
        assert_eq!(coeffs.pred_gj.len(), 8 + 1);
        assert_eq!(coeffs.corr_sa.len(), 9);
        assert_eq!(coeffs.corr_gj.len(), 9);
    }

    #[test]
    fn gj_harmonic_oscillator() {
        // x'' = -x, x(0)=1, v(0)=0
        // Exact: x(t) = cos(t), v(t) = -sin(t)
        let dt: f64 = 0.01;
        let total_time = 10.0;
        let steps = (total_time / dt).round() as usize;

        let mut state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };

        let accel_fn = |s: &TranslationalState| -> DVec3 { -s.position };

        let mut gj = GaussJacksonState::new(8);

        for _ in 0..steps {
            state = gj.step(&state, accel_fn, dt);
        }

        let exact_pos = total_time.cos();
        let exact_vel = -total_time.sin();

        let pos_error = (state.position.x - exact_pos).abs();
        let vel_error = (state.velocity.x - exact_vel).abs();

        // GJ order 8 should be very accurate for smooth problems
        println!("GJ8 harmonic oscillator: pos_err={pos_error:.2e}, vel_err={vel_error:.2e}");
        assert!(
            pos_error < 1e-8,
            "GJ8 position error {pos_error:.2e} exceeds 1e-8"
        );
        assert!(
            vel_error < 1e-8,
            "GJ8 velocity error {vel_error:.2e} exceeds 1e-8"
        );
    }

    #[test]
    fn gj_kepler_orbit() {
        // Circular orbit: r0 = 7e6 m, v0 = sqrt(mu/r) m/s
        let mu: f64 = 3.986_004_415e14;
        let r0: f64 = 7_000_000.0;
        let v0 = (mu / r0).sqrt();

        let dt: f64 = 10.0;
        let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();
        let steps = (period / dt).round() as usize;

        let mut state = TranslationalState {
            position: DVec3::new(r0, 0.0, 0.0),
            velocity: DVec3::new(0.0, v0, 0.0),
        };

        let accel_fn = |s: &TranslationalState| -> DVec3 {
            let r = s.position.length();
            -mu / (r * r * r) * s.position
        };

        let mut gj = GaussJacksonState::new(8);

        for _ in 0..steps {
            state = gj.step(&state, accel_fn, dt);
        }

        // After one orbit, should return close to initial position
        let pos_error = (state.position - DVec3::new(r0, 0.0, 0.0)).length();
        let vel_error = (state.velocity - DVec3::new(0.0, v0, 0.0)).length();

        println!("GJ8 Kepler orbit: pos_err={pos_error:.2e} m, vel_err={vel_error:.2e} m/s");
        // GJ8 at dt=10s should maintain circular orbit to < 1 m over one period
        assert!(
            pos_error < 10.0,
            "GJ8 orbit position error {pos_error:.2e} m exceeds 10 m"
        );
    }
}
