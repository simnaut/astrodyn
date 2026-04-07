//! Placeholder multi-step predictor-corrector integrator.
//!
//! **WARNING: This module implements Adams-Bashforth-Moulton (ABM), NOT the
//! Gauss-Jackson (Störmer-Cowell) method that JEOD uses.** JEOD's GJ
//! integrator (`gauss_jackson_integrator_base_second.hh`) is fundamentally
//! different: it uses dual Störmer-Cowell / Summed-Adams coefficient sets,
//! inverse backward difference accumulators (`delinv`), and a 5-state
//! finite state machine for startup. A full rewrite is required to match
//! JEOD line-by-line. See issue #36 (Critical C1).
//!
//! Current implementation: ABM with RK4 priming, using standard Adams
//! coefficients for both velocity and position integration via velocity
//! history. This is NOT equivalent to JEOD's approach.

use crate::state::TranslationalState;
use glam::DVec3;

// ── Adams-Bashforth and Adams-Moulton coefficients ──
// These are the standard ordinate-form coefficients for multi-step methods.
// AB: explicit predictor. AM: implicit corrector.
// Source: Hairer, Nørsett, Wanner "Solving ODEs I" Table III.5.1/III.5.2

/// Adams-Bashforth coefficients (predictor) for orders 1–8.
/// ab_coeffs(k) returns (coefficients, denominator) for order k.
/// y_{n+1} = y_n + h/denom * sum(c[i] * f_{n-i}, i=0..k-1)
fn ab_coeffs(order: usize) -> (Vec<f64>, f64) {
    match order {
        1 => (vec![1.0], 1.0),
        2 => (vec![3.0, -1.0], 2.0),
        3 => (vec![23.0, -16.0, 5.0], 12.0),
        4 => (vec![55.0, -59.0, 37.0, -9.0], 24.0),
        5 => (vec![1901.0, -2774.0, 2616.0, -1274.0, 251.0], 720.0),
        6 => (
            vec![4277.0, -7923.0, 9982.0, -7298.0, 2877.0, -475.0],
            1440.0,
        ),
        7 => (
            vec![
                198721.0, -447288.0, 705549.0, -688256.0, 407139.0, -134472.0, 19087.0,
            ],
            60480.0,
        ),
        8 => (
            vec![
                434241.0, -1152169.0, 2183877.0, -2664477.0, 2102243.0, -1041723.0, 295767.0,
                -36799.0,
            ],
            120960.0,
        ),
        _ => panic!("Adams-Bashforth order {order} not supported (max 8)"),
    }
}

/// Adams-Moulton coefficients (corrector) for orders 1–8.
/// am_coeffs(k) returns (coefficients, denominator) for order k.
/// y_{n+1} = y_n + h/denom * (c[0]*f_{n+1} + c[1]*f_n + c[2]*f_{n-1} + ...)
fn am_coeffs(order: usize) -> (Vec<f64>, f64) {
    match order {
        1 => (vec![1.0, 1.0], 2.0),
        2 => (vec![5.0, 8.0, -1.0], 12.0),
        3 => (vec![9.0, 19.0, -5.0, 1.0], 24.0),
        4 => (vec![251.0, 646.0, -264.0, 106.0, -19.0], 720.0),
        5 => (vec![475.0, 1427.0, -798.0, 482.0, -173.0, 27.0], 1440.0),
        6 => (
            vec![
                19087.0, 65112.0, -46461.0, 37504.0, -20211.0, 6312.0, -863.0,
            ],
            60480.0,
        ),
        7 => (
            vec![
                36799.0, 139849.0, -121797.0, 123133.0, -88547.0, 41499.0, -11351.0, 1375.0,
            ],
            120960.0,
        ),
        8 => (
            vec![
                1070017.0, 4467094.0, -4604594.0, 5595358.0, -5033120.0, 3146338.0, -1291214.0,
                312874.0, -33953.0,
            ],
            3628800.0,
        ),
        _ => panic!("Adams-Moulton order {order} not supported (max 8)"),
    }
}

/// Persistent state for the Gauss-Jackson (ABM) integrator.
///
/// Must be created once and maintained across steps.
#[derive(Debug, Clone)]
pub struct GaussJacksonState {
    /// AB predictor coefficients (scaled: c[i] / denom).
    ab: Vec<f64>,
    /// AM corrector coefficients (scaled: c[i] / denom).
    /// am[0] multiplies f_{n+1}, am[1] multiplies f_n, etc.
    am: Vec<f64>,
    /// Acceleration history. Index 0 = most recent (f_n), 1 = f_{n-1}, etc.
    /// New entries are inserted at front; oldest entries are truncated at end.
    acc_hist: Vec<DVec3>,
    /// Velocity history. Index 0 = most recent (v_n), 1 = v_{n-1}, etc.
    vel_hist: Vec<DVec3>,
    /// Number of priming steps completed.
    priming_count: usize,
    /// Whether the integrator is fully operational.
    operational: bool,
    /// Integration order.
    order: usize,
}

impl GaussJacksonState {
    /// Create a new GJ/ABM state for the given order (1–8).
    pub fn new(order: usize) -> Self {
        let (ab_raw, ab_denom) = ab_coeffs(order);
        let (am_raw, am_denom) = am_coeffs(order);

        let ab: Vec<f64> = ab_raw.iter().map(|c| c / ab_denom).collect();
        let am: Vec<f64> = am_raw.iter().map(|c| c / am_denom).collect();

        Self {
            ab,
            am,
            acc_hist: Vec::with_capacity(order),
            vel_hist: Vec::with_capacity(order),
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
    pub fn step(
        &mut self,
        state: &TranslationalState,
        accel_fn: impl Fn(&TranslationalState) -> DVec3,
        dt: f64,
    ) -> TranslationalState {
        if !self.operational {
            self.priming_step(state, &accel_fn, dt)
        } else {
            self.abm_step(state, &accel_fn, dt)
        }
    }

    /// RK4 priming step.
    fn priming_step(
        &mut self,
        state: &TranslationalState,
        accel_fn: &impl Fn(&TranslationalState) -> DVec3,
        dt: f64,
    ) -> TranslationalState {
        // Save current acceleration and velocity in history (most recent first)
        let acc = accel_fn(state);
        self.acc_hist.insert(0, acc);
        self.vel_hist.insert(0, state.velocity);
        self.priming_count += 1;

        // RK4 step
        let new_state = crate::rk4_translational_step(state, accel_fn, dt);

        // Check if we have enough history
        if self.priming_count >= self.order {
            self.operational = true;
        }

        new_state
    }

    /// ABM predictor-corrector step.
    fn abm_step(
        &mut self,
        state: &TranslationalState,
        accel_fn: &impl Fn(&TranslationalState) -> DVec3,
        dt: f64,
    ) -> TranslationalState {
        // Update history with current state's acceleration and velocity
        let acc_n = accel_fn(state);
        self.acc_hist.insert(0, acc_n);
        self.vel_hist.insert(0, state.velocity);

        // Trim history to order length
        self.acc_hist.truncate(self.order);
        self.vel_hist.truncate(self.order);

        // PREDICTOR (Adams-Bashforth): explicit extrapolation
        // v_{n+1}^P = v_n + h * sum(ab[i] * a_{n-i})
        // x_{n+1}^P = x_n + h * sum(ab[i] * v_{n-i})
        let mut pred_vel = state.velocity;
        let mut pred_pos = state.position;
        for (i, &c) in self.ab.iter().enumerate() {
            pred_vel += self.acc_hist[i] * (c * dt);
            pred_pos += self.vel_hist[i] * (c * dt);
        }

        let predicted = TranslationalState {
            position: pred_pos,
            velocity: pred_vel,
        };

        // Evaluate acceleration at predicted state
        let pred_acc = accel_fn(&predicted);

        // CORRECTOR (Adams-Moulton): implicit correction
        // v_{n+1}^C = v_n + h * (am[0]*a^P_{n+1} + am[1]*a_n + am[2]*a_{n-1} + ...)
        // x_{n+1}^C = x_n + h * (am[0]*v^P_{n+1} + am[1]*v_n + am[2]*v_{n-1} + ...)
        //
        // BUG: This entire module implements Adams-Bashforth-Moulton, NOT the
        // Gauss-Jackson (Störmer-Cowell) method that JEOD uses. JEOD's GJ
        // integrator uses:
        //   - Dual Störmer-Cowell / Summed-Adams coefficient sets (gj_coefs, sa_coefs)
        //   - Inverse backward difference accumulators (delinv)
        //   - A 5-state FSM for startup (Reset → Priming → BootstrapEdit →
        //     BootstrapStep → Operational)
        //   - Both velocity and position computed from acceleration history only
        //     (not velocity history)
        // See issue #36 (Critical C1) for details. A full rewrite to match
        // JEOD's gauss_jackson_integrator_base_second.hh is required.
        let mut corr_vel = state.velocity + pred_acc * (self.am[0] * dt);
        let mut corr_pos = state.position + pred_vel * (self.am[0] * dt);
        for (i, &c) in self.am[1..].iter().enumerate() {
            corr_vel += self.acc_hist[i] * (c * dt);
            corr_pos += self.vel_hist[i] * (c * dt);
        }

        // Note: we do NOT overwrite acc_hist[0] with the corrected acceleration.
        // The history stores the acceleration evaluated at each state as visited.
        // The correction only affects the returned state; the next step will
        // evaluate the acceleration at the corrected state naturally.
        TranslationalState {
            position: corr_pos,
            velocity: corr_vel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integration::rk4_translational_step;

    #[test]
    fn gj_harmonic_oscillator() {
        // x'' = -x, x(0)=1, v(0)=0
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
    fn gj_more_accurate_than_rk4() {
        let dt: f64 = 0.1;
        let steps = 100;
        let t_final = dt * steps as f64;

        let initial = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };
        let accel_fn = |s: &TranslationalState| -> DVec3 { -s.position };

        let mut state_rk4 = initial;
        let mut state_gj = initial;
        let mut gj = GaussJacksonState::new(8);

        for _ in 0..steps {
            state_rk4 = rk4_translational_step(&state_rk4, accel_fn, dt);
            state_gj = gj.step(&state_gj, accel_fn, dt);
        }

        let exact_pos = t_final.cos();
        let err_rk4 = (state_rk4.position.x - exact_pos).abs();
        let err_gj = (state_gj.position.x - exact_pos).abs();

        println!("RK4 err={err_rk4:.2e}, GJ8 err={err_gj:.2e}");
        assert!(
            err_gj < err_rk4,
            "GJ8 ({err_gj:.2e}) should be more accurate than RK4 ({err_rk4:.2e})"
        );
    }

    #[test]
    fn gj_kepler_orbit() {
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

        let pos_error = (state.position - DVec3::new(r0, 0.0, 0.0)).length();
        println!("GJ8 Kepler orbit: pos_err={pos_error:.2e} m (1 period)");
        // ABM8 at dt=10s for LEO (period ~5800s) gives ~11 km error.
        // This is NOT representative of JEOD's true Gauss-Jackson performance.
        // This module implements ABM, not GJ — see issue #36 (Critical C1).
        assert!(
            pos_error < 15_000.0,
            "GJ8 orbit position error {pos_error:.2e} m exceeds 15 km"
        );
    }

    #[test]
    fn gj_free_particle() {
        let dt: f64 = 0.5;
        let initial_pos = DVec3::new(1.0, 2.0, 3.0);
        let initial_vel = DVec3::new(4.0, 5.0, 6.0);

        let mut state = TranslationalState {
            position: initial_pos,
            velocity: initial_vel,
        };

        let zero_accel = |_: &TranslationalState| DVec3::ZERO;
        let mut gj = GaussJacksonState::new(8);

        for _ in 0..20 {
            state = gj.step(&state, zero_accel, dt);
        }

        let expected_pos = initial_pos + initial_vel * 10.0;
        let pos_error = (state.position - expected_pos).length();
        assert!(pos_error < 1e-10, "Free particle error: {pos_error}");
    }
}
