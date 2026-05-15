// JEOD_INV: TS.01 — the integrator stage kernels operate on the raw
// `SixDofState` storage shape (one anonymous vehicle per call), so the
// typed-seam lift to `AngularVelocity<BodyFrame<SelfRef>>` here uses
// the per-entity storage-boundary wildcard. The quaternion stays a
// raw `JeodQuat` by design: RK4/RKF45 substages produce intermediate
// quaternion accumulator values that are *not* unit-norm, so the
// witness-gated `BodyAttitude<V>` constructor would panic on
// legitimate mid-stage state — only ω is the kind-distinct frame
// invariant that benefits from compile-time checking here. See
// `docs/JEOD_invariants.md` row TS.01 and the lint at
// `tests/self_ref_self_planet_discipline.rs`.
//! Integration-method dispatch and the per-method translational /
//! rotational step kernels.
//!
//! Mirrors JEOD's
//! [`models/utils/integration/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/utils/integration/)
//! integration manager (v5.4.0). RK4, RKF45, Adams-Bashforth-Moulton 4
//! (ABM4), and Gauss-Jackson 8 are dispatched through the
//! `IntegrationMethod` enum below; multi-stage methods run as an inner
//! loop within a single `FixedUpdate` tick.

use crate::mass::MassProperties;
use crate::rotational::*;
use crate::state::TranslationalState;
use astrodyn_math::JeodQuat;
use astrodyn_quantities::aliases::AngularVelocity;
use astrodyn_quantities::frame::{BodyFrame, SelfRef};
use glam::DVec3;

/// Integration method selection.
///
/// Follows the existing enum-dispatch pattern (`GravityModel`). Each variant
/// dispatches to its own pure-function step implementation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IntegratorType {
    /// Classical 4th-order Runge-Kutta (fixed step).
    #[default]
    Rk4,
    /// Runge-Kutta-Fehlberg 4(5) (fixed step, 5th-order result).
    /// Uses 6 function evaluations per step vs RK4's 4, but achieves
    /// 5th-order accuracy. Step size is fixed (no adaptive control).
    Rkf45,
    /// Gauss-Jackson (Störmer-Cowell) multi-step predictor-corrector.
    ///
    /// Port of JEOD's Gauss-Jackson integrator using dual Störmer-Cowell /
    /// Summed-Adams coefficient sets, inverse backward difference accumulators
    /// (delinv), and a 5-state FSM for startup. Uses RK4 for priming, then
    /// Gauss-Jackson predict/correct for high-order stepping.
    ///
    /// Requires persistent `GaussJacksonState` across steps. The state must
    /// be stored externally (in `SimBody` or as a Bevy component) and passed
    /// to the integration function.
    ///
    /// **Forward-time only.** Multi-step methods cannot be run with a
    /// negative `sim_dt` or `time_scale_factor`; callers that need
    /// reverse-time integration must select [`IntegratorType::Rk4`] or
    /// [`IntegratorType::Rkf45`] instead. `GaussJacksonState::integrate`
    /// asserts both arguments are finite and non-negative on entry.
    GaussJackson(crate::gauss_jackson::config::GaussJacksonConfig),
    /// Adams-Bashforth-Moulton 4th-order (PECE scheme, fixed step).
    ///
    /// Port of Trick's `ABM4SimpleSecondOrderODEIntegrator`, the integrator
    /// exercised by JEOD's `SIM_integ_test/RUN_abm4`. Uses RK4 for the first
    /// 3 priming steps, then the classic AB4 predictor + AM3 corrector.
    ///
    /// This is also the non-stiff Adams mode used by JEOD's LSODE for orbital
    /// problems — at fixed order 4. The variable-order Adams method and BDF
    /// (stiff) support from LSODE are future work. For the verification sim
    /// `RUN_lsode` (LSODE with default ImplicitAdamsNonStiff) this fixed-order
    /// ABM4 is the best available approximation.
    ///
    /// Requires persistent `Abm4State` across steps, stored externally.
    /// Translational-only; 6-DOF is not yet supported.
    Abm4,
}

/// Advance translational state by one RK4 step.
///
/// The `accel_fn` computes acceleration from the current state. It is called
/// 4 times (once per RK4 stage) at intermediate positions, enabling correct
/// multi-stage integration even when forces depend on position (e.g., gravity).
///
/// `dt` may be negative for time-reversed integration (used by JEOD's
/// `SIM_7_time_reversal` cross-validation). It must be finite.
pub fn rk4_translational_step(
    state: &TranslationalState,
    accel_fn: impl Fn(&TranslationalState, f64) -> DVec3,
    dt: f64,
) -> TranslationalState {
    assert!(
        dt.is_finite(),
        "rk4_translational_step requires a finite dt, got {dt}"
    );
    // Stage 1: evaluate at current state
    let k1_a = accel_fn(state, 0.0);
    let k1_v = state.velocity;

    // Stage 2: evaluate at t + dt/2, using k1
    let s2 = TranslationalState {
        position: state.position + k1_v * (dt * 0.5),
        velocity: state.velocity + k1_a * (dt * 0.5),
    };
    let k2_a = accel_fn(&s2, 0.5);
    let k2_v = s2.velocity;

    // Stage 3: evaluate at t + dt/2, using k2
    let s3 = TranslationalState {
        position: state.position + k2_v * (dt * 0.5),
        velocity: state.velocity + k2_a * (dt * 0.5),
    };
    let k3_a = accel_fn(&s3, 0.5);
    let k3_v = s3.velocity;

    // Stage 4: evaluate at t + dt, using k3
    let s4 = TranslationalState {
        position: state.position + k3_v * dt,
        velocity: state.velocity + k3_a * dt,
    };
    let k4_a = accel_fn(&s4, 1.0);
    let k4_v = s4.velocity;

    // Combine: weighted average
    let sixth_dt = dt / 6.0;
    TranslationalState {
        position: state.position + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * sixth_dt,
        velocity: state.velocity + (k1_a + k2_a * 2.0 + k3_a * 2.0 + k4_a) * sixth_dt,
    }
}

/// Advance a 6-DOF state by one RK4 step, integrating 13 variables simultaneously:
/// `position[3]`, `velocity[3]`, `quaternion[4]`, `angular velocity[3]`.
///
/// The `accel_fn` computes translational acceleration from the current 6-DOF state.
/// The `torque_fn` computes body-frame external torque from the current 6-DOF state.
///
/// At each RK4 stage the rotational acceleration is computed via Euler's equation
/// (`compute_rotational_acceleration`) using the mass properties inertia tensor.
/// The quaternion derivative is computed via `compute_left_quat_deriv`.
///
/// After the final RK4 combination, the quaternion is renormalized using
/// `normalize_integ` (without forcing scalar non-negative).
///
/// # Quaternion drift across stages
///
/// The four `qdot` evaluations stack additively on `q0` with stage weights, then
/// `normalize_integ` runs once at the end. Intermediate stages 2 / 3 / 4 use the
/// un-normalized quaternion when evaluating Euler's equation and torques.
///
/// For the dt regime our Tier 3 suite exercises (60 s LEO, 300 s GEO, 100–200 s
/// translunar, with `|ω| ≲ 1 rad/s`), per-stage drift of `|q| − 1` stays below
/// `1e-4`. Post-step normalization absorbs that drift while preserving JEOD
/// parity. Callers integrating fast tumblers or with much larger dt should
/// renormalize between stages 2 and 3.
pub fn rk4_sixdof_step(
    state: &SixDofState,
    accel_fn: impl Fn(&SixDofState, f64) -> DVec3,
    torque_fn: impl Fn(&SixDofState) -> DVec3,
    mass_props: &MassProperties,
    dt: f64,
) -> SixDofState {
    assert!(
        dt.is_finite(),
        "rk4_sixdof_step requires a finite dt, got {dt}"
    );
    // Extract initial flat state: [pos(3), vel(3), quat(4), omega(3)]
    let pos0 = state.trans.position;
    let vel0 = state.trans.velocity;
    let q0 = state.rot.quaternion.data;
    let omega0 = state.rot.ang_vel_body;

    // Helper: reconstruct SixDofState from flat components
    let make_state = |pos: DVec3, vel: DVec3, q: [f64; 4], omega: DVec3| -> SixDofState {
        SixDofState {
            trans: TranslationalState {
                position: pos,
                velocity: vel,
            },
            rot: RotationalState {
                quaternion: JeodQuat::new(q[0], q[1], q[2], q[3]),
                ang_vel_body: omega,
            },
        }
    };

    // Helper: evaluate derivatives at a given state
    let eval_derivs = |s: &SixDofState, time_frac: f64| -> (DVec3, DVec3, [f64; 4], DVec3) {
        let k_v = s.trans.velocity;
        let k_a = accel_fn(s, time_frac);
        // Lift the body-rate into the typed seam so
        // `compute_left_quat_deriv_typed` /
        // `compute_rotational_acceleration_typed` reject an
        // inertial-frame ω at compile time. The quaternion stays raw —
        // intermediate RK4 stages are not unit-norm by design.
        // allowed: typed-integrator seam — lifts the per-stage `DVec3` body rate into the
        // typed kernel so `compute_left_quat_deriv_typed` rejects an inertial ω at compile time.
        let typed_omega = AngularVelocity::<BodyFrame<SelfRef>>::from_raw_si(s.rot.ang_vel_body);
        let k_qdot = compute_left_quat_deriv_typed::<SelfRef>(&s.rot.quaternion, typed_omega);
        let k_alpha = compute_rotational_acceleration_typed::<SelfRef>(
            &mass_props.inertia,
            &mass_props.inverse_inertia,
            typed_omega,
            torque_fn(s),
        );
        (k_v, k_a, k_qdot, k_alpha)
    };

    // Helper: step quaternion 4-vector
    let step_q = |q_base: [f64; 4], k_qdot: [f64; 4], h: f64| -> [f64; 4] {
        [
            q_base[0] + k_qdot[0] * h,
            q_base[1] + k_qdot[1] * h,
            q_base[2] + k_qdot[2] * h,
            q_base[3] + k_qdot[3] * h,
        ]
    };

    // Stage 1: evaluate at current state
    let (k1_v, k1_a, k1_qdot, k1_alpha) = eval_derivs(state, 0.0);

    // Stage 2: evaluate at t + dt/2, using k1
    let half_dt = dt * 0.5;
    let s2 = make_state(
        pos0 + k1_v * half_dt,
        vel0 + k1_a * half_dt,
        step_q(q0, k1_qdot, half_dt),
        omega0 + k1_alpha * half_dt,
    );
    let (k2_v, k2_a, k2_qdot, k2_alpha) = eval_derivs(&s2, 0.5);

    // Stage 3: evaluate at t + dt/2, using k2
    let s3 = make_state(
        pos0 + k2_v * half_dt,
        vel0 + k2_a * half_dt,
        step_q(q0, k2_qdot, half_dt),
        omega0 + k2_alpha * half_dt,
    );
    let (k3_v, k3_a, k3_qdot, k3_alpha) = eval_derivs(&s3, 0.5);

    // Stage 4: evaluate at t + dt, using k3
    let s4 = make_state(
        pos0 + k3_v * dt,
        vel0 + k3_a * dt,
        step_q(q0, k3_qdot, dt),
        omega0 + k3_alpha * dt,
    );
    let (k4_v, k4_a, k4_qdot, k4_alpha) = eval_derivs(&s4, 1.0);

    // Combine: weighted average (1/6)(k1 + 2*k2 + 2*k3 + k4) * dt
    let sixth_dt = dt / 6.0;

    let final_pos = pos0 + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * sixth_dt;
    let final_vel = vel0 + (k1_a + k2_a * 2.0 + k3_a * 2.0 + k4_a) * sixth_dt;
    let final_omega = omega0 + (k1_alpha + k2_alpha * 2.0 + k3_alpha * 2.0 + k4_alpha) * sixth_dt;

    let final_q = [
        q0[0] + (k1_qdot[0] + 2.0 * k2_qdot[0] + 2.0 * k3_qdot[0] + k4_qdot[0]) * sixth_dt,
        q0[1] + (k1_qdot[1] + 2.0 * k2_qdot[1] + 2.0 * k3_qdot[1] + k4_qdot[1]) * sixth_dt,
        q0[2] + (k1_qdot[2] + 2.0 * k2_qdot[2] + 2.0 * k3_qdot[2] + k4_qdot[2]) * sixth_dt,
        q0[3] + (k1_qdot[3] + 2.0 * k2_qdot[3] + 2.0 * k3_qdot[3] + k4_qdot[3]) * sixth_dt,
    ];

    // JEOD_INV: DB.09 — quaternion normalized after every integration step
    // JEOD_INV: RF.09 — quaternion assumed normalized for left_quat_to_transformation
    // Normalize the final quaternion (without forcing scalar non-negative)
    let mut final_quat = JeodQuat::new(final_q[0], final_q[1], final_q[2], final_q[3]);
    normalize_integ(&mut final_quat);

    SixDofState {
        trans: TranslationalState {
            position: final_pos,
            velocity: final_vel,
        },
        rot: RotationalState {
            quaternion: final_quat,
            ang_vel_body: final_omega,
        },
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "RK4 round-trip tests assert bit-exact recovery of analytic literals (e.g. ω·dt)"
)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test step counts fit exactly in f64 mantissa and usize"
)]
mod tests {
    use super::*;
    use crate::mass::MassProperties;
    use crate::rotational::{RotationalState, SixDofState};
    use astrodyn_math::JeodQuat;
    use glam::DMat3;

    /// Harmonic oscillator: x'' = -x
    /// Analytical solution: x(t) = cos(t), v(t) = -sin(t) with x(0) = 1, v(0) = 0.
    /// Propagate for 628 steps at dt=0.01 (t_final = 6.28, slightly less than 2*pi).
    /// Compare against analytical solution at t_final. Error < 1e-8.
    #[test]
    fn harmonic_oscillator() {
        let dt = 0.01;
        let steps = 628; // t_final = 6.28 (close to but not exactly 2*pi = 6.2832...)
        let t_final = dt * steps as f64;

        let mut state = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, 0.0),
        };

        // Acceleration: a = -x (simple harmonic oscillator along x-axis)
        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        for _ in 0..steps {
            state = rk4_translational_step(&state, accel_fn, dt);
        }

        // Compare against analytical solution at the actual final time:
        // x(t) = cos(t), v(t) = -sin(t)
        let exact_pos = t_final.cos();
        let exact_vel = -t_final.sin();

        let pos_error = (state.position.x - exact_pos).abs();
        let vel_error = (state.velocity.x - exact_vel).abs();

        // RK4 with dt=0.01 over ~628 steps. The O(h^4) local truncation error
        // accumulates to well below 1e-8 for this smooth oscillator.
        assert!(pos_error < 1e-8, "Position error {pos_error} exceeds 1e-8");
        assert!(vel_error < 1e-8, "Velocity error {vel_error} exceeds 1e-8");
    }

    /// Convergence order test: RK4 is 4th-order, so halving dt should reduce
    /// the error by a factor of ~16. We run the harmonic oscillator with dt
    /// and dt/2, then check that error_coarse / error_fine is approximately 16.
    #[test]
    fn convergence_order() {
        let dt_coarse = 0.1;
        let dt_fine = dt_coarse / 2.0;
        let total_time: f64 = 1.0; // Propagate for 1 second

        let initial = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, 0.0),
        };

        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        // Analytical solution at t=1: x = cos(1), v = -sin(1)
        let exact_pos = total_time.cos();
        let exact_vel = -total_time.sin();

        // Coarse run
        let steps_coarse = (total_time / dt_coarse).round() as usize;
        let mut state_coarse = initial;
        for _ in 0..steps_coarse {
            state_coarse = rk4_translational_step(&state_coarse, accel_fn, dt_coarse);
        }
        let error_coarse = (state_coarse.position.x - exact_pos).abs();

        // Fine run
        let steps_fine = (total_time / dt_fine).round() as usize;
        let mut state_fine = initial;
        for _ in 0..steps_fine {
            state_fine = rk4_translational_step(&state_fine, accel_fn, dt_fine);
        }
        let error_fine = (state_fine.position.x - exact_pos).abs();

        // Error ratio should be approximately 2^4 = 16 for a 4th-order method
        let ratio = error_coarse / error_fine;
        assert!(
            (ratio - 16.0).abs() < 2.0,
            "Convergence ratio {ratio} is not close to 16 (4th order)"
        );

        // Also verify the fine solution velocity convergence
        let vel_error_coarse = (state_coarse.velocity.x - exact_vel).abs();
        let vel_error_fine = (state_fine.velocity.x - exact_vel).abs();
        let vel_ratio = vel_error_coarse / vel_error_fine;
        assert!(
            (vel_ratio - 16.0).abs() < 2.0,
            "Velocity convergence ratio {vel_ratio} is not close to 16 (4th order)"
        );
    }

    /// Free particle: zero acceleration means position changes linearly and
    /// velocity remains constant.
    #[test]
    fn free_particle() {
        let dt = 0.5;
        let initial_pos = DVec3::new(1.0, 2.0, 3.0);
        let initial_vel = DVec3::new(4.0, 5.0, 6.0);

        let mut state = TranslationalState {
            position: initial_pos,
            velocity: initial_vel,
        };

        let zero_accel = |_: &TranslationalState, _t: f64| -> DVec3 { DVec3::ZERO };

        let num_steps = 10;
        for _ in 0..num_steps {
            state = rk4_translational_step(&state, zero_accel, dt);
        }

        let total_time = dt * num_steps as f64;
        let expected_pos = initial_pos + initial_vel * total_time;

        // Position should advance linearly
        let pos_error = (state.position - expected_pos).length();
        assert!(
            pos_error < 1e-12,
            "Free particle position error {pos_error} exceeds 1e-12"
        );

        // Velocity should remain constant
        let vel_error = (state.velocity - initial_vel).length();
        assert!(
            vel_error < 1e-12,
            "Free particle velocity error {vel_error} exceeds 1e-12"
        );
    }

    // ---------------------------------------------------------------
    // 6-DOF integration tests
    // ---------------------------------------------------------------

    /// Helper: create MassProperties with a diagonal inertia tensor.
    fn mass_with_inertia(mass: f64, ix: f64, iy: f64, iz: f64) -> MassProperties {
        let inertia = DMat3::from_diagonal(DVec3::new(ix, iy, iz));
        let inverse_inertia = DMat3::from_diagonal(DVec3::new(1.0 / ix, 1.0 / iy, 1.0 / iz));
        MassProperties {
            mass,
            inverse_mass: 1.0 / mass,
            inertia,
            inverse_inertia,
            position: DVec3::ZERO,
            t_parent_this: DMat3::IDENTITY,
            dirty: false,
        }
    }

    /// Torque-free symmetric body: I = diag(10, 10, 20), omega_0 = [0.01, 0, 0.1].
    ///
    /// For a torque-free axisymmetric body (I1 = I2), the spin about the
    /// symmetry axis (omega_z) is constant and the transverse components
    /// precess at rate omega_p = (I3 - I1) / I1 * omega_z.
    ///
    /// With I1=I2=10, I3=20, omega_z=0.1:
    ///   omega_p = (20-10)/10 * 0.1 = 0.1 rad/s
    ///   Period = 2*pi / 0.1 = 62.83... seconds
    ///
    /// After one full precession period, omega should return to its initial value.
    #[test]
    fn torque_free_symmetric_body() {
        let mass_props = mass_with_inertia(100.0, 10.0, 10.0, 20.0);
        let omega_z = 0.1;
        let omega_x0 = 0.01;

        // Precession rate and period
        let omega_p = (20.0 - 10.0) / 10.0 * omega_z; // = 0.1 rad/s
        let period = std::f64::consts::TAU / omega_p; // = 2*pi/0.1 s

        let initial = SixDofState {
            trans: TranslationalState {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            rot: RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::new(omega_x0, 0.0, omega_z),
            },
        };

        let dt = 0.001;
        let steps = (period / dt).round() as usize;

        let zero_accel = |_: &SixDofState, _t: f64| -> DVec3 { DVec3::ZERO };
        let zero_torque = |_: &SixDofState| -> DVec3 { DVec3::ZERO };

        let mut state = initial;
        for _ in 0..steps {
            state = rk4_sixdof_step(&state, zero_accel, zero_torque, &mass_props, dt);
        }

        // omega_z should remain constant (axisymmetric body)
        let omega_z_err = (state.rot.ang_vel_body.z - omega_z).abs();
        assert!(
            omega_z_err < 1e-10,
            "omega_z should be constant, error = {}",
            omega_z_err,
        );

        // Transverse omega should return to initial after one precession period.
        // omega_x(t) = omega_x0 * cos(omega_p * t)
        // omega_y(t) = -omega_x0 * sin(omega_p * t)   (sign depends on convention)
        // At t = period: omega_x = omega_x0, omega_y = 0
        let omega_x_err = (state.rot.ang_vel_body.x - omega_x0).abs();
        let omega_y_err = state.rot.ang_vel_body.y.abs();
        let rel_err_x = omega_x_err / omega_x0;
        let rel_err_y = omega_y_err / omega_x0;

        assert!(
            rel_err_x < 1e-3,
            "omega_x relative error {} exceeds 0.1% after one precession period",
            rel_err_x,
        );
        assert!(
            rel_err_y < 1e-3,
            "omega_y relative error {} exceeds 0.1% after one precession period",
            rel_err_y,
        );
    }

    /// Quaternion norm preservation over a long propagation (86400s at dt=1s).
    /// The norm should stay within 1e-14 of unity thanks to normalize_integ.
    #[test]
    fn quaternion_norm_preservation() {
        let mass_props = mass_with_inertia(100.0, 10.0, 20.0, 30.0);

        let initial = SixDofState {
            trans: TranslationalState {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
            rot: RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::new(0.01, 0.02, 0.05),
            },
        };

        let dt = 1.0;
        let total_seconds = 86400;

        let zero_accel = |_: &SixDofState, _t: f64| -> DVec3 { DVec3::ZERO };
        let zero_torque = |_: &SixDofState| -> DVec3 { DVec3::ZERO };

        let mut state = initial;
        let mut max_norm_err = 0.0_f64;

        for _ in 0..total_seconds {
            state = rk4_sixdof_step(&state, zero_accel, zero_torque, &mass_props, dt);
            let norm_err = (state.rot.quaternion.norm_sq() - 1.0).abs();
            max_norm_err = max_norm_err.max(norm_err);
        }

        assert!(
            max_norm_err < 1e-14,
            "Max quaternion norm error over 86400s: {} (exceeds 1e-14)",
            max_norm_err,
        );
    }

    /// Pure translation (zero torque, zero angular velocity): the 6-DOF
    /// integrator should produce the same translational result as the
    /// 3-DOF integrator, and the quaternion should remain identity.
    #[test]
    fn sixdof_pure_translation_matches_translational() {
        let mass_props = MassProperties::new(100.0);

        let initial_pos = DVec3::new(7_000_000.0, 0.0, 0.0);
        let initial_vel = DVec3::new(0.0, 7_500.0, 0.0);

        // Simple 1/r^2 gravity for testing
        let mu = 3.986_004_415e14;

        // 3-DOF reference
        let mut state_3dof = TranslationalState {
            position: initial_pos,
            velocity: initial_vel,
        };

        // 6-DOF
        let mut state_6dof = SixDofState {
            trans: TranslationalState {
                position: initial_pos,
                velocity: initial_vel,
            },
            rot: RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            },
        };

        let dt = 10.0;
        let steps = 100;

        for _ in 0..steps {
            state_3dof = rk4_translational_step(
                &state_3dof,
                |s, _t| {
                    let r = s.position.length();
                    -mu / (r * r * r) * s.position
                },
                dt,
            );

            state_6dof = rk4_sixdof_step(
                &state_6dof,
                |s, _t| {
                    let r = s.trans.position.length();
                    -mu / (r * r * r) * s.trans.position
                },
                |_| DVec3::ZERO,
                &mass_props,
                dt,
            );
        }

        // Translational states should match exactly (within floating-point)
        let pos_diff = (state_6dof.trans.position - state_3dof.position).length();
        let vel_diff = (state_6dof.trans.velocity - state_3dof.velocity).length();
        assert!(
            pos_diff < 1e-6,
            "Position difference between 3DOF and 6DOF: {} m",
            pos_diff,
        );
        assert!(
            vel_diff < 1e-9,
            "Velocity difference between 3DOF and 6DOF: {} m/s",
            vel_diff,
        );

        // Quaternion should remain identity (no rotation)
        let q = state_6dof.rot.quaternion;
        assert!(
            (q.scalar() - 1.0).abs() < 1e-14,
            "Quaternion scalar should be 1.0, got {}",
            q.scalar(),
        );
        assert!(
            q.vector().length() < 1e-14,
            "Quaternion vector should be zero, got {:?}",
            q.vector(),
        );
    }
}
