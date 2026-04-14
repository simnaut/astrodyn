//! Runge-Kutta-Fehlberg 4(5) integrator.
//!
//! Port of JEOD/Trick ER7 `rkf45_butcher_tableau.cc` coefficients and
//! `rkf45_second_order_ode_integrator.cc` step logic.
//!
//! This implementation uses the RKF45 6-stage tableau and applies the
//! 5th-order solution (b5 weights) for the state update.
//! It does not currently compute or expose the embedded 4th-order solution
//! or an error estimate for adaptive step control.

use crate::mass::MassProperties;
use crate::rotational::*;
use crate::state::TranslationalState;
use glam::DVec3;
use jeod_math::JeodQuat;

// ── Butcher tableau coefficients ──
// From trick_source/er7_utils/integration/rkf45/src/rkf45_butcher_tableau.cc

// a_ij matrix (lower triangular)
const A21: f64 = 1.0 / 4.0;

const A31: f64 = 3.0 / 32.0;
const A32: f64 = 9.0 / 32.0;

const A41: f64 = 1932.0 / 2197.0;
const A42: f64 = -7200.0 / 2197.0;
const A43: f64 = 7296.0 / 2197.0;

const A51: f64 = 439.0 / 216.0;
const A52: f64 = -8.0;
const A53: f64 = 3680.0 / 513.0;
const A54: f64 = -845.0 / 4104.0;

const A61: f64 = -8.0 / 27.0;
const A62: f64 = 2.0;
const A63: f64 = -3544.0 / 2565.0;
const A64: f64 = 1859.0 / 4104.0;
const A65: f64 = -11.0 / 40.0;

// b5 weights (5th order — used for state update)
const B51: f64 = 16.0 / 135.0;
// B52 = 0
const B53: f64 = 6656.0 / 12825.0;
const B54: f64 = 28561.0 / 56430.0;
const B55: f64 = -9.0 / 50.0;
const B56: f64 = 2.0 / 55.0;

// c coefficients (stage times) for reference:
// c = [0, 1/4, 3/8, 12/13, 1, 1/2]
// Not used directly — the a_ij coefficients encode the same information.

/// Advance translational state by one RKF45 step (5th-order result).
///
/// Uses 6 derivative evaluations to compute the 5th-order (b5) solution.
/// The `accel_fn` is called 6 times at intermediate states.
pub fn rkf45_translational_step(
    state: &TranslationalState,
    accel_fn: impl Fn(&TranslationalState, f64) -> DVec3,
    dt: f64,
) -> TranslationalState {
    let pos0 = state.position;
    let vel0 = state.velocity;

    // Stage 1: evaluate at t
    let k1_v = vel0;
    let k1_a = accel_fn(state, 0.0);

    // Stage 2: evaluate at t + c2*dt
    let s2 = TranslationalState {
        position: pos0 + k1_v * (A21 * dt),
        velocity: vel0 + k1_a * (A21 * dt),
    };
    let k2_v = s2.velocity;
    let k2_a = accel_fn(&s2, 0.25);

    // Stage 3: evaluate at t + c3*dt
    let s3 = TranslationalState {
        position: pos0 + (k1_v * A31 + k2_v * A32) * dt,
        velocity: vel0 + (k1_a * A31 + k2_a * A32) * dt,
    };
    let k3_v = s3.velocity;
    let k3_a = accel_fn(&s3, 3.0 / 8.0);

    // Stage 4: evaluate at t + c4*dt
    let s4 = TranslationalState {
        position: pos0 + (k1_v * A41 + k2_v * A42 + k3_v * A43) * dt,
        velocity: vel0 + (k1_a * A41 + k2_a * A42 + k3_a * A43) * dt,
    };
    let k4_v = s4.velocity;
    let k4_a = accel_fn(&s4, 12.0 / 13.0);

    // Stage 5: evaluate at t + dt
    let s5 = TranslationalState {
        position: pos0 + (k1_v * A51 + k2_v * A52 + k3_v * A53 + k4_v * A54) * dt,
        velocity: vel0 + (k1_a * A51 + k2_a * A52 + k3_a * A53 + k4_a * A54) * dt,
    };
    let k5_v = s5.velocity;
    let k5_a = accel_fn(&s5, 1.0);

    // Stage 6: evaluate at t + c6*dt
    let s6 = TranslationalState {
        position: pos0 + (k1_v * A61 + k2_v * A62 + k3_v * A63 + k4_v * A64 + k5_v * A65) * dt,
        velocity: vel0 + (k1_a * A61 + k2_a * A62 + k3_a * A63 + k4_a * A64 + k5_a * A65) * dt,
    };
    let k6_v = s6.velocity;
    let k6_a = accel_fn(&s6, 0.5);

    // 5th-order combination (b5 weights)
    TranslationalState {
        position: pos0 + (k1_v * B51 + k3_v * B53 + k4_v * B54 + k5_v * B55 + k6_v * B56) * dt,
        velocity: vel0 + (k1_a * B51 + k3_a * B53 + k4_a * B54 + k5_a * B55 + k6_a * B56) * dt,
    }
}

/// Advance a 6-DOF state by one RKF45 step (5th-order result).
///
/// Integrates 13 variables simultaneously: position[3], velocity[3],
/// quaternion[4], angular velocity[3]. Uses 6 derivative evaluations.
pub fn rkf45_sixdof_step(
    state: &SixDofState,
    accel_fn: impl Fn(&SixDofState, f64) -> DVec3,
    torque_fn: impl Fn(&SixDofState) -> DVec3,
    mass_props: &MassProperties,
    dt: f64,
) -> SixDofState {
    let pos0 = state.trans.position;
    let vel0 = state.trans.velocity;
    let q0 = state.rot.quaternion.data;
    let omega0 = state.rot.ang_vel_body;

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

    let eval_derivs = |s: &SixDofState, time_frac: f64| -> (DVec3, DVec3, [f64; 4], DVec3) {
        let k_v = s.trans.velocity;
        let k_a = accel_fn(s, time_frac);
        let k_qdot = compute_left_quat_deriv(&s.rot.quaternion, s.rot.ang_vel_body);
        let k_alpha = compute_rotational_acceleration(
            &mass_props.inertia,
            &mass_props.inverse_inertia,
            s.rot.ang_vel_body,
            torque_fn(s),
        );
        (k_v, k_a, k_qdot, k_alpha)
    };

    let step_q = |q_base: [f64; 4], stages: &[([f64; 4], f64)]| -> [f64; 4] {
        let mut result = q_base;
        for (k, coeff) in stages {
            for i in 0..4 {
                result[i] += k[i] * coeff;
            }
        }
        result
    };

    // Stage 1
    let (k1_v, k1_a, k1_qd, k1_al) = eval_derivs(state, 0.0);
    let h1 = A21 * dt;

    // Stage 2
    let s2 = make_state(
        pos0 + k1_v * h1,
        vel0 + k1_a * h1,
        step_q(q0, &[(k1_qd, h1)]),
        omega0 + k1_al * h1,
    );
    let (k2_v, k2_a, k2_qd, k2_al) = eval_derivs(&s2, 0.25);

    // Stage 3
    let s3 = make_state(
        pos0 + (k1_v * A31 + k2_v * A32) * dt,
        vel0 + (k1_a * A31 + k2_a * A32) * dt,
        step_q(q0, &[(k1_qd, A31 * dt), (k2_qd, A32 * dt)]),
        omega0 + (k1_al * A31 + k2_al * A32) * dt,
    );
    let (k3_v, k3_a, k3_qd, k3_al) = eval_derivs(&s3, 3.0 / 8.0);

    // Stage 4
    let s4 = make_state(
        pos0 + (k1_v * A41 + k2_v * A42 + k3_v * A43) * dt,
        vel0 + (k1_a * A41 + k2_a * A42 + k3_a * A43) * dt,
        step_q(
            q0,
            &[(k1_qd, A41 * dt), (k2_qd, A42 * dt), (k3_qd, A43 * dt)],
        ),
        omega0 + (k1_al * A41 + k2_al * A42 + k3_al * A43) * dt,
    );
    let (k4_v, k4_a, k4_qd, k4_al) = eval_derivs(&s4, 12.0 / 13.0);

    // Stage 5
    let s5 = make_state(
        pos0 + (k1_v * A51 + k2_v * A52 + k3_v * A53 + k4_v * A54) * dt,
        vel0 + (k1_a * A51 + k2_a * A52 + k3_a * A53 + k4_a * A54) * dt,
        step_q(
            q0,
            &[
                (k1_qd, A51 * dt),
                (k2_qd, A52 * dt),
                (k3_qd, A53 * dt),
                (k4_qd, A54 * dt),
            ],
        ),
        omega0 + (k1_al * A51 + k2_al * A52 + k3_al * A53 + k4_al * A54) * dt,
    );
    let (k5_v, k5_a, k5_qd, k5_al) = eval_derivs(&s5, 1.0);

    // Stage 6
    let s6 = make_state(
        pos0 + (k1_v * A61 + k2_v * A62 + k3_v * A63 + k4_v * A64 + k5_v * A65) * dt,
        vel0 + (k1_a * A61 + k2_a * A62 + k3_a * A63 + k4_a * A64 + k5_a * A65) * dt,
        step_q(
            q0,
            &[
                (k1_qd, A61 * dt),
                (k2_qd, A62 * dt),
                (k3_qd, A63 * dt),
                (k4_qd, A64 * dt),
                (k5_qd, A65 * dt),
            ],
        ),
        omega0 + (k1_al * A61 + k2_al * A62 + k3_al * A63 + k4_al * A64 + k5_al * A65) * dt,
    );
    let (k6_v, k6_a, k6_qd, k6_al) = eval_derivs(&s6, 0.5);

    // 5th-order combination (b5 weights, b52=0)
    let final_pos = pos0 + (k1_v * B51 + k3_v * B53 + k4_v * B54 + k5_v * B55 + k6_v * B56) * dt;
    let final_vel = vel0 + (k1_a * B51 + k3_a * B53 + k4_a * B54 + k5_a * B55 + k6_a * B56) * dt;
    let final_omega =
        omega0 + (k1_al * B51 + k3_al * B53 + k4_al * B54 + k5_al * B55 + k6_al * B56) * dt;

    let final_q = step_q(
        q0,
        &[
            (k1_qd, B51 * dt),
            (k3_qd, B53 * dt),
            (k4_qd, B54 * dt),
            (k5_qd, B55 * dt),
            (k6_qd, B56 * dt),
        ],
    );

    // JEOD_INV: DB.09 — quaternion normalized after every integration step
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
mod tests {
    use super::*;
    use crate::integration::rk4_translational_step;

    /// RKF45 should be more accurate than RK4 on the harmonic oscillator
    /// for the same step size (5th order vs 4th order).
    #[test]
    fn rkf45_more_accurate_than_rk4() {
        let dt: f64 = 0.1;
        let steps = 100; // t_final = 10.0
        let t_final = dt * steps as f64;

        let initial = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };

        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };

        let mut state_rk4 = initial;
        let mut state_rkf45 = initial;
        for _ in 0..steps {
            state_rk4 = rk4_translational_step(&state_rk4, accel_fn, dt);
            state_rkf45 = rkf45_translational_step(&state_rkf45, accel_fn, dt);
        }

        let exact_pos = t_final.cos();
        let err_rk4 = (state_rk4.position.x - exact_pos).abs();
        let err_rkf45 = (state_rkf45.position.x - exact_pos).abs();

        // RKF45 (5th order) should be at least 10x more accurate than RK4
        // at this step size.
        assert!(
            err_rkf45 < err_rk4 / 10.0,
            "RKF45 error ({err_rkf45:.2e}) should be << RK4 error ({err_rk4:.2e})"
        );
    }

    /// Convergence order: halving dt should reduce error by ~32x (5th order).
    #[test]
    fn rkf45_convergence_order() {
        let dt_coarse: f64 = 0.1;
        let dt_fine = dt_coarse / 2.0;
        let total_time: f64 = 1.0;

        let initial = TranslationalState {
            position: DVec3::new(1.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        };

        let accel_fn = |s: &TranslationalState, _t: f64| -> DVec3 { -s.position };
        let exact_pos = total_time.cos();

        let steps_coarse = (total_time / dt_coarse).round() as usize;
        let mut state_coarse = initial;
        for _ in 0..steps_coarse {
            state_coarse = rkf45_translational_step(&state_coarse, accel_fn, dt_coarse);
        }
        let err_coarse = (state_coarse.position.x - exact_pos).abs();

        let steps_fine = (total_time / dt_fine).round() as usize;
        let mut state_fine = initial;
        for _ in 0..steps_fine {
            state_fine = rkf45_translational_step(&state_fine, accel_fn, dt_fine);
        }
        let err_fine = (state_fine.position.x - exact_pos).abs();

        // 5th order: error ratio should be ~2^5 = 32
        let ratio = err_coarse / err_fine;
        assert!(
            (ratio - 32.0).abs() < 8.0,
            "Convergence ratio {ratio:.1} not close to 32 (5th order)"
        );
    }

    /// Free particle: zero acceleration, position advances linearly.
    #[test]
    fn rkf45_free_particle() {
        let initial_pos = DVec3::new(1.0, 2.0, 3.0);
        let initial_vel = DVec3::new(4.0, 5.0, 6.0);
        let dt = 0.5;

        let mut state = TranslationalState {
            position: initial_pos,
            velocity: initial_vel,
        };

        let zero_accel = |_: &TranslationalState, _t: f64| DVec3::ZERO;

        for _ in 0..10 {
            state = rkf45_translational_step(&state, zero_accel, dt);
        }

        let expected_pos = initial_pos + initial_vel * 5.0;
        let pos_error = (state.position - expected_pos).length();
        assert!(pos_error < 1e-12, "Free particle pos error: {pos_error}");
    }
}
