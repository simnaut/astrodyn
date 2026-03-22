use crate::state::TranslationalState;
use glam::DVec3;

/// Advance translational state by one RK4 step.
///
/// The `accel_fn` computes acceleration from the current state. It is called
/// 4 times (once per RK4 stage) at intermediate positions, enabling correct
/// multi-stage integration even when forces depend on position (e.g., gravity).
pub fn rk4_translational_step(
    state: &TranslationalState,
    accel_fn: impl Fn(&TranslationalState) -> DVec3,
    dt: f64,
) -> TranslationalState {
    // Stage 1: evaluate at current state
    let k1_a = accel_fn(state);
    let k1_v = state.velocity;

    // Stage 2: evaluate at t + dt/2, using k1
    let s2 = TranslationalState {
        position: state.position + k1_v * (dt * 0.5),
        velocity: state.velocity + k1_a * (dt * 0.5),
    };
    let k2_a = accel_fn(&s2);
    let k2_v = s2.velocity;

    // Stage 3: evaluate at t + dt/2, using k2
    let s3 = TranslationalState {
        position: state.position + k2_v * (dt * 0.5),
        velocity: state.velocity + k2_a * (dt * 0.5),
    };
    let k3_a = accel_fn(&s3);
    let k3_v = s3.velocity;

    // Stage 4: evaluate at t + dt, using k3
    let s4 = TranslationalState {
        position: state.position + k3_v * dt,
        velocity: state.velocity + k3_a * dt,
    };
    let k4_a = accel_fn(&s4);
    let k4_v = s4.velocity;

    // Combine: weighted average
    let sixth_dt = dt / 6.0;
    TranslationalState {
        position: state.position + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * sixth_dt,
        velocity: state.velocity + (k1_a + k2_a * 2.0 + k3_a * 2.0 + k4_a) * sixth_dt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let accel_fn = |s: &TranslationalState| -> DVec3 { -s.position };

        for _ in 0..steps {
            state = rk4_translational_step(&state, &accel_fn, dt);
        }

        // Compare against analytical solution at the actual final time:
        // x(t) = cos(t), v(t) = -sin(t)
        let exact_pos = t_final.cos();
        let exact_vel = -t_final.sin();

        let pos_error = (state.position.x - exact_pos).abs();
        let vel_error = (state.velocity.x - exact_vel).abs();

        // RK4 with dt=0.01 over ~628 steps. The O(h^4) local truncation error
        // accumulates to well below 1e-8 for this smooth oscillator.
        assert!(
            pos_error < 1e-8,
            "Position error {pos_error} exceeds 1e-8"
        );
        assert!(
            vel_error < 1e-8,
            "Velocity error {vel_error} exceeds 1e-8"
        );
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

        let accel_fn = |s: &TranslationalState| -> DVec3 { -s.position };

        // Analytical solution at t=1: x = cos(1), v = -sin(1)
        let exact_pos = total_time.cos();
        let exact_vel = -total_time.sin();

        // Coarse run
        let steps_coarse = (total_time / dt_coarse).round() as usize;
        let mut state_coarse = initial;
        for _ in 0..steps_coarse {
            state_coarse = rk4_translational_step(&state_coarse, &accel_fn, dt_coarse);
        }
        let error_coarse = (state_coarse.position.x - exact_pos).abs();

        // Fine run
        let steps_fine = (total_time / dt_fine).round() as usize;
        let mut state_fine = initial;
        for _ in 0..steps_fine {
            state_fine = rk4_translational_step(&state_fine, &accel_fn, dt_fine);
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

        let zero_accel = |_: &TranslationalState| -> DVec3 { DVec3::ZERO };

        let num_steps = 10;
        for _ in 0..num_steps {
            state = rk4_translational_step(&state, &zero_accel, dt);
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
}
