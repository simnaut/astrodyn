//! Holonomic constraint dynamics with Baumgarte stabilization.
//!
//! This module provides a simplified constraint-dynamics facility for point-mass
//! systems. Unlike JEOD's full matrix-based solver
//! (`experimental/constraints/src/dyn_body_constraints_solver.cc` — which handles
//! coupled multi-constraint systems with force/torque constraints on rigid bodies),
//! this implementation covers the single-constraint, single-particle case that is
//! sufficient for pendulum-style analytical testing.
//!
//! # Mathematical background
//!
//! A **holonomic constraint** is a position-level equation of the form
//! `g(q) = 0`. Differentiating once gives the velocity-level constraint
//! `g_dot = J(q) · v = 0`, where `J = dg/dq` is the constraint Jacobian.
//! Differentiating again gives the acceleration-level constraint
//! `g_ddot = J(q) · a + h(q, v) = 0`, where `h(q, v) = v^T · H · v` is the
//! Hessian contribution (the `hessian_contribution` trait method returns `h`
//! directly — callers do not need to form the Hessian matrix `H`).
//!
//! # Baumgarte stabilization
//!
//! Numerically integrating `g_ddot = 0` allows `g` and `g_dot` to drift from
//! zero due to floating-point error. Baumgarte's trick replaces the pure
//! acceleration-level constraint with a damped harmonic oscillator in `g`:
//!
//! ```text
//! g_ddot + 2·alpha·g_dot + beta^2·g = 0
//! ```
//!
//! where `alpha` (damping) and `beta` (stiffness) are tuning parameters with
//! units of inverse time. Choosing `alpha = beta` gives critical damping of
//! the constraint error. Typical values are a few multiples of the natural
//! frequency of the constrained system (e.g., 1–10 rad/s for a 1 m pendulum).
//!
//! # Lagrange multiplier
//!
//! For a single scalar constraint on a point mass, the constraint force is
//! `F_c = lambda · J`, where `lambda` is the Lagrange multiplier. The
//! constrained acceleration is
//!
//! ```text
//! a = a_free + lambda · J / m
//! ```
//!
//! Substituting into the stabilized constraint equation and solving for
//! `lambda`:
//!
//! ```text
//! lambda = -m · (J · a_free + h + 2·alpha·(J·v) + beta^2·g) / (J · J)
//! ```
//!
//! The constrained acceleration is then `a_free + lambda · J / m`.

use glam::DVec3;

/// Holonomic position-level constraint: `g(q) = 0`.
///
/// Implementors provide the constraint scalar `g`, its gradient `J = dg/dq`,
/// and the Hessian-velocity product `v^T · H · v` (the second derivative of
/// `g` arising from `v` alone, without `a`).
pub trait HolonomicConstraint {
    /// Evaluate the constraint scalar `g(pos)`.
    ///
    /// Zero when the constraint is satisfied.
    fn evaluate(&self, pos: DVec3) -> f64;

    /// Evaluate the constraint Jacobian `dg/dq` at `pos`.
    ///
    /// For a scalar constraint on a 3D position this is a 3-vector (the
    /// gradient of `g`).
    fn jacobian(&self, pos: DVec3) -> DVec3;

    /// Evaluate the Hessian-velocity contribution `v^T · H · v` to `g_ddot`.
    ///
    /// This is the part of `g_ddot` that does not depend on acceleration —
    /// i.e., the `d/dt(J · v)` term minus `J · a`.
    fn hessian_contribution(&self, pos: DVec3, vel: DVec3) -> f64;
}

/// Spherical pendulum constraint: the distance from `pos` to `attachment_point`
/// is held at `length`.
///
/// Uses the squared-length form `g = |r|^2 - L^2` (with `r = pos - attachment`)
/// rather than `g = |r| - L` because the squared form is polynomial and has a
/// well-defined gradient at all positions (the unsquared form has a singular
/// gradient when `|r| = 0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PendulumConstraint {
    /// Fixed hinge point in the inertial frame, in meters.
    pub attachment_point: DVec3,
    /// Constant distance from the attachment point to the bob, in meters.
    pub length: f64,
}

impl PendulumConstraint {
    /// Construct a new pendulum constraint.
    ///
    /// `length` must be positive.
    pub fn new(attachment_point: DVec3, length: f64) -> Self {
        assert!(length > 0.0, "pendulum length must be positive");
        Self {
            attachment_point,
            length,
        }
    }
}

impl HolonomicConstraint for PendulumConstraint {
    /// `g(pos) = |pos - attachment|^2 - L^2`.
    fn evaluate(&self, pos: DVec3) -> f64 {
        let r = pos - self.attachment_point;
        r.length_squared() - self.length * self.length
    }

    /// `dg/dq = 2 (pos - attachment)`.
    fn jacobian(&self, pos: DVec3) -> DVec3 {
        2.0 * (pos - self.attachment_point)
    }

    /// `v^T · H · v = 2 |v|^2`.
    ///
    /// With `g = (pos - a)·(pos - a) - L^2`, we have `g_dot = 2 (pos-a)·v`
    /// and `g_ddot = 2 (pos-a)·a + 2 v·v`. The second term is the Hessian
    /// contribution.
    fn hessian_contribution(&self, _pos: DVec3, vel: DVec3) -> f64 {
        2.0 * vel.length_squared()
    }
}

/// Baumgarte stabilization parameters for a holonomic constraint.
///
/// `alpha` damps velocity-level drift; `beta` restores position-level drift.
/// Choose `alpha = beta` for critical damping. Units are 1/s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaumgarteSolver {
    /// Velocity-level damping coefficient, in 1/s.
    pub alpha: f64,
    /// Position-level stiffness coefficient, in 1/s.
    pub beta: f64,
}

impl BaumgarteSolver {
    /// Critically damped solver: `alpha = beta = omega`.
    pub fn critically_damped(omega: f64) -> Self {
        Self {
            alpha: omega,
            beta: omega,
        }
    }
}

impl Default for BaumgarteSolver {
    /// Default to 5 rad/s critical damping (reasonable for sub-meter pendulums).
    fn default() -> Self {
        Self::critically_damped(5.0)
    }
}

/// Project a free (unconstrained) acceleration onto the constraint manifold.
///
/// Given the current position, velocity, free (unconstrained) acceleration,
/// and mass of a point particle, plus a holonomic constraint and Baumgarte
/// stabilization parameters, returns the constrained acceleration that
/// approximately satisfies the constraint at the acceleration level:
///
/// ```text
/// g_ddot + 2·alpha·g_dot + beta^2·g = 0
/// ```
///
/// If the constraint Jacobian is (numerically) zero — i.e., the particle sits
/// at a stationary point of `g` — no constraint force can be applied and the
/// free acceleration is returned unchanged.
///
/// # Arguments
/// - `pos`: current position in the inertial frame (m).
/// - `vel`: current velocity in the inertial frame (m/s).
/// - `free_accel`: acceleration the particle would experience without the
///   constraint (m/s^2). This is the sum of all non-constraint forces divided
///   by mass.
/// - `mass`: particle mass (kg). Must be positive.
/// - `constraint`: the holonomic constraint to enforce.
/// - `solver`: Baumgarte stabilization parameters.
pub fn apply_constraint(
    pos: DVec3,
    vel: DVec3,
    free_accel: DVec3,
    mass: f64,
    constraint: &impl HolonomicConstraint,
    solver: &BaumgarteSolver,
) -> DVec3 {
    assert!(mass > 0.0, "mass must be positive");

    let g = constraint.evaluate(pos);
    let j = constraint.jacobian(pos);
    let h = constraint.hessian_contribution(pos, vel);

    let jj = j.length_squared();
    // If the Jacobian vanishes the constraint degenerates — the particle is at
    // a stationary point of g and no force can change g_ddot. Return unchanged.
    if jj == 0.0 {
        return free_accel;
    }

    let g_dot = j.dot(vel);
    let j_a = j.dot(free_accel);

    // Baumgarte: g_ddot + 2 alpha g_dot + beta^2 g = 0
    // g_ddot = J·a + h, and a = a_free + lambda·J/m, giving
    //   J·a_free + lambda·(J·J)/m + h + 2 alpha g_dot + beta^2 g = 0
    //   lambda = -m (J·a_free + h + 2 alpha g_dot + beta^2 g) / (J·J)
    let lambda =
        -mass * (j_a + h + 2.0 * solver.alpha * g_dot + solver.beta * solver.beta * g) / jj;

    free_accel + (lambda / mass) * j
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PendulumConstraint trait impl ──

    #[test]
    fn pendulum_constraint_evaluates_to_zero_at_correct_distance() {
        let c = PendulumConstraint::new(DVec3::ZERO, 2.0);
        // Any point on the sphere of radius 2 gives g = 0
        assert!(c.evaluate(DVec3::new(2.0, 0.0, 0.0)).abs() < 1e-15);
        assert!(c.evaluate(DVec3::new(0.0, 2.0, 0.0)).abs() < 1e-15);
        assert!(c.evaluate(DVec3::new(0.0, 0.0, -2.0)).abs() < 1e-15);
        let off_axis = DVec3::new(1.0, 1.0, (2.0_f64).sqrt());
        assert!(c.evaluate(off_axis).abs() < 1e-14);
    }

    #[test]
    fn pendulum_constraint_nonzero_off_manifold() {
        let c = PendulumConstraint::new(DVec3::ZERO, 1.0);
        // Inside the sphere: g < 0
        assert!(c.evaluate(DVec3::new(0.5, 0.0, 0.0)) < 0.0);
        // Outside the sphere: g > 0
        assert!(c.evaluate(DVec3::new(2.0, 0.0, 0.0)) > 0.0);
    }

    #[test]
    fn pendulum_jacobian_points_radially_outward() {
        let attachment = DVec3::new(1.0, 2.0, 3.0);
        let c = PendulumConstraint::new(attachment, 5.0);
        let pos = DVec3::new(4.0, 2.0, 3.0); // offset +3 in x from attachment
        let j = c.jacobian(pos);
        // Expected: 2 * (pos - attachment) = (6, 0, 0)
        assert!((j - DVec3::new(6.0, 0.0, 0.0)).length() < 1e-14);
    }

    #[test]
    fn pendulum_hessian_contribution_matches_2_vsq() {
        let c = PendulumConstraint::new(DVec3::ZERO, 1.0);
        let vel = DVec3::new(3.0, 4.0, 0.0);
        // |v|^2 = 25, so h = 50
        assert!((c.hessian_contribution(DVec3::X, vel) - 50.0).abs() < 1e-14);
    }

    #[test]
    #[should_panic(expected = "length must be positive")]
    fn pendulum_constraint_rejects_nonpositive_length() {
        PendulumConstraint::new(DVec3::ZERO, 0.0);
    }

    // ── apply_constraint() ──

    #[test]
    fn unconstrained_free_fall_unchanged_when_tangent() {
        // Bob at (1, 0, 0), attachment at origin, L = 1.
        // Velocity tangent to the constraint surface (perpendicular to radial).
        // Gravity g_vec along -y is tangent to the constraint at this position;
        // however, the centripetal Hessian term means the constraint *does*
        // apply a force. The test we want: a purely tangential acceleration
        // *combined with* the required centripetal acceleration is unchanged.
        //
        // Here we check: with alpha=beta=0 (no stabilization) and zero velocity
        // and a zero free_accel, the constrained accel is also zero.
        let c = PendulumConstraint::new(DVec3::ZERO, 1.0);
        let solver = BaumgarteSolver {
            alpha: 0.0,
            beta: 0.0,
        };
        let pos = DVec3::X;
        let vel = DVec3::ZERO;
        let free_accel = DVec3::ZERO;
        let a = apply_constraint(pos, vel, free_accel, 1.0, &c, &solver);
        assert!(a.length() < 1e-14, "expected zero accel, got {a:?}");
    }

    #[test]
    fn radial_free_accel_gets_projected_out() {
        // Bob at (1, 0, 0), L = 1, v = 0. Apply a purely radial free accel.
        // With alpha=beta=0, the constraint should kill the radial component.
        let c = PendulumConstraint::new(DVec3::ZERO, 1.0);
        let solver = BaumgarteSolver {
            alpha: 0.0,
            beta: 0.0,
        };
        let pos = DVec3::X;
        let vel = DVec3::ZERO;
        let free_accel = DVec3::new(7.0, 0.0, 0.0); // purely radial outward
        let a = apply_constraint(pos, vel, free_accel, 1.0, &c, &solver);
        // Expected: zero (radial acceleration entirely cancelled, no centripetal
        // needed since v = 0)
        assert!(a.length() < 1e-13, "expected zero, got {a:?}");
    }

    #[test]
    fn tangential_free_accel_preserved_with_centripetal_added() {
        // Bob at (1, 0, 0), L = 1, v = (0, 1, 0) (tangential).
        // Free accel in -y (tangential). Expected constrained accel:
        // tangential component preserved, plus centripetal term -omega^2 * r.
        // omega = v/L = 1 rad/s, centripetal = (-1, 0, 0).
        let c = PendulumConstraint::new(DVec3::ZERO, 1.0);
        let solver = BaumgarteSolver {
            alpha: 0.0,
            beta: 0.0,
        };
        let pos = DVec3::X;
        let vel = DVec3::Y;
        let free_accel = DVec3::new(0.0, -1.0, 0.0); // tangential "gravity"
        let a = apply_constraint(pos, vel, free_accel, 1.0, &c, &solver);
        // Tangential component should match free_accel.y
        // Radial component: -|v|^2/L = -1.0 (centripetal)
        let expected = DVec3::new(-1.0, -1.0, 0.0);
        assert!(
            (a - expected).length() < 1e-13,
            "expected {expected:?}, got {a:?}"
        );
    }

    #[test]
    fn zero_jacobian_returns_free_accel() {
        // Degenerate "constraint" whose Jacobian is always zero.
        struct Degenerate;
        impl HolonomicConstraint for Degenerate {
            fn evaluate(&self, _pos: DVec3) -> f64 {
                0.0
            }
            fn jacobian(&self, _pos: DVec3) -> DVec3 {
                DVec3::ZERO
            }
            fn hessian_contribution(&self, _pos: DVec3, _vel: DVec3) -> f64 {
                0.0
            }
        }
        let d = Degenerate;
        let s = BaumgarteSolver::default();
        let free = DVec3::new(1.0, 2.0, 3.0);
        let a = apply_constraint(DVec3::ZERO, DVec3::ZERO, free, 1.0, &d, &s);
        assert_eq!(a, free);
    }

    // ── Integration tests ──
    //
    // These tests integrate a simple pendulum with gravity using symplectic
    // Euler (kick-drift-kick would require re-evaluating the constraint at
    // midstep; semi-implicit Euler is sufficient for verifying the constraint
    // itself).

    /// Symplectic (semi-implicit) Euler step for a constrained point mass.
    /// Returns the new (pos, vel).
    fn step_constrained(
        pos: DVec3,
        vel: DVec3,
        free_accel_fn: impl Fn(DVec3) -> DVec3,
        mass: f64,
        constraint: &impl HolonomicConstraint,
        solver: &BaumgarteSolver,
        dt: f64,
    ) -> (DVec3, DVec3) {
        let free_accel = free_accel_fn(pos);
        let a = apply_constraint(pos, vel, free_accel, mass, constraint, solver);
        let new_vel = vel + a * dt;
        let new_pos = pos + new_vel * dt;
        (new_pos, new_vel)
    }

    #[test]
    fn pendulum_constraint_preserves_length_over_integration() {
        // Pendulum of length 1 m, attached at origin, under gravity g = -9.81 m/s^2.
        // Start at angle theta_0 = 30° from downward (x axis).
        // Integrate for 10 periods using RK-ish symplectic Euler.
        // Length deviation should remain bounded by Baumgarte stabilization.
        let length = 1.0_f64;
        let g = 9.81_f64;
        let c = PendulumConstraint::new(DVec3::ZERO, length);
        // Small-angle period T ≈ 2π sqrt(L/g) ≈ 2.006 s. Use omega_B matched.
        let omega_b = (g / length).sqrt();
        let solver = BaumgarteSolver::critically_damped(10.0 * omega_b);

        // Initial: bob hangs 30° from straight down. "Down" = +x.
        // pos = L (cos(30°), sin(30°), 0) — but we want to measure length from
        // attachment; that's already what the constraint does.
        let theta0 = 30.0_f64.to_radians();
        let mut pos = DVec3::new(length * theta0.cos(), length * theta0.sin(), 0.0);
        let mut vel = DVec3::ZERO;

        let period = 2.0 * std::f64::consts::PI * (length / g).sqrt();
        let total_time = 10.0 * period;
        let dt = 1e-4;
        let n_steps = (total_time / dt).round() as usize;

        let mut max_length_err: f64 = 0.0;
        for _ in 0..n_steps {
            let (p, v) = step_constrained(
                pos,
                vel,
                |_p| DVec3::new(g, 0.0, 0.0), // "down" along +x
                1.0,
                &c,
                &solver,
                dt,
            );
            pos = p;
            vel = v;
            let len_err = (pos.length() - length).abs();
            if len_err > max_length_err {
                max_length_err = len_err;
            }
        }
        // Baumgarte should keep length error bounded.
        assert!(
            max_length_err < 5e-4,
            "max length error over 10 periods: {max_length_err} m"
        );
    }

    #[test]
    fn pendulum_small_angle_period_matches_analytical() {
        // Small angle: T = 2π sqrt(L/g). Integrate and measure zero-crossing
        // times to recover the period.
        let length = 1.0_f64;
        let g = 9.81_f64;
        let c = PendulumConstraint::new(DVec3::ZERO, length);
        let omega_n = (g / length).sqrt();
        // Use modest Baumgarte damping so stabilization doesn't distort the period
        let solver = BaumgarteSolver::critically_damped(omega_n);

        // Very small initial angle (1°)
        let theta0 = 1.0_f64.to_radians();
        let mut pos = DVec3::new(length * theta0.cos(), length * theta0.sin(), 0.0);
        let mut vel = DVec3::ZERO;

        let dt = 1e-4;
        let total_time = 5.0 * (2.0 * std::f64::consts::PI / omega_n);
        let n_steps = (total_time / dt).round() as usize;

        // Detect zero crossings of y-coordinate going from + to - (peak then descent)
        let mut last_y = pos.y;
        let mut crossing_times: Vec<f64> = Vec::new();
        for i in 1..n_steps {
            let (p, v) =
                step_constrained(pos, vel, |_p| DVec3::new(g, 0.0, 0.0), 1.0, &c, &solver, dt);
            pos = p;
            vel = v;
            // Zero crossing (linear interpolation)
            if last_y > 0.0 && pos.y <= 0.0 {
                let t1 = (i as f64) * dt;
                let t0 = t1 - dt;
                let t_cross = t0 + dt * last_y / (last_y - pos.y);
                crossing_times.push(t_cross);
            }
            last_y = pos.y;
        }

        // Need at least 3 crossings to estimate period (consecutive + to - crossings
        // are one full period apart).
        assert!(
            crossing_times.len() >= 3,
            "only got {} zero crossings",
            crossing_times.len()
        );
        let analytical_period = 2.0 * std::f64::consts::PI * (length / g).sqrt();
        // Average interval between successive + to - crossings = one period.
        let measured_period = (crossing_times.last().unwrap() - crossing_times.first().unwrap())
            / (crossing_times.len() as f64 - 1.0);
        let rel_err = (measured_period - analytical_period).abs() / analytical_period;
        assert!(
            rel_err < 0.01,
            "measured period {measured_period} vs analytical {analytical_period} (rel err {rel_err})"
        );
    }

    #[test]
    fn constraint_drift_bounded_with_baumgarte() {
        // Compare drift with and without Baumgarte stabilization. Without it,
        // numerical error accumulates linearly; with it, error stays bounded.
        let length = 1.0_f64;
        let g = 9.81_f64;
        let c = PendulumConstraint::new(DVec3::ZERO, length);

        let theta0 = 30.0_f64.to_radians();
        let initial_pos = DVec3::new(length * theta0.cos(), length * theta0.sin(), 0.0);

        let dt = 1e-3_f64;
        let total_time = 20.0_f64; // ~10 periods
        let n_steps = (total_time / dt).round() as usize;

        let run = |solver: BaumgarteSolver| -> f64 {
            let mut pos = initial_pos;
            let mut vel = DVec3::ZERO;
            let mut max_err: f64 = 0.0;
            for _ in 0..n_steps {
                let (p, v) =
                    step_constrained(pos, vel, |_p| DVec3::new(g, 0.0, 0.0), 1.0, &c, &solver, dt);
                pos = p;
                vel = v;
                let err = (pos.length() - length).abs();
                if err > max_err {
                    max_err = err;
                }
            }
            max_err
        };

        let no_stab = run(BaumgarteSolver {
            alpha: 0.0,
            beta: 0.0,
        });
        let with_stab = run(BaumgarteSolver::critically_damped(20.0));

        // Stabilized version should be dramatically better.
        assert!(
            with_stab < 0.1 * no_stab,
            "Baumgarte stabilization didn't help: no_stab={no_stab}, with_stab={with_stab}"
        );
        // And its absolute drift should stay tiny.
        assert!(with_stab < 1e-3, "Baumgarte drift too large: {with_stab}");
    }
}
