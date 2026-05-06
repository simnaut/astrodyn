// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Per-step state-error metrics used by archetype-B Tier 3 tests.
//!
//! Hoisted in Phase 8 of #101 from
//! `crates/jeod_runner/tests/sim_test_helpers/mod.rs` (verbatim — only
//! the location changes).

use glam::{DMat3, DQuat};

use crate::TranslationalState;
use jeod_math::{JeodQuat, OrbitalElements};

/// Smallest rotation angle (radians) between two scalar-first JEOD
/// quaternions — `2 * acos(|<q1, q2>|)`.
pub fn jeodquat_angle_error(q1: &JeodQuat, q2: &JeodQuat) -> f64 {
    let dot = (q1.scalar() * q2.scalar()
        + q1.vector().x * q2.vector().x
        + q1.vector().y * q2.vector().y
        + q1.vector().z * q2.vector().z)
        .abs();
    2.0 * dot.min(1.0).acos()
}

/// Smallest rotation angle (radians) between two `glam::DQuat` values.
pub fn dquat_angle_error(a: DQuat, b: DQuat) -> f64 {
    let dot = a.dot(b).abs();
    2.0 * dot.min(1.0).acos()
}

/// Angular difference accounting for 2π wraparound. Returns the
/// magnitude (≥ 0) in `[0, π]`.
pub fn angle_diff(a: f64, b: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    let mut d = (a - b) % tau;
    if d > std::f64::consts::PI {
        d -= tau;
    }
    if d < -std::f64::consts::PI {
        d += tau;
    }
    d.abs()
}

/// Angular difference for angles known to lie within `[-max_magnitude, max_magnitude]`,
/// where wrap-around at 2π cannot occur. Asserts the bound on both inputs (in
/// release builds too) and returns plain `(a - b).abs()`.
///
/// Use this when a metric's domain is guaranteed bounded (e.g. solar beta is
/// constrained to `[-π/2, π/2]` in JEOD). It documents the bound, catches a
/// violated assumption in any build, and avoids the temptation to "simplify"
/// to `.abs()` somewhere else and lose the wrap-around invariant.
pub fn angle_diff_restricted(a: f64, b: f64, max_magnitude: f64) -> f64 {
    assert!(
        a.abs() <= max_magnitude && b.abs() <= max_magnitude,
        "angle_diff_restricted: input outside [-{max_magnitude}, {max_magnitude}] (a={a}, b={b})"
    );
    (a - b).abs()
}

/// Max absolute element-wise difference between two 3×3 matrices.
pub fn max_mat_diff(a: &DMat3, b: &DMat3) -> f64 {
    let mut max_d = 0.0_f64;
    for c in 0..3 {
        for r in 0..3 {
            let d = (a.col(c)[r] - b.col(c)[r]).abs();
            max_d = max_d.max(d);
        }
    }
    max_d
}

/// Initialize a [`TranslationalState`] from classical orbital elements.
/// Works for all eccentricities including hyperbolic (e ≥ 1).
///
/// Mirrors `OrbitalElements::to_cartesian` but accepts loose scalar
/// inputs for the parametric orbinit families that test edge cases
/// across many eccentricities.
pub fn state_from_elements(
    a: f64,
    e: f64,
    i: f64,
    raan: f64,
    argp: f64,
    nu: f64,
    mu: f64,
) -> TranslationalState {
    let mut oe = OrbitalElements::<jeod_quantities::frame::SelfPlanet>::default();
    oe.semi_major_axis = a;
    oe.e_mag = e;
    oe.inclination = i;
    oe.long_asc_node = raan;
    oe.arg_periapsis = argp;
    oe.true_anom = nu;

    if e < 1.0 {
        oe.semiparam = a * (1.0 - e * e);
    } else {
        // Hyperbolic: a is negative, p = a (1 - e²) = |a| (e² - 1).
        oe.semiparam = a.abs() * (e * e - 1.0);
    }
    oe.nu_to_anomalies();

    let (position, velocity) = oe.to_cartesian(mu).expect("to_cartesian failed");
    TranslationalState { position, velocity }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dquat_angle_identity_zero() {
        assert!(dquat_angle_error(DQuat::IDENTITY, DQuat::IDENTITY).abs() < 1e-15);
    }

    #[test]
    fn jeodquat_angle_identity_zero() {
        let q = JeodQuat::identity();
        assert!(jeodquat_angle_error(&q, &q).abs() < 1e-15);
    }

    #[test]
    fn angle_diff_wraparound() {
        let tau = 2.0 * std::f64::consts::PI;
        assert!(angle_diff(0.1, tau - 0.1) < 0.21);
        assert!(angle_diff(0.0, 0.0).abs() < 1e-15);
    }

    #[test]
    fn angle_diff_restricted_identity_zero() {
        let half_pi = std::f64::consts::FRAC_PI_2;
        assert!(angle_diff_restricted(0.0, 0.0, half_pi).abs() < 1e-15);
    }

    #[test]
    fn angle_diff_restricted_full_range() {
        let half_pi = std::f64::consts::FRAC_PI_2;
        let d = angle_diff_restricted(half_pi, -half_pi, half_pi);
        assert!((d - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "angle_diff_restricted")]
    fn angle_diff_restricted_bound_violation_panics() {
        let half_pi = std::f64::consts::FRAC_PI_2;
        let _ = angle_diff_restricted(half_pi + 0.1, 0.0, half_pi);
    }

    #[test]
    fn max_mat_diff_zero_for_identical() {
        assert!(max_mat_diff(&DMat3::IDENTITY, &DMat3::IDENTITY).abs() < 1e-15);
    }

    #[test]
    fn max_mat_diff_picks_largest_element() {
        let mut b = DMat3::IDENTITY;
        // Mutate the (1,2) element via column 2 of the matrix.
        let mut col2 = b.col(2);
        col2.y = 1.5; // off-diagonal element 1,2
        b = DMat3::from_cols(b.col(0), b.col(1), col2);
        let d = max_mat_diff(&DMat3::IDENTITY, &b);
        assert!((d - 1.5).abs() < 1e-15);
    }

    #[test]
    fn state_from_elements_circular() {
        let mu = 3.986_004_415e14;
        let a = 7_000_000.0;
        let s = state_from_elements(a, 0.0, 0.0, 0.0, 0.0, 0.0, mu);
        // Circular: |r| = a, |v| = sqrt(mu/a).
        assert!((s.position.length() - a).abs() < 1.0);
        assert!((s.velocity.length() - (mu / a).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn state_from_elements_hyperbolic() {
        let mu = 1.327e20;
        let a = -1.0e12; // a < 0 for hyperbolic
        let e = 1.5;
        // Just check it produces something finite without panicking.
        let s = state_from_elements(a, e, 0.0, 0.0, 0.0, 0.0, mu);
        for v in [
            s.position.x,
            s.position.y,
            s.position.z,
            s.velocity.x,
            s.velocity.y,
            s.velocity.z,
        ] {
            assert!(v.is_finite());
        }
    }
}
