//! Top-level gravity-acceleration computation.
//!
//! The point-mass kernel and the multi-source dispatcher that selects
//! between [`calc_spherical`] and the spherical-harmonics path in
//! [`crate::spherical_harmonics_calc_nonspherical`] depending on the
//! [`crate::GravityModel`] variant.
//!
//! Ports
//! [`models/environment/gravity/src/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/gravity/src/)
//! from JEOD v5.4.0.

use astrodyn_dynamics::GravityAcceleration;
use astrodyn_math::{
    matrix3x3_transpose_transform_matrix, vector3_transform, vector3_transform_transpose,
};
use glam::{DMat3, DVec3};

use crate::gravity_source::{GravityModel, GravitySource};

/// Decomposed output of the gravity kernel.
///
/// Splitting the contribution into a planet-fixed spherical-harmonics
/// piece and an inertial point-mass piece moves the inverse rotation
/// (`t_parent_this^T`) out of the kernel boundary and into a separate
/// step the caller controls via [`Self::into_inertial`]. This buys two
/// concrete things:
///
/// 1. **Gradient-skip when the caller doesn't want it.** The kernel
///    previously applied `matrix3x3_transpose_transform_matrix` on
///    every call, regardless of whether the gradient was consumed.
///    [`Self::into_inertial`] now gates that 9-mul/9-add transform on
///    a `compute_gradient` flag, so the RK4 inner-loop call site
///    (`evaluate_accel_only`) pays only for the accel transform.
/// 2. **Point-mass short-circuit.** Point-mass sources leave
///    `sh_pfix` as `None`, and [`Self::into_inertial`] then skips the
///    accel and gradient transforms entirely. The pre-hoist kernel
///    already returned point-mass in inertial coordinates, but the
///    branch shape is clearer at the call site now.
///
/// Per-substage rotation is still applied: `position` changes between
/// RK4 substages, so the resulting inertial-frame `acc` and `grad`
/// differ per call and cannot be hoisted across substages. The
/// decomposition surfaces the planet-fixed pieces in case a future
/// caller wants to combine multiple kernel outputs in pfix before
/// applying one shared rotation; no production caller does this today.
///
/// Bit-identity: the f64 operations at the kernel boundary are the
/// same — only the *site* of the `vector3_transform_transpose` and
/// `matrix3x3_transpose_transform_matrix` calls moved. The point-mass
/// piece is returned in inertial coordinates unchanged.
#[derive(Debug, Clone, Default)]
pub struct GravityKernelOutput {
    /// Spherical-harmonics contribution in the planet-fixed frame
    /// (`acc_pf`, `grad_pf`, `pot`). `None` for point-mass sources or
    /// when the SH path is bypassed (e.g. effective degree 0).
    pub sh_pfix: Option<GravityAcceleration>,
    /// Point-mass contribution already in the inertial frame. `None`
    /// when `perturbing_only` is true (which skips point-mass) or when
    /// the SH path was selected but produced no point-mass piece.
    pub pm_inertial: Option<GravityAcceleration>,
}

impl GravityKernelOutput {
    /// Apply the inverse rotation to the SH piece (if any) and combine
    /// with the point-mass piece (if any) to produce a single
    /// inertial-frame [`GravityAcceleration`].
    ///
    /// This is the inverse of the decomposition the kernel produces:
    /// callers that don't want to hoist the inverse rotation themselves
    /// can call this to recover the pre-hoist combined form. The
    /// `compute_gradient` flag mirrors the kernel's gradient gate so
    /// the caller doesn't pay for the matrix transform when the
    /// gradient is unused.
    #[inline]
    pub fn into_inertial(
        self,
        t_parent_this: &DMat3,
        compute_gradient: bool,
    ) -> GravityAcceleration {
        let mut out = self.pm_inertial.unwrap_or_default();
        if let Some(sh) = self.sh_pfix {
            // Vector3::transform_transpose(T_parent_this, sh.grav_accel)
            out.grav_accel += vector3_transform_transpose(t_parent_this, sh.grav_accel);
            if compute_gradient {
                // Matrix3x3::transpose_transform_matrix(T_parent_this, sh.grav_grad)
                out.grav_grad += matrix3x3_transpose_transform_matrix(t_parent_this, &sh.grav_grad);
            }
            out.grav_pot += sh.grav_pot;
        }
        out
    }
}

/// Compute point-mass gravitational acceleration, gradient, and potential.
/// Position is the body's position relative to the gravity source center.
pub fn calc_spherical(mu: f64, position: DVec3) -> GravityAcceleration {
    let r_sq = position.length_squared();
    assert!(
        r_sq > 0.0,
        "position must be non-zero (body is at gravity source center)"
    );
    let r_mag = r_sq.sqrt();
    let r_3rd = r_sq * r_mag;

    // Acceleration: a = -mu/r^3 * r
    let grav_accel = position * (-mu / r_3rd);

    // Potential: V = +mu/r (JEOD convention, positive for bound orbits)
    let grav_pot = mu / r_mag;

    // Gradient tensor: G[i][j] = mu/r^5 * (3*r[i]*r[j] - delta_ij*r^2)
    let r_5th = r_3rd * r_sq;
    let mu_r5 = mu / r_5th;
    // Build as columns for glam's column-major DMat3
    let grav_grad = DMat3::from_cols(
        DVec3::new(
            mu_r5 * (3.0 * position.x * position.x - r_sq),
            mu_r5 * 3.0 * position.y * position.x,
            mu_r5 * 3.0 * position.z * position.x,
        ),
        DVec3::new(
            mu_r5 * 3.0 * position.x * position.y,
            mu_r5 * (3.0 * position.y * position.y - r_sq),
            mu_r5 * 3.0 * position.z * position.y,
        ),
        DVec3::new(
            mu_r5 * 3.0 * position.x * position.z,
            mu_r5 * 3.0 * position.y * position.z,
            mu_r5 * (3.0 * position.z * position.z - r_sq),
        ),
    );

    GravityAcceleration {
        grav_accel,
        grav_grad,
        grav_pot,
    }
}

/// Dispatch gravity computation based on model type.
///
/// Returns the spherical-harmonics piece in the planet-fixed frame and
/// the point-mass piece in the inertial frame, decomposed in a
/// [`GravityKernelOutput`]. Callers apply the inverse rotation via
/// [`GravityKernelOutput::into_inertial`] (or hoist it across multiple
/// kernel calls that share the same `t_parent_this`).
///
/// For `PointMass`, `position` is relative to the source center in any frame;
/// `t_parent_this`, truncation, and gradient parameters are ignored.
///
/// For `SphericalHarmonics`, `position` is in inertial coordinates and
/// `t_parent_this` is the inertial-to-planet-fixed rotation matrix.
/// Matching JEOD's `GravityControls::gravitation`, the kernel internally
/// transforms position to planet-fixed and runs the SH recurrence
/// there. The inverse transform on the SH output is hoisted to the
/// caller (see [`GravityKernelOutput::into_inertial`]), which lets the
/// caller gate the gradient transform on whether the gradient is
/// consumed and skip both transforms entirely when the kernel produced
/// no SH piece (point-mass).
///
/// When `perturbing_only` is false, both pieces are populated (the
/// caller combines them into point-mass + nonspherical). When true,
/// only the nonspherical perturbation is returned (matching JEOD's
/// `perturbation_only` mode for third-body differential acceleration).
///
/// `degree`/`order` truncate the harmonics evaluation.
///
/// Matching JEOD's `check_validity()`, callers must ensure:
/// - `degree <= data.degree`, `order <= data.order`, `order <= degree`
///
/// This is a convenience wrapper that allocates scratch internally.
/// For the RK4 inner loop, prefer [`gravitation_with_scratch`].
#[allow(clippy::too_many_arguments)]
pub fn gravitation(
    source: &GravitySource,
    position: DVec3,
    t_parent_this: &DMat3,
    degree: usize,
    order: usize,
    perturbing_only: bool,
    compute_gradient: bool,
    gradient_degree: usize,
    gradient_order: usize,
    delta_c20: f64,
    has_delta_coeffs: bool,
) -> GravityKernelOutput {
    match &source.model {
        GravityModel::PointMass => {
            if perturbing_only {
                GravityKernelOutput::default()
            } else {
                GravityKernelOutput {
                    sh_pfix: None,
                    pm_inertial: Some(calc_spherical(source.mu, position)),
                }
            }
        }
        GravityModel::SphericalHarmonics(_) => {
            // Reuse a thread-local scratch buffer across calls. Allocating
            // a fresh `GottliebScratch::new(degree)` here was ~79 % of
            // self-time on the Earth-Moon profile (LP150Q 60×60, RK4 4×
            // per step × 3 sources × 100k steps ≈ 1.2 M allocations).
            // The buffer grows monotonically: a request for a higher
            // degree than previously seen reallocates once and the
            // larger buffer covers all subsequent lower-degree calls.
            // Bit-identical to the per-call `::new` path because the
            // kernel overwrites the scratch on entry — no leftover
            // state ever participates in the math.
            crate::spherical_harmonics_calc_nonspherical::with_scratch(degree, |scratch| {
                gravitation_with_scratch(
                    source,
                    position,
                    t_parent_this,
                    degree,
                    order,
                    perturbing_only,
                    compute_gradient,
                    gradient_degree,
                    gradient_order,
                    scratch,
                    delta_c20,
                    has_delta_coeffs,
                )
            })
        }
    }
}

/// Dispatch gravity computation with a reusable scratch workspace.
///
/// Same as [`gravitation`] but avoids heap allocation by reusing
/// pre-allocated buffers.
///
/// For `PointMass`, the scratch buffer is unused. For `SphericalHarmonics`,
/// `scratch` must have been created with `degree >=` the requested degree.
#[allow(clippy::too_many_arguments)]
pub fn gravitation_with_scratch(
    source: &GravitySource,
    position: DVec3,
    t_parent_this: &DMat3,
    degree: usize,
    order: usize,
    perturbing_only: bool,
    compute_gradient: bool,
    gradient_degree: usize,
    gradient_order: usize,
    scratch: &mut crate::spherical_harmonics_calc_nonspherical::GottliebScratch,
    delta_c20: f64,
    has_delta_coeffs: bool,
) -> GravityKernelOutput {
    match &source.model {
        GravityModel::PointMass => {
            if perturbing_only {
                GravityKernelOutput::default()
            } else {
                GravityKernelOutput {
                    sh_pfix: None,
                    pm_inertial: Some(calc_spherical(source.mu, position)),
                }
            }
        }
        GravityModel::SphericalHarmonics(data) => {
            // Configuration invariant: source.mu must match data.mu. Validated at
            // construction time; debug_assert avoids per-call overhead in the RK4
            // inner loop (called 4x per body per timestep).
            debug_assert!(
                (source.mu - data.mu).abs() < 1e-10,
                "GravitySource.mu ({}) must match SphericalHarmonicsData.mu ({})",
                source.mu,
                data.mu
            );

            // Vector3::transform(T_parent_this, posn, posn_pf)
            let posn_pf = vector3_transform(t_parent_this, position);

            let sh_pf =
                crate::spherical_harmonics_calc_nonspherical::calc_nonspherical_with_scratch(
                    data,
                    posn_pf,
                    degree,
                    order,
                    compute_gradient,
                    gradient_degree,
                    gradient_order,
                    scratch,
                    delta_c20,
                    has_delta_coeffs,
                );

            // SH stays in planet-fixed; the caller applies the inverse
            // rotation via `GravityKernelOutput::into_inertial`, where
            // the gradient transform is gated on `compute_gradient`.
            // Same f64 ops as the pre-hoist in-kernel form —
            // `vector3_transform_transpose` and
            // `matrix3x3_transpose_transform_matrix` just moved one
            // frame up the stack.
            let pm_inertial = if perturbing_only {
                None
            } else {
                Some(calc_spherical(source.mu, position))
            };

            GravityKernelOutput {
                sh_pfix: Some(sh_pf),
                pm_inertial,
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "typed-vs-raw parity tests assert bit-exact identity at the type boundary"
)]
mod tests {
    use super::*;

    const EARTH_MU: f64 = 3.986_004_415e14; // m^3/s^2
    const EARTH_RADIUS: f64 = 6_378_137.0; // m

    #[test]
    fn earth_surface_gravity_magnitude() {
        // At Earth's surface along x-axis
        let pos = DVec3::new(EARTH_RADIUS, 0.0, 0.0);
        let result = calc_spherical(EARTH_MU, pos);
        let accel_mag = result.grav_accel.length();
        // Should be approximately 9.80 m/s^2
        assert!(
            (accel_mag - 9.80).abs() < 0.1,
            "Earth surface gravity: expected ~9.80 m/s^2, got {} m/s^2",
            accel_mag
        );
    }

    #[test]
    fn inverse_square_law() {
        let mu = 1e14;
        let pos1 = DVec3::new(1e7, 0.0, 0.0);
        let pos2 = DVec3::new(2e7, 0.0, 0.0); // double distance

        let result1 = calc_spherical(mu, pos1);
        let result2 = calc_spherical(mu, pos2);

        let a1 = result1.grav_accel.length();
        let a2 = result2.grav_accel.length();

        // Double distance -> accel reduces by factor 4
        let ratio = a1 / a2;
        assert!(
            (ratio - 4.0).abs() < 1e-14,
            "Inverse-square ratio: expected 4.0, got {}",
            ratio
        );
    }

    #[test]
    fn gradient_symmetry() {
        // Test with an off-axis position so all gradient components are non-zero
        let mu = 3.986e14;
        let pos = DVec3::new(4e6, 3e6, 5e6);
        let result = calc_spherical(mu, pos);
        let g = result.grav_grad;

        // G[i][j] == G[j][i]
        // glam: col(j)[i] is element (i,j) in row-col notation
        let tol = 1e-10;
        assert!(
            (g.col(1)[0] - g.col(0)[1]).abs() < tol,
            "G[0][1]={} != G[1][0]={}",
            g.col(1)[0],
            g.col(0)[1]
        );
        assert!(
            (g.col(2)[0] - g.col(0)[2]).abs() < tol,
            "G[0][2]={} != G[2][0]={}",
            g.col(2)[0],
            g.col(0)[2]
        );
        assert!(
            (g.col(2)[1] - g.col(1)[2]).abs() < tol,
            "G[1][2]={} != G[2][1]={}",
            g.col(2)[1],
            g.col(1)[2]
        );
    }

    #[test]
    fn gradient_trace_is_zero() {
        // Laplace's equation: trace of gravity gradient = 0 outside the body
        let mu = 3.986e14;
        let pos = DVec3::new(4e6, 3e6, 5e6);
        let result = calc_spherical(mu, pos);
        let g = result.grav_grad;

        // trace = G[0][0] + G[1][1] + G[2][2]
        let trace = g.col(0)[0] + g.col(1)[1] + g.col(2)[2];
        assert!(
            trace.abs() < 1e-10,
            "Gradient trace should be ~0 (Laplace), got {}",
            trace
        );
    }

    #[test]
    fn potential_at_known_distance() {
        let mu = 3.986e14;
        let r = 7e6;
        let pos = DVec3::new(r, 0.0, 0.0);
        let result = calc_spherical(mu, pos);

        let expected_potential = mu / r;
        assert!(
            (result.grav_pot - expected_potential).abs() < 1e-6,
            "Potential: expected {}, got {}",
            expected_potential,
            result.grav_pot
        );
    }

    #[test]
    fn acceleration_direction_opposite_to_position() {
        let mu = 1e14;
        let pos = DVec3::new(3e6, 4e6, 5e6);
        let result = calc_spherical(mu, pos);

        // Acceleration should point opposite to position (toward the source)
        let accel_dir = result.grav_accel.normalize();
        let pos_dir = pos.normalize();

        // accel_dir should be -pos_dir
        let sum = accel_dir + pos_dir;
        assert!(
            sum.length() < 1e-14,
            "Acceleration should be opposite to position: accel_dir={:?}, pos_dir={:?}",
            accel_dir,
            pos_dir
        );
    }

    #[test]
    fn gravitation_dispatches_point_mass() {
        let source = GravitySource {
            mu: EARTH_MU,
            model: GravityModel::PointMass,
        };
        let pos = DVec3::new(EARTH_RADIUS, 0.0, 0.0);
        let kernel_out = gravitation(
            &source,
            pos,
            &DMat3::IDENTITY,
            0,
            0,
            false,
            false,
            0,
            0,
            0.0,
            false,
        );
        let direct = calc_spherical(EARTH_MU, pos);

        // Point-mass returns only the inertial piece — no SH, no
        // rotation needed.
        assert!(kernel_out.sh_pfix.is_none());
        let pm = kernel_out.pm_inertial.expect("point-mass piece");
        assert_eq!(pm.grav_accel, direct.grav_accel);
        assert_eq!(pm.grav_pot, direct.grav_pot);
        assert_eq!(pm.grav_grad, direct.grav_grad);
    }

    #[test]
    fn earth_surface_gravity_off_axis() {
        // Test at a 45-degree position to verify off-axis consistency
        let r = EARTH_RADIUS;
        let component = r / 3.0_f64.sqrt();
        let pos = DVec3::new(component, component, component);
        let result = calc_spherical(EARTH_MU, pos);
        let accel_mag = result.grav_accel.length();
        assert!(
            (accel_mag - 9.80).abs() < 0.1,
            "Off-axis Earth surface gravity: expected ~9.80, got {}",
            accel_mag
        );
    }

    #[test]
    fn gradient_at_earth_surface() {
        // The gradient magnitude at Earth surface should be on the order of
        // mu/r^3 ~ 3.986e14 / (6.378e6)^3 ~ 1.5e-6 /s^2
        let pos = DVec3::new(EARTH_RADIUS, 0.0, 0.0);
        let result = calc_spherical(EARTH_MU, pos);
        let g = result.grav_grad;

        // For position along x-axis: G[0][0] = mu/r^5 * (3*x^2 - r^2) = mu/r^5 * 2*r^2 = 2*mu/r^3
        let expected_g00 = 2.0 * EARTH_MU / (EARTH_RADIUS * EARTH_RADIUS * EARTH_RADIUS);
        assert!(
            (g.col(0)[0] - expected_g00).abs() / expected_g00.abs() < 1e-10,
            "G[0][0]: expected {}, got {}",
            expected_g00,
            g.col(0)[0]
        );

        // Off-diagonal should be zero for on-axis position
        assert!(
            g.col(1)[0].abs() < 1e-20,
            "G[0][1] should be 0 for on-axis, got {}",
            g.col(1)[0]
        );
    }
}
