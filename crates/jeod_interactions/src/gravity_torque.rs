//! Gravity gradient torque computation.
//!
//! Port of JEOD `models/interactions/gravity_torque/src/gravity_torque.cc`.
//!
//! The gravity gradient torque arises from the differential gravitational
//! acceleration across an extended body. For a rigid body with inertia tensor
//! I in a gravity gradient field G (both in body frame), the torque is:
//!
//!   τ_i = ε_ijk G_jm I_mk
//!
//! This implementation uses JEOD's expanded form which handles the full
//! (potentially non-diagonal) inertia tensor.

use glam::{DMat3, DVec3};

/// Compute gravity gradient torque on a rigid body.
///
/// Faithfully ports JEOD `gravity_torque.cc` lines 81-100.
///
/// # Arguments
/// * `grav_grad` - Gravity gradient tensor (∂²Φ/∂x_i∂x_j) in the
///   inertial frame. Symmetric 3x3, trace ≈ 0 outside attracting body.
///   From `GravityInteraction::grav_grad` (JEOD name).
/// * `t_parent_this` - Rotation matrix from inertial (parent) frame to body
///   (composite_body) frame. From `RotationalState` quaternion → matrix.
///   Matches JEOD `RefFrameRot::T_parent_this`.
/// * `inertia` - Inertia tensor in the body frame (kg·m²). From
///   `MassProperties::inertia`.
///
/// # Returns
/// Torque vector in the body frame (N·m).
///
/// # Notes
/// - Returns the torque in the **body frame** (composite_body), not the
///   structure frame. JEOD's `gravity_torque.cc` has a final transform to
///   structure, but our integrator expects torque in body frame.
/// - For a spherically symmetric body (I = c·identity), the torque is zero.
/// - For a diagonal inertia with the body tilted by angle θ from nadir,
///   the dominant torque component matches the analytical formula:
///   τ = 3μ/(2r³) · sin(2θ) · ΔI
pub fn compute_gravity_torque(grav_grad: &DMat3, t_parent_this: &DMat3, inertia: &DMat3) -> DVec3 {
    // Transform gradient from inertial to body frame:
    //   g = T * grad * T^T  (similarity transform)
    let g = *t_parent_this * *grav_grad * t_parent_this.transpose();

    // Extract matrix elements: g.col(j)[i] = g[row=i][col=j]
    let g00 = g.col(0)[0];
    let g01 = g.col(1)[0];
    let g02 = g.col(2)[0];
    let g11 = g.col(1)[1];
    let g12 = g.col(2)[1];
    let g22 = g.col(2)[2];

    let i00 = inertia.col(0)[0];
    let i01 = inertia.col(1)[0];
    let i02 = inertia.col(2)[0];
    let i11 = inertia.col(1)[1];
    let i12 = inertia.col(2)[1];
    let i22 = inertia.col(2)[2];

    // JEOD gravity_torque.cc lines 93-100 (torque in body frame)
    let tx = g12 * (i22 - i11) - g02 * i01 + g01 * i02 - i12 * (g22 - g11);
    let ty = g02 * (i00 - i22) + g12 * i01 - g01 * i12 - i02 * (g00 - g22);
    let tz = g01 * (i11 - i00) - g12 * i02 + g02 * i12 - i01 * (g11 - g00);

    DVec3::new(tx, ty, tz)
}

/// Typed sibling of [`compute_gravity_torque`].
///
/// Accepts a typed [`InertiaTensor<BodyFrame<V>>`] from Phase 0 and
/// returns [`Torque<BodyFrame<V>>`]. The gradient tensor and
/// inertial→body rotation stay as raw `DMat3`: gradients don't have
/// a typed alias in `jeod_quantities` (a 1/s² 3×3 tensor evaluated
/// within a single frame), and the rotation matrix comes from
/// upstream untyped storage. Numeric kernel is the existing untyped
/// function — bit-identical output for equal numeric inputs.
pub fn compute_gravity_torque_typed<V: jeod_quantities::frame::Vehicle>(
    grav_grad: &DMat3,
    t_parent_this: &DMat3,
    inertia: jeod_quantities::aliases::InertiaTensor<jeod_quantities::frame::BodyFrame<V>>,
) -> jeod_quantities::aliases::Torque<jeod_quantities::frame::BodyFrame<V>> {
    let inertia_dmat = inertia.as_dmat3();
    let untyped_torque = compute_gravity_torque(grav_grad, t_parent_this, &inertia_dmat);
    jeod_quantities::aliases::Torque::<jeod_quantities::frame::BodyFrame<V>>::from_raw_si(
        untyped_torque,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Spherically symmetric body: torque must be zero regardless of gradient.
    #[test]
    fn spherical_body_zero_torque() {
        let inertia = DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0));

        // Point-mass gradient at 7000 km, nadir along +X in inertial
        let mu = 3.986_004_415e14;
        let r = 7_000_000.0;
        let r3 = r * r * r;
        // Gradient for point mass with nadir along X:
        // G = (mu/r^3) * diag(−2, 1, 1)  (trace = 0)
        let grad = DMat3::from_diagonal(DVec3::new(-2.0, 1.0, 1.0)) * (mu / r3);

        // Arbitrary rotation (45 degrees about Z)
        let angle = PI / 4.0;
        let c = angle.cos();
        let s = angle.sin();
        let t = DMat3::from_cols(
            DVec3::new(c, s, 0.0),
            DVec3::new(-s, c, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );

        let torque = compute_gravity_torque(&grad, &t, &inertia);
        assert!(
            torque.length() < 1e-20,
            "Torque on spherical body should be zero, got {:?} (magnitude {})",
            torque,
            torque.length()
        );
    }

    /// Axisymmetric body tilted by θ from nadir: torque matches analytical.
    ///
    /// For a diagonal inertia I = diag(A, B, C) with the body tilted by θ
    /// about the Y axis from nadir (nadir along Z in inertial), the gravity
    /// gradient torque about the Y axis is:
    ///   τ_y = 3μ/(2r³) · sin(2θ) · (A − C)
    #[test]
    fn analytical_gravity_gradient_torque() {
        let mu = 3.986_004_415e14;
        let r = 7_000_000.0;
        let r3 = r * r * r;

        // Inertia: I = diag(10, 20, 30) kg·m²
        let ia = 10.0;
        let ic = 30.0;
        let inertia = DMat3::from_diagonal(DVec3::new(ia, 20.0, ic));

        // Gradient for point mass with nadir along Z:
        // G = (mu/r^3) * diag(1, 1, -2)
        let grad = DMat3::from_diagonal(DVec3::new(1.0, 1.0, -2.0)) * (mu / r3);

        // Body tilted by θ = 30° about the Y axis
        let theta = 30.0_f64.to_radians();
        let c = theta.cos();
        let s = theta.sin();
        // Rotation about Y (inertial -> body): rotates nadir into body frame
        let t = DMat3::from_cols(
            DVec3::new(c, 0.0, -s),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(s, 0.0, c),
        );

        let torque = compute_gravity_torque(&grad, &t, &inertia);

        // Analytical: τ_y = 3μ/(2r³) · sin(2θ) · (C - A)
        // Derivation: g[0][2] = -3·sin(θ)·cos(θ)·μ/r³ from similarity transform,
        // torque[1] = g[0][2]·(A-C) = 3·sin(θ)·cos(θ)·μ/r³·(C-A)
        let expected_ty = 3.0 * mu / (2.0 * r3) * (2.0 * theta).sin() * (ic - ia);

        // X and Z torque should be zero (tilt is purely about Y, diagonal inertia)
        assert!(
            torque.x.abs() < 1e-15,
            "X torque should be ~0, got {}",
            torque.x
        );
        assert!(
            torque.z.abs() < 1e-15,
            "Z torque should be ~0, got {}",
            torque.z
        );

        let rel_err = (torque.y - expected_ty).abs() / expected_ty.abs();
        assert!(
            rel_err < 1e-10,
            "Y torque: expected {expected_ty}, got {}, relative error {rel_err}",
            torque.y
        );
    }

    /// Non-diagonal inertia tensor: verify formula works for general case.
    #[test]
    fn non_diagonal_inertia() {
        let mu = 3.986_004_415e14;
        let r = 7_000_000.0;
        let r3 = r * r * r;

        // Non-diagonal inertia (products of inertia are nonzero)
        let inertia = DMat3::from_cols(
            DVec3::new(100.0, -5.0, 3.0),
            DVec3::new(-5.0, 200.0, -7.0),
            DVec3::new(3.0, -7.0, 150.0),
        );

        // Gradient for point mass along X
        let grad = DMat3::from_diagonal(DVec3::new(-2.0, 1.0, 1.0)) * (mu / r3);

        // Identity rotation (body aligned with inertial)
        let t = DMat3::IDENTITY;

        let torque = compute_gravity_torque(&grad, &t, &inertia);

        // With identity rotation and gradient along X:
        // g = grad (identity transform)
        // g01 = 0, g02 = 0, g12 = 0 (diagonal gradient)
        // g00 = -2μ/r³, g11 = μ/r³, g22 = μ/r³
        //
        // torque[0] = 0*(i22-i11) - 0*i01 + 0*i02 - i12*(g22-g11) = -(-7)*(0) = 0
        // torque[1] = 0*(i00-i22) + 0*i01 - 0*i12 - i02*(g00-g22) = -3*(-2μ/r³-μ/r³) = 3*(3μ/r³)
        // torque[2] = 0*(i11-i00) - 0*i02 + 0*i12 - i01*(g11-g00) = -(-5)*(μ/r³-(-2μ/r³)) = 5*(3μ/r³)

        let scale = mu / r3;
        let expected_tx = -(-7.0) * (scale - scale); // g22-g11 = 0
        let expected_ty = -3.0 * (-2.0 * scale - scale); // i02*(g00-g22) = 3*(-3*scale)
        let expected_tz = -(-5.0) * (scale - (-2.0 * scale)); // i01*(g11-g00) = -(-5)*(3*scale)

        assert!(
            (torque.x - expected_tx).abs() < 1e-10,
            "X torque: expected {expected_tx}, got {}",
            torque.x
        );
        assert!(
            (torque.y - expected_ty).abs() / expected_ty.abs() < 1e-10,
            "Y torque: expected {expected_ty}, got {}",
            torque.y
        );
        assert!(
            (torque.z - expected_tz).abs() / expected_tz.abs() < 1e-10,
            "Z torque: expected {expected_tz}, got {}",
            torque.z
        );
    }

    /// Zero gradient tensor: no torque regardless of inertia.
    #[test]
    fn zero_gradient_zero_torque() {
        let inertia = DMat3::from_diagonal(DVec3::new(10.0, 20.0, 30.0));
        let grad = DMat3::ZERO;
        let t = DMat3::IDENTITY;

        let torque = compute_gravity_torque(&grad, &t, &inertia);
        assert_eq!(torque, DVec3::ZERO);
    }

    /// Torque magnitude increases with inertia asymmetry.
    #[test]
    fn torque_scales_with_inertia_difference() {
        let mu = 3.986_004_415e14;
        let r = 7_000_000.0;
        let r3 = r * r * r;
        let grad = DMat3::from_diagonal(DVec3::new(1.0, 1.0, -2.0)) * (mu / r3);

        let theta: f64 = 0.5; // 0.5 rad tilt about Y
        let c = theta.cos();
        let s = theta.sin();
        let t = DMat3::from_cols(
            DVec3::new(c, 0.0, -s),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(s, 0.0, c),
        );

        // Small asymmetry
        let inertia_small = DMat3::from_diagonal(DVec3::new(100.0, 100.0, 101.0));
        let torque_small = compute_gravity_torque(&grad, &t, &inertia_small);

        // Large asymmetry
        let inertia_large = DMat3::from_diagonal(DVec3::new(100.0, 100.0, 200.0));
        let torque_large = compute_gravity_torque(&grad, &t, &inertia_large);

        assert!(
            torque_large.length() > torque_small.length(),
            "Torque should increase with inertia asymmetry"
        );
    }
}
