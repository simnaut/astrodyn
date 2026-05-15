//! JEOD quaternion glue: re-exports the unified
//! [`astrodyn_quantities::JeodQuat`] type and keeps JEOD-specific constants and
//! free helpers that don't belong in the generic quantities crate.
//!
//! All algebraic and conversion methods (`identity`, `new`, `scalar`,
//! `vector`, `conjugate`, `multiply`, `normalize`, glam round-trips,
//! axis-angle construction, left-transform ↔ matrix conversions, Rodrigues
//! vector transform) are inherent methods on
//! [`Quat<ScalarFirst, LeftTransform>`] defined in
//! `astrodyn_quantities::quat`. This module is intentionally small — it exists
//! so downstream crates can continue to say `astrodyn_math::JeodQuat` and
//! `astrodyn_math::quaternion::NORM_LIMIT` after the Phase 2 unification.

pub use astrodyn_quantities::quat::NORM_LIMIT;
pub use astrodyn_quantities::{
    JeodQuat, Layout, LeftTransform, NormalizedQuat, Quat, RightTransform, ScalarFirst, ScalarLast,
    Transform,
};

// ====================================================================
// Tests
// ====================================================================
//
// These tests exercise the unified methods now defined in
// `astrodyn_quantities::quat`. They stay here (rather than in astrodyn_quantities)
// because they depend on `astrodyn_math::test_utils::approx_*` helpers and on
// JEOD-parity edge cases that are owned by this crate.

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "quaternion identity / round-trip tests assert bit-exact recovery of literal components"
)]
mod tests {
    use super::*;
    use crate::test_utils::{approx_eq_f64, approx_eq_mat3, approx_eq_vec3};
    use crate::types::{mat3_from_rows, DMat3, DVec3};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    const TOL: f64 = 1e-12;

    fn approx_eq_quat(a: &JeodQuat, b: &JeodQuat, tol: f64) -> bool {
        // Quaternions q and -q represent the same rotation.
        // Check both q==q' and q==-q' and accept whichever matches.
        let mut diff_pos = 0.0_f64;
        let mut diff_neg = 0.0_f64;
        for i in 0..4 {
            diff_pos += (a.data[i] - b.data[i]).powi(2);
            diff_neg += (a.data[i] + b.data[i]).powi(2);
        }
        diff_pos.sqrt() < tol || diff_neg.sqrt() < tol
    }

    // ---------------------------------------------------------------
    // identity -> identity matrix
    // ---------------------------------------------------------------
    #[test]
    fn identity_to_matrix() {
        let q = JeodQuat::identity();
        let m = q.left_quat_to_transformation();
        assert!(approx_eq_mat3(&m, &DMat3::IDENTITY, TOL));
    }

    // ---------------------------------------------------------------
    // 90 degrees about Z
    // ---------------------------------------------------------------
    #[test]
    fn rotation_90_z() {
        let q = JeodQuat::left_quat_from_eigen_rotation(FRAC_PI_2, DVec3::Z);
        let m = q.left_quat_to_transformation();

        // JEOD left-transformation matrix for 90-deg about Z:
        // This is the *passive* rotation (inertial -> body).
        // T = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        // Applying T to X-hat=[1,0,0] gives [0,-1,0]: in the rotated body
        // frame, the inertial X axis points along -Y_body.
        let expected = mat3_from_rows(
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        assert!(
            approx_eq_mat3(&m, &expected, TOL),
            "90-deg Z rotation matrix mismatch:\n{:?}\nvs expected:\n{:?}",
            m,
            expected,
        );
    }

    // ---------------------------------------------------------------
    // Quaternion <-> matrix round-trip for many rotations
    // ---------------------------------------------------------------
    #[test]
    fn quat_matrix_roundtrip() {
        let cases: Vec<(f64, DVec3)> = vec![
            (0.0, DVec3::Z),                                       // zero rotation
            (PI, DVec3::X),                                        // 180 X
            (PI, DVec3::Y),                                        // 180 Y
            (PI, DVec3::Z),                                        // 180 Z
            (FRAC_PI_4, DVec3::X),                                 // 45 X
            (1.234, DVec3::Y),                                     // arbitrary Y
            (2.5, DVec3::new(1.0, 1.0, 0.0).normalize()),          // 2.5 rad about (1,1,0)
            (0.01, DVec3::new(0.0, 0.0, 1.0)),                     // small angle Z
            (3.0, DVec3::new(1.0, 2.0, 3.0).normalize()),          // large angle
            (TAU - 0.001, DVec3::new(-1.0, 0.5, 0.3).normalize()), // near full turn
            (0.7, DVec3::new(0.577, 0.577, 0.577).normalize()),    // ~(1,1,1) axis
            (FRAC_PI_2, DVec3::new(0.0, 1.0, 1.0).normalize()),    // 90 about (0,1,1)
        ];

        for (angle, axis) in &cases {
            let q = JeodQuat::left_quat_from_eigen_rotation(*angle, *axis);
            let m = q.left_quat_to_transformation();
            let q2 = JeodQuat::left_quat_from_transformation(&m);
            let m2 = q2.left_quat_to_transformation();

            assert!(
                approx_eq_mat3(&m, &m2, 1e-10),
                "Round-trip failed for angle={}, axis={:?}\nm={:?}\nm2={:?}",
                angle,
                axis,
                m,
                m2
            );

            assert!(
                approx_eq_quat(&q, &q2, 1e-10),
                "Quat round-trip failed for angle={}, axis={:?}\nq={:?}\nq2={:?}",
                angle,
                axis,
                q,
                q2,
            );
        }
    }

    // ---------------------------------------------------------------
    // Normalize
    // ---------------------------------------------------------------
    #[test]
    fn normalize_unit_and_scalar_positive() {
        let mut q = JeodQuat::new(-0.5, 0.5, 0.5, 0.5);
        q.normalize();
        assert!(
            approx_eq_f64(q.norm_sq(), 1.0, 1e-14),
            "norm_sq after normalize: {}",
            q.norm_sq()
        );
        assert!(q.scalar() >= 0.0, "scalar should be non-negative");

        // Also test a far-from-unit quaternion
        let mut q2 = JeodQuat::new(3.0, 4.0, 0.0, 0.0);
        q2.normalize();
        assert!(
            approx_eq_f64(q2.norm_sq(), 1.0, 1e-14),
            "norm_sq after normalize (large): {}",
            q2.norm_sq()
        );
        assert!(q2.scalar() >= 0.0);
    }

    // ---------------------------------------------------------------
    // Multiply: q * conj(q) == identity
    // ---------------------------------------------------------------
    #[test]
    fn multiply_with_conjugate_is_identity() {
        let q =
            JeodQuat::left_quat_from_eigen_rotation(1.23, DVec3::new(1.0, 2.0, 3.0).normalize());
        let qc = q.conjugate();
        let prod = q.multiply(&qc);

        assert!(
            approx_eq_quat(&prod, &JeodQuat::identity(), 1e-12),
            "q * conj(q) should be identity, got {:?}",
            prod
        );
    }

    // ---------------------------------------------------------------
    // Transform: quat vs. matrix
    // ---------------------------------------------------------------
    #[test]
    fn transform_matches_matrix() {
        let test_vecs = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(-4.5, 0.1, 7.8),
        ];

        let q =
            JeodQuat::left_quat_from_eigen_rotation(0.9, DVec3::new(0.3, -0.7, 0.5).normalize());
        let m = q.left_quat_to_transformation();

        for v in &test_vecs {
            let via_quat = q.left_quat_transform(*v);
            let via_mat = m * *v;
            assert!(
                approx_eq_vec3(via_quat, via_mat, 1e-12),
                "Transform mismatch for v={:?}:\n  quat={:?}\n  mat={:?}",
                v,
                via_quat,
                via_mat
            );
        }
    }

    // ---------------------------------------------------------------
    // glam round-trip
    // ---------------------------------------------------------------
    #[test]
    fn glam_conversion_roundtrip() {
        let q = JeodQuat::left_quat_from_eigen_rotation(1.5, DVec3::new(0.0, 1.0, 0.0));
        let g = q.to_glam();
        let q2 = JeodQuat::from_glam(g);
        assert!(
            approx_eq_quat(&q, &q2, 1e-14),
            "glam round-trip failed: {:?} vs {:?}",
            q,
            q2
        );
    }

    // ---------------------------------------------------------------
    // 180-degree rotations (trace ~ -1, edge case for from_transformation)
    // ---------------------------------------------------------------
    #[test]
    fn rotation_180_all_axes() {
        for axis in &[DVec3::X, DVec3::Y, DVec3::Z] {
            let q = JeodQuat::left_quat_from_eigen_rotation(PI, *axis);
            let m = q.left_quat_to_transformation();
            let q2 = JeodQuat::left_quat_from_transformation(&m);
            let m2 = q2.left_quat_to_transformation();
            assert!(
                approx_eq_mat3(&m, &m2, 1e-10),
                "180-degree round-trip failed for axis={:?}",
                axis
            );
        }
    }

    // ---------------------------------------------------------------
    // Small angle rotation
    // ---------------------------------------------------------------
    #[test]
    fn small_angle_rotation() {
        let angle = 1e-10;
        let q = JeodQuat::left_quat_from_eigen_rotation(angle, DVec3::Z);
        let m = q.left_quat_to_transformation();
        // For tiny angle about Z, matrix should be very close to identity
        assert!(approx_eq_mat3(&m, &DMat3::IDENTITY, 1e-8));
    }

    // ---------------------------------------------------------------
    // Composition of rotations
    // ---------------------------------------------------------------
    #[test]
    fn composition() {
        // Two 90-degree rotations about Z should give a 180-degree rotation about Z
        let q90 = JeodQuat::left_quat_from_eigen_rotation(FRAC_PI_2, DVec3::Z);
        let q180_composed = q90.multiply(&q90);
        let q180_direct = JeodQuat::left_quat_from_eigen_rotation(PI, DVec3::Z);

        assert!(
            approx_eq_quat(&q180_composed, &q180_direct, 1e-12),
            "Composition failed: {:?} vs {:?}",
            q180_composed,
            q180_direct,
        );
    }

    // ---------------------------------------------------------------
    // Pin the JEOD-source threshold value as a regression guard.
    // ---------------------------------------------------------------
    #[test]
    fn norm_limit_is_jeod_value() {
        assert_eq!(NORM_LIMIT, 2.107_342e-8);
    }
}
