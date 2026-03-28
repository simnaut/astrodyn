use crate::types::{mat3_from_rows, DMat3, DQuat, DVec3};

/// Scalar-first, left-transformation quaternion matching JEOD convention.
///
/// Layout: `[scalar, vx, vy, vz]`
///
/// For a rotation of angle theta about unit axis u-hat:
///   scalar = cos(theta/2)
///   vector = -sin(theta/2) * u-hat
///
/// A "left quaternion" transforms a vector as: v' = q * v * q_conj
/// which is equivalent to the rotation matrix produced by
/// `left_quat_to_transformation`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JeodQuat {
    pub data: [f64; 4], // [scalar, vx, vy, vz]
}

/// Threshold below which the fast renormalization (Padé approximant) path is used.
/// From JEOD `models/utils/quaternion/src/quaternion_normalize.cc`.
pub const NORM_LIMIT: f64 = 2.107_342e-8;

impl JeodQuat {
    /// Identity quaternion: no rotation.
    pub fn identity() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Construct from explicit components.
    pub fn new(scalar: f64, vx: f64, vy: f64, vz: f64) -> Self {
        Self {
            data: [scalar, vx, vy, vz],
        }
    }

    /// Scalar (real) part.
    #[inline]
    pub fn scalar(&self) -> f64 {
        self.data[0]
    }

    /// Vector (imaginary) part.
    #[inline]
    pub fn vector(&self) -> DVec3 {
        DVec3::new(self.data[1], self.data[2], self.data[3])
    }

    // ----------------------------------------------------------------
    // Conversions to/from glam
    // ----------------------------------------------------------------

    /// Convert to a glam `DQuat`.
    ///
    /// glam stores `(x, y, z, w)` where w is the scalar.
    pub fn to_glam(&self) -> DQuat {
        DQuat::from_xyzw(self.data[1], self.data[2], self.data[3], self.data[0])
    }

    /// Create from a glam `DQuat`.
    pub fn from_glam(q: DQuat) -> Self {
        Self {
            data: [q.w, q.x, q.y, q.z],
        }
    }

    // ----------------------------------------------------------------
    // Norm helpers
    // ----------------------------------------------------------------

    /// Squared norm of the quaternion.
    pub fn norm_sq(&self) -> f64 {
        self.data[0] * self.data[0]
            + self.data[1] * self.data[1]
            + self.data[2] * self.data[2]
            + self.data[3] * self.data[3]
    }

    /// Normalize the quaternion in place.
    ///
    /// Uses JEOD's fast two-step approximation when the quaternion is already
    /// close to unit length.  Always forces scalar >= 0.
    pub fn normalize(&mut self) {
        let qmagsq = self.norm_sq();
        assert!(qmagsq > 0.0, "cannot normalize a zero quaternion");

        let fact = if (1.0 - qmagsq).abs() < NORM_LIMIT {
            // Near-unit: first-order Padé approximant  2 / (1 + ||q||²)
            2.0 / (1.0 + qmagsq)
        } else {
            1.0 / qmagsq.sqrt()
        };

        for d in self.data.iter_mut() {
            *d *= fact;
        }

        // Force scalar non-negative (canonical hemisphere).
        if self.data[0] < 0.0 {
            for d in self.data.iter_mut() {
                *d = -*d;
            }
        }
    }

    // ----------------------------------------------------------------
    // Algebraic operations
    // ----------------------------------------------------------------

    /// Quaternion conjugate: `[s, -v]`.
    pub fn conjugate(&self) -> Self {
        Self {
            data: [self.data[0], -self.data[1], -self.data[2], -self.data[3]],
        }
    }

    /// Quaternion product `self * other`.
    ///
    /// ```text
    /// prod.scalar = s1*s2 - v1 . v2
    /// prod.vector = s1*v2 + s2*v1 + v1 x v2
    /// ```
    pub fn multiply(&self, other: &Self) -> Self {
        let s1 = self.scalar();
        let v1 = self.vector();
        let s2 = other.scalar();
        let v2 = other.vector();

        let ps = s1 * s2 - v1.dot(v2);
        let pv = v2 * s1 + v1 * s2 + v1.cross(v2);

        Self {
            data: [ps, pv.x, pv.y, pv.z],
        }
    }

    // ----------------------------------------------------------------
    // Matrix <-> quaternion (left-transformation convention)
    // ----------------------------------------------------------------

    // JEOD_INV: RF.09 — assumes quaternion is normalized (caller must normalize after integration)
    /// Build the 3x3 rotation (transformation) matrix from a left quaternion.
    ///
    /// Uses the half-angle formula from JEOD
    /// `models/utils/quaternion/src/quaternion_to_matrix.cc`.
    ///
    /// ```text
    /// cost  = 2*qs^2 - 1
    /// T[i][i] = cost + 2*qv[i]^2
    /// T[i][j] = 2*(qv[i]*qv[j] -/+ qs*qv[k])
    /// ```
    pub fn left_quat_to_transformation(&self) -> DMat3 {
        let qs = self.data[0];
        let qv = [self.data[1], self.data[2], self.data[3]];

        let cost = 2.0 * qs * qs - 1.0;

        // Diagonal
        let t00 = cost + 2.0 * qv[0] * qv[0];
        let t11 = cost + 2.0 * qv[1] * qv[1];
        let t22 = cost + 2.0 * qv[2] * qv[2];

        // Off-diagonal:
        //   T[0][1] = 2*(qv0*qv1 - qs*qv2)
        //   T[1][0] = 2*(qv1*qv0 + qs*qv2)
        //   T[0][2] = 2*(qv0*qv2 + qs*qv1)
        //   T[2][0] = 2*(qv2*qv0 - qs*qv1)
        //   T[1][2] = 2*(qv1*qv2 - qs*qv0)
        //   T[2][1] = 2*(qv2*qv1 + qs*qv0)
        let t01 = 2.0 * (qv[0] * qv[1] - qs * qv[2]);
        let t10 = 2.0 * (qv[1] * qv[0] + qs * qv[2]);
        let t02 = 2.0 * (qv[0] * qv[2] + qs * qv[1]);
        let t20 = 2.0 * (qv[2] * qv[0] - qs * qv[1]);
        let t12 = 2.0 * (qv[1] * qv[2] - qs * qv[0]);
        let t21 = 2.0 * (qv[2] * qv[1] + qs * qv[0]);

        mat3_from_rows(
            DVec3::new(t00, t01, t02),
            DVec3::new(t10, t11, t12),
            DVec3::new(t20, t21, t22),
        )
    }

    /// Build a left quaternion from a transformation matrix.
    ///
    /// Robust method from JEOD
    /// `models/utils/quaternion/src/quaternion_from_matrix.cc`:
    /// selects among 4 extraction branches based on which of
    /// {trace, T\[0\]\[0\], T\[1\]\[1\], T\[2\]\[2\]} is largest.
    ///
    /// `mat` is a glam `DMat3` (column-major). Access element T\[row\]\[col\]
    /// via `mat.col(col)[row]`.
    pub fn left_quat_from_transformation(mat: &DMat3) -> Self {
        // Convenience macro: T[i][j] = mat.col(j)[i]
        let t = |r: usize, c: usize| -> f64 { mat.col(c)[r] };

        let tr = t(0, 0) + t(1, 1) + t(2, 2);

        // Find maximum of (tr, T00, T11, T22)
        let vals = [tr, t(0, 0), t(1, 1), t(2, 2)];
        let max_idx = vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let mut q = [0.0_f64; 4];

        // For the JEOD left-transformation matrix, the off-diagonal
        // anti-symmetric part gives:
        //   T[2][1] - T[1][2] =  4*qs*qv[0]
        //   T[0][2] - T[2][0] =  4*qs*qv[1]
        //   T[1][0] - T[0][1] =  4*qs*qv[2]
        // The symmetric off-diagonal part gives:
        //   T[0][1] + T[1][0] =  4*qv[0]*qv[1]
        //   T[0][2] + T[2][0] =  4*qv[0]*qv[2]
        //   T[1][2] + T[2][1] =  4*qv[1]*qv[2]
        match max_idx {
            0 => {
                // tr dominates -> solve for qs first
                q[0] = 0.5 * (1.0 + tr).sqrt();
                let inv4qs = 0.25 / q[0];
                q[1] = (t(2, 1) - t(1, 2)) * inv4qs;
                q[2] = (t(0, 2) - t(2, 0)) * inv4qs;
                q[3] = (t(1, 0) - t(0, 1)) * inv4qs;
            }
            1 => {
                // T[0][0] dominates -> solve for qv[0] first
                q[1] = 0.5 * (1.0 + 2.0 * t(0, 0) - tr).sqrt();
                let inv4qv0 = 0.25 / q[1];
                q[0] = (t(2, 1) - t(1, 2)) * inv4qv0;
                q[2] = (t(0, 1) + t(1, 0)) * inv4qv0;
                q[3] = (t(0, 2) + t(2, 0)) * inv4qv0;
            }
            2 => {
                // T[1][1] dominates -> solve for qv[1] first
                q[2] = 0.5 * (1.0 + 2.0 * t(1, 1) - tr).sqrt();
                let inv4qv1 = 0.25 / q[2];
                q[0] = (t(0, 2) - t(2, 0)) * inv4qv1;
                q[1] = (t(0, 1) + t(1, 0)) * inv4qv1;
                q[3] = (t(1, 2) + t(2, 1)) * inv4qv1;
            }
            3 => {
                // T[2][2] dominates -> solve for qv[2] first
                q[3] = 0.5 * (1.0 + 2.0 * t(2, 2) - tr).sqrt();
                let inv4qv2 = 0.25 / q[3];
                q[0] = (t(1, 0) - t(0, 1)) * inv4qv2;
                q[1] = (t(0, 2) + t(2, 0)) * inv4qv2;
                q[2] = (t(1, 2) + t(2, 1)) * inv4qv2;
            }
            _ => unreachable!(),
        }

        // Force scalar non-negative (canonical hemisphere)
        if q[0] < 0.0 {
            for v in q.iter_mut() {
                *v = -*v;
            }
        }

        let mut result = Self { data: q };
        result.normalize();
        result
    }

    // ----------------------------------------------------------------
    // Vector transformation
    // ----------------------------------------------------------------

    /// Transform a vector using the quaternion without building a matrix.
    ///
    /// Rodrigues formula via quaternion:
    /// ```text
    /// t = 2 * (qv x v)
    /// v' = v + qs*t + qv x t
    /// ```
    pub fn left_quat_transform(&self, v: DVec3) -> DVec3 {
        let qs = self.scalar();
        let qv = self.vector();

        let qv_cross_v = qv.cross(v);
        v + 2.0 * qs * qv_cross_v + 2.0 * qv.cross(qv_cross_v)
    }

    // ----------------------------------------------------------------
    // Axis-angle construction
    // ----------------------------------------------------------------

    /// Construct a left-transform quaternion from an axis-angle rotation.
    ///
    /// JEOD convention: scalar = cos(theta/2),  vector = -sin(theta/2) * axis
    ///
    /// `axis` must be a unit vector.
    pub fn left_quat_from_eigen_rotation(angle: f64, axis: DVec3) -> Self {
        let half = angle * 0.5;
        let s = half.cos();
        let v = -half.sin() * axis;
        let mut q = Self {
            data: [s, v.x, v.y, v.z],
        };
        q.normalize();
        q
    }
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{approx_eq_f64, approx_eq_mat3, approx_eq_vec3};
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
            (2.5, DVec3::new(1.0, 1.0, 0.0).normalize()),         // 2.5 rad about (1,1,0)
            (0.01, DVec3::new(0.0, 0.0, 1.0)),                    // small angle Z
            (3.0, DVec3::new(1.0, 2.0, 3.0).normalize()),         // large angle
            (TAU - 0.001, DVec3::new(-1.0, 0.5, 0.3).normalize()), // near full turn
            (0.7, DVec3::new(0.577, 0.577, 0.577).normalize()),   // ~(1,1,1) axis
            (FRAC_PI_2, DVec3::new(0.0, 1.0, 1.0).normalize()),   // 90 about (0,1,1)
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
        let q = JeodQuat::left_quat_from_eigen_rotation(1.23, DVec3::new(1.0, 2.0, 3.0).normalize());
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

        let q = JeodQuat::left_quat_from_eigen_rotation(0.9, DVec3::new(0.3, -0.7, 0.5).normalize());
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
}
