//! Typed frame transforms: `FrameTransform<From, To>` rotates vectors from
//! frame `From` to frame `To`.
//!
//! Two `FrameTransform`s compose (`A→B` ∘ `B→C` = `A→C`) iff the "inner"
//! frames match — a compile-time check, not a runtime assertion.

use core::marker::PhantomData;
use core::ops::Mul;

use glam::{DMat3, DVec3};
use uom::si::Dimension;

use crate::frame::Frame;
use crate::qty3::Qty3;
use crate::quat::{JeodQuat, LeftTransform, NormalizedQuat, ScalarFirst};

/// Proper rotation taking vectors expressed in `From` to the same vectors
/// expressed in `To`.
///
/// Internally stores a JEOD canonical (scalar-first, left-transformation)
/// quaternion *and* the equivalent 3×3 rotation matrix cached for hot-path
/// application. Both are kept in sync by construction.
#[derive(Debug, Clone, Copy)]
pub struct FrameTransform<From: Frame, To: Frame> {
    quat: NormalizedQuat<ScalarFirst, LeftTransform>,
    matrix: DMat3,
    _from: PhantomData<From>,
    _to: PhantomData<To>,
}

impl<F: Frame> FrameTransform<F, F> {
    /// The identity transform. Only defined when `From = To`, so
    /// `FrameTransform::<A, B>::identity()` with `A ≠ B` fails to typecheck
    /// rather than silently returning a `FrameTransform<A, A>`.
    #[inline]
    pub fn identity() -> FrameTransform<F, F> {
        FrameTransform {
            quat: NormalizedQuat::new(JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]))
                .expect("identity quaternion is normalized"),
            matrix: DMat3::IDENTITY,
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}

impl<From: Frame, To: Frame> FrameTransform<From, To> {
    /// Build a `FrameTransform` from a normalized JEOD quaternion. The
    /// 3×3 rotation matrix is derived once and cached.
    #[inline]
    pub fn from_quat(q: NormalizedQuat<ScalarFirst, LeftTransform>) -> Self {
        let inner = q.inner();
        // JEOD: scalar-first left-transformation. Convert to glam's DQuat
        // (scalar-last) for the rotation-matrix derivation; the transformation
        // convention is preserved because DMat3 := DQuat.to_mat3() yields the
        // same rotation regardless of storage layout.
        let g = glam::DQuat::from_xyzw(inner.data[1], inner.data[2], inner.data[3], inner.data[0]);
        Self {
            quat: q,
            matrix: DMat3::from_quat(g),
            _from: PhantomData,
            _to: PhantomData,
        }
    }

    /// Apply the transform to any `Qty3<D, From>`, producing a `Qty3<D, To>`
    /// of the same dimension.
    #[inline]
    pub fn apply<D: ?Sized + Dimension>(&self, v: Qty3<D, From>) -> Qty3<D, To> {
        let raw: DVec3 = self.matrix * v.raw_si();
        Qty3::<D, To>::from_raw_si(raw)
    }

    /// Inverse transform (`To` → `From`).
    #[inline]
    pub fn inverse(self) -> FrameTransform<To, From> {
        // Conjugate of a unit quaternion is its inverse.
        let q = self.quat.inner();
        let conj = JeodQuat::from_array([q.data[0], -q.data[1], -q.data[2], -q.data[3]]);
        FrameTransform {
            quat: NormalizedQuat::new(conj).expect("conjugate of unit quat is unit"),
            matrix: self.matrix.transpose(),
            _from: PhantomData,
            _to: PhantomData,
        }
    }

    /// The underlying normalized rotation quaternion (read-only).
    #[inline]
    pub const fn quat(&self) -> NormalizedQuat<ScalarFirst, LeftTransform> {
        self.quat
    }

    /// The underlying 3×3 rotation matrix (read-only).
    #[inline]
    pub const fn matrix(&self) -> DMat3 {
        self.matrix
    }

    /// Borrowing accessor for the cached matrix. Parallels [`Self::matrix`]
    /// (which copies by value); prefer this in tight loops or when passing
    /// the matrix to a function that takes `&DMat3`.
    #[inline]
    pub const fn matrix_ref(&self) -> &DMat3 {
        &self.matrix
    }

    /// Construct from a raw rotation matrix. The matrix is stored as-is
    /// (bit-identical to the input); the cached quaternion is derived via
    /// `DQuat::from_mat3`.
    ///
    /// The caller must pass a proper orthonormal rotation matrix — both
    /// debug and release builds `assert!` that `|det(M) - 1| < 1e-9` and
    /// `max|M·Mᵀ - I| < 1e-9`. See the `# Panics` section below for the
    /// exact preconditions and the post-#485 release-mode policy.
    ///
    /// **Use when** the matrix is the source of truth — e.g., RNP / Mars /
    /// Moon / DE421 rotation kernels — and round-tripping through
    /// [`Self::from_quat`] would introduce floating-point drift in the
    /// stored matrix.
    ///
    /// **Bit-stability invariant**: `from_matrix(M).matrix_ref() == &M`
    /// exactly. The quaternion derivation does not influence the stored
    /// matrix.
    ///
    /// Prefer [`from_matrix_validated`](Self::from_matrix_validated) when
    /// the input is unverified (e.g. user-supplied YAML, network protocol)
    /// and you need a typed error rather than a panic.
    ///
    /// # Panics
    ///
    /// In both debug and release builds, panics if the input matrix is not
    /// a proper orthonormal rotation:
    /// - `|det(M) - 1| ≥ 1e-9`, or
    /// - `max |M · Mᵀ − I| ≥ 1e-9` (any element-wise drift from identity).
    ///
    /// Release-mode enforcement is the post-#485 policy: silently caching
    /// a quaternion derived from a non-orthonormal matrix would propagate
    /// wrong physics through every subsequent `.apply()` call. The
    /// per-construction cost (one determinant + one M·Mᵀ multiply) is
    /// negligible at the typical once-per-step callsites (RNP, Mars,
    /// Moon, DE421 rotation updates).
    #[inline]
    pub fn from_matrix(matrix: DMat3) -> Self {
        // JEOD_INV: RF.12 — `FrameTransform::from_matrix` orthonormal input
        // precondition. Release-mode enforced (#485 C3): silently caching a
        // quaternion derived from a non-orthonormal matrix would propagate
        // wrong physics through every `.apply()` call. Callers with
        // untrusted input must use `from_matrix_validated` for the
        // fallible alternative.
        assert!(
            (matrix.determinant() - 1.0).abs() < 1.0e-9,
            "FrameTransform::from_matrix: input must have determinant ≈ 1.0 \
             (got {}). Use `from_matrix_validated` for untrusted input.",
            matrix.determinant()
        );
        assert!(
            {
                let drift = (matrix * matrix.transpose() - DMat3::IDENTITY)
                    .to_cols_array()
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0_f64, f64::max);
                drift < 1.0e-9
            },
            "FrameTransform::from_matrix: input must be orthonormal \
             (M·Mᵀ ≈ I within 1e-9). Use `from_matrix_validated` for \
             untrusted input."
        );

        // Derive the JEOD-canonical (scalar-first, left-transform) quaternion
        // from the matrix. `glam::DQuat::from_mat3` returns a scalar-last
        // quaternion; we re-pack into JEOD layout.
        let g = glam::DQuat::from_mat3(&matrix);
        let jeod = JeodQuat::from_array([g.w, g.x, g.y, g.z]);
        let quat = NormalizedQuat::renormalize(jeod)
            .expect("DQuat::from_mat3 of a non-zero matrix yields a non-zero quaternion");
        Self {
            quat,
            matrix,
            _from: PhantomData,
            _to: PhantomData,
        }
    }

    /// Construct from a raw rotation matrix, returning an error if the
    /// matrix is not a proper orthonormal rotation. Same bit-stability
    /// invariant as [`from_matrix`](Self::from_matrix).
    ///
    /// Prefer this over `from_matrix` when the input is unvalidated
    /// (file I/O, RPC, user-supplied configuration). For per-step rotation
    /// updates from JEOD CSV / DE421 / RNP kernels, `from_matrix` keeps
    /// the runtime check off the hot path.
    ///
    /// Non-finite inputs (any `NaN` or `±∞` element) are rejected up front
    /// with [`FrameTransformError::NonFinite`] — the determinant /
    /// orthonormality checks downstream both silently accept `NaN` (via
    /// `(NaN - 1.0).abs() >= eps == false` and `f64::max(0.0, NaN) == 0.0`),
    /// so the explicit guard is what keeps a `NaN` matrix from reaching
    /// `DQuat::from_mat3`.
    #[inline]
    pub fn from_matrix_validated(matrix: DMat3) -> Result<Self, FrameTransformError> {
        let cols = matrix.to_cols_array();
        if !cols.iter().all(|x| x.is_finite()) {
            return Err(FrameTransformError::NonFinite);
        }
        let det = matrix.determinant();
        if (det - 1.0).abs() >= 1.0e-9 {
            return Err(FrameTransformError::DeterminantNotOne { determinant: det });
        }
        let drift = (matrix * matrix.transpose() - DMat3::IDENTITY)
            .to_cols_array()
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        if drift >= 1.0e-9 {
            return Err(FrameTransformError::NotOrthonormal { drift });
        }
        let g = glam::DQuat::from_mat3(&matrix);
        let jeod = JeodQuat::from_array([g.w, g.x, g.y, g.z]);
        let quat = NormalizedQuat::renormalize(jeod).ok_or(FrameTransformError::ZeroQuaternion)?;
        Ok(Self {
            quat,
            matrix,
            _from: PhantomData,
            _to: PhantomData,
        })
    }
}

/// Reasons [`FrameTransform::from_matrix_validated`] can reject an input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameTransformError {
    /// Input contains a `NaN` or infinite element. The determinant /
    /// orthonormality checks downstream do not reject `NaN` reliably, so
    /// this is checked first.
    NonFinite,
    /// Determinant is not within `1e-9` of `1.0`. A reflection
    /// (`det ≈ -1`) or a scaling (`|det| ≠ 1`) lands here.
    DeterminantNotOne {
        /// The offending determinant value.
        determinant: f64,
    },
    /// `M · Mᵀ` differs from identity by more than `1e-9` in any element.
    NotOrthonormal {
        /// Maximum element-wise drift of `M·Mᵀ` from the identity.
        drift: f64,
    },
    /// `glam::DQuat::from_mat3` produced a zero quaternion (degenerate input).
    ZeroQuaternion,
}

impl core::fmt::Display for FrameTransformError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite => {
                f.write_str("FrameTransform: input matrix has a NaN or infinite element")
            }
            Self::DeterminantNotOne { determinant } => write!(
                f,
                "FrameTransform: input matrix determinant {determinant} not within 1e-9 of 1.0"
            ),
            Self::NotOrthonormal { drift } => write!(
                f,
                "FrameTransform: input matrix not orthonormal — max |M·Mᵀ - I| = {drift}"
            ),
            Self::ZeroQuaternion => {
                f.write_str("FrameTransform: input matrix degenerate — derived quaternion is zero")
            }
        }
    }
}

impl core::error::Error for FrameTransformError {}

/// Compose two transforms: `(A→B) ∘ (B→C) = A→C`.
///
/// The compiler rejects compositions where the inner frames don't match.
///
/// Composition goes through the quaternion representation (the product of
/// two unit quaternions is still unit to within rounding) and then re-derives
/// the cached matrix from the normalized result. Composing the matrices
/// directly and extracting a quaternion from a slightly non-orthonormal
/// product would let `quat()` and `matrix()` drift apart over repeated
/// compositions; this path keeps both cached forms bit-exactly in sync.
impl<A: Frame, B: Frame, C: Frame> Mul<FrameTransform<B, C>> for FrameTransform<A, B> {
    type Output = FrameTransform<A, C>;

    #[inline]
    fn mul(self, rhs: FrameTransform<B, C>) -> Self::Output {
        // Convert both inner quaternions to glam (scalar-last) for the
        // product. In left-transformation convention applying `self` (A→B)
        // and then `rhs` (B→C) yields A→C with quaternion `rhs · self`.
        let s = self.quat.inner();
        let r = rhs.quat.inner();
        let q_self = glam::DQuat::from_xyzw(s.data[1], s.data[2], s.data[3], s.data[0]);
        let q_rhs = glam::DQuat::from_xyzw(r.data[1], r.data[2], r.data[3], r.data[0]);
        let g = (q_rhs * q_self).normalize();
        let composed = JeodQuat::from_array([g.w, g.x, g.y, g.z]);
        FrameTransform {
            quat: NormalizedQuat::new(composed)
                .expect("normalize() of a non-zero quaternion yields a unit quaternion"),
            matrix: DMat3::from_quat(g),
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Ecef, RootInertial};

    /// `from_matrix(M).matrix_ref() == &M` exactly (bit-identical).
    /// The quaternion derivation does not influence the stored matrix —
    /// this is the load-bearing invariant of the Strategy 5 Component
    /// erasure (#155): rotation matrices entering via Components must
    /// emerge bit-identical when read back.
    #[test]
    fn from_matrix_preserves_input_bit_exact() {
        // Use a non-trivial rotation (45° about Y) so any quaternion
        // round-trip would surface as a few-ULP drift.
        let theta = core::f64::consts::FRAC_PI_4;
        let m = DMat3::from_cols(
            DVec3::new(theta.cos(), 0.0, -theta.sin()),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(theta.sin(), 0.0, theta.cos()),
        );
        let t: FrameTransform<RootInertial, Ecef> = FrameTransform::from_matrix(m);
        assert_eq!(
            *t.matrix_ref(),
            m,
            "matrix not bit-preserved by from_matrix"
        );
        assert_eq!(t.matrix(), m, "matrix() must agree with matrix_ref()");
    }

    /// Identity input round-trips through `from_matrix` cleanly and the
    /// derived quaternion is the identity unit quaternion.
    #[test]
    fn from_matrix_identity_round_trip() {
        let t: FrameTransform<RootInertial, RootInertial> =
            FrameTransform::from_matrix(DMat3::IDENTITY);
        assert_eq!(*t.matrix_ref(), DMat3::IDENTITY);
        // The cached quaternion should be near-identity (q0 ≈ 1, qi ≈ 0).
        let q = t.quat().inner();
        assert!((q.data[0].abs() - 1.0).abs() < 1.0e-12);
        assert!(q.data[1].abs() < 1.0e-12);
        assert!(q.data[2].abs() < 1.0e-12);
        assert!(q.data[3].abs() < 1.0e-12);
    }

    /// `apply()` after `from_matrix` produces the same vector as a raw
    /// `M * v` multiply — confirms `apply()` reads from the stored matrix,
    /// not from the derived quaternion.
    #[test]
    fn apply_after_from_matrix_matches_raw_multiply() {
        let theta: f64 = 0.37; // arbitrary non-special angle
        let m = DMat3::from_cols(
            DVec3::new(theta.cos(), theta.sin(), 0.0),
            DVec3::new(-theta.sin(), theta.cos(), 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let t: FrameTransform<RootInertial, Ecef> = FrameTransform::from_matrix(m);
        let v_raw = DVec3::new(1.0, 2.0, 3.0);
        let v_in: Qty3<uom::si::length::Dimension, RootInertial> = Qty3::from_raw_si(v_raw);
        let v_out = t.apply(v_in);
        assert_eq!(v_out.raw_si(), m * v_raw);
    }

    /// `debug_assert!` panic on a non-orthonormal input (in debug builds).
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "determinant")]
    fn from_matrix_rejects_non_unit_determinant() {
        let m = DMat3::from_diagonal(DVec3::new(2.0, 1.0, 1.0)); // det = 2
        let _: FrameTransform<RootInertial, Ecef> = FrameTransform::from_matrix(m);
    }

    /// `from_matrix_validated` accepts a proper rotation and round-trips
    /// the matrix bit-exactly (same invariant as `from_matrix`).
    #[test]
    fn from_matrix_validated_accepts_rotation() {
        let theta: f64 = 0.7;
        let m = DMat3::from_cols(
            DVec3::new(theta.cos(), theta.sin(), 0.0),
            DVec3::new(-theta.sin(), theta.cos(), 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let t: FrameTransform<RootInertial, Ecef> =
            FrameTransform::from_matrix_validated(m).expect("rotation should validate");
        assert_eq!(t.matrix_ref(), &m);
    }

    /// `from_matrix_validated` rejects a scaling matrix with a typed
    /// `DeterminantNotOne` error rather than panicking.
    #[test]
    fn from_matrix_validated_rejects_scaling() {
        let m = DMat3::from_diagonal(DVec3::new(2.0, 1.0, 1.0));
        let err = FrameTransform::<RootInertial, Ecef>::from_matrix_validated(m)
            .expect_err("scaling should reject");
        match err {
            FrameTransformError::DeterminantNotOne { determinant } => {
                assert!((determinant - 2.0).abs() < 1e-12);
            }
            other => panic!("expected DeterminantNotOne, got {other:?}"),
        }
    }

    /// `from_matrix_validated` rejects a near-orthonormal but skewed matrix
    /// with `NotOrthonormal` (det ≈ 1 but `M·Mᵀ ≠ I`).
    #[test]
    fn from_matrix_validated_rejects_non_orthonormal() {
        // Shear: det = 1 but M*M^T != I. cols (1,a,0), (0,1,0), (0,0,1)
        let m = DMat3::from_cols(
            DVec3::new(1.0, 0.1, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        let err = FrameTransform::<RootInertial, Ecef>::from_matrix_validated(m)
            .expect_err("shear should reject");
        assert!(matches!(err, FrameTransformError::NotOrthonormal { .. }));
    }

    /// `from_matrix_validated` rejects a matrix with NaN elements via the
    /// explicit `NonFinite` guard. The downstream determinant /
    /// orthonormality checks both silently accept NaN — `(NaN-1).abs() >=
    /// eps` is `false` and `f64::max(0.0, NaN) == 0.0` — so the explicit
    /// pre-check is what keeps a NaN matrix from reaching `from_mat3`.
    #[test]
    fn from_matrix_validated_rejects_nan() {
        let mut m = DMat3::IDENTITY;
        m.x_axis.x = f64::NAN;
        let err = FrameTransform::<RootInertial, Ecef>::from_matrix_validated(m)
            .expect_err("NaN matrix must reject");
        assert!(matches!(err, FrameTransformError::NonFinite));
    }

    /// `from_matrix_validated` rejects a matrix with infinite elements via
    /// the same `NonFinite` guard.
    #[test]
    fn from_matrix_validated_rejects_infinite() {
        let mut m = DMat3::IDENTITY;
        m.y_axis.z = f64::INFINITY;
        let err = FrameTransform::<RootInertial, Ecef>::from_matrix_validated(m)
            .expect_err("infinite matrix must reject");
        assert!(matches!(err, FrameTransformError::NonFinite));
    }
}
