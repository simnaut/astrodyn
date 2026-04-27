//! Quaternion convention tags and a normalization witness.
//!
//! JEOD canonically uses **scalar-first, left-transformation** quaternions
//! `[q0, q1, q2, q3]` with `q0` scalar. `glam::DQuat` uses `[x, y, z, w]`
//! with `w` scalar. This module lifts the *convention* to the type system so
//! that code which expects a JEOD-layout, left-transformation quaternion
//! cannot be handed a glam-layout or right-transformation one by mistake.
//!
//! The glam bridge lives in this crate, restricted to the one convention
//! that matches glam (`ScalarLast` + `LeftTransform`). Convert from
//! `JeodQuat` via `q.to_scalar_last().to_glam()` and back via
//! `Quat::<ScalarLast, LeftTransform>::from(glam_q).to_scalar_first()`.
//! `jeod_math::JeodQuat` still exposes convenience helpers for callers
//! who are already working with that type at the JEOD↔Rust boundary.

use core::marker::PhantomData;

use glam::{DMat3, DQuat, DVec3};

use crate::sealed::QuatSealed;

/// Compile-time quaternion storage layout marker.
///
/// Sealed at the type-system level: only `jeod_quantities` can impl
/// this trait (the seal trait `QuatSealed` is private to the crate).
pub trait Layout: QuatSealed + 'static {
    const NAME: &'static str;
}

/// Storage layout `[q0, q1, q2, q3]` where `q0` is the scalar part (JEOD).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalarFirst;
impl QuatSealed for ScalarFirst {}
impl Layout for ScalarFirst {
    const NAME: &'static str = "ScalarFirst";
}

/// Storage layout `[x, y, z, w]` where `w` is the scalar part (glam).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalarLast;
impl QuatSealed for ScalarLast {}
impl Layout for ScalarLast {
    const NAME: &'static str = "ScalarLast";
}

/// Compile-time quaternion transformation convention marker.
///
/// Sealed at the type-system level: only `jeod_quantities` can impl
/// this trait (the seal trait `QuatSealed` is private to the crate).
pub trait Transform: QuatSealed + 'static {
    const NAME: &'static str;
}

/// `r' = q r q⁻¹` — the JEOD convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeftTransform;
impl QuatSealed for LeftTransform {}
impl Transform for LeftTransform {
    const NAME: &'static str = "LeftTransform";
}

/// `r' = q⁻¹ r q` — the opposite of JEOD; common in many textbooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RightTransform;
impl QuatSealed for RightTransform {}
impl Transform for RightTransform {
    const NAME: &'static str = "RightTransform";
}

/// Quaternion tagged with its storage layout and transformation convention.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat<L: Layout, T: Transform> {
    /// Four components in the order dictated by `L`.
    pub data: [f64; 4],
    _l: PhantomData<L>,
    _t: PhantomData<T>,
}

impl<L: Layout, T: Transform> Quat<L, T> {
    /// Construct from a raw `[f64; 4]`. The caller is responsible for the
    /// ordering (scalar-first vs scalar-last).
    #[inline]
    pub const fn from_array(data: [f64; 4]) -> Self {
        Self {
            data,
            _l: PhantomData,
            _t: PhantomData,
        }
    }

    /// Raw squared norm.
    #[inline]
    pub fn norm_squared(&self) -> f64 {
        self.data[0] * self.data[0]
            + self.data[1] * self.data[1]
            + self.data[2] * self.data[2]
            + self.data[3] * self.data[3]
    }

    /// Raw norm.
    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

// --- JEOD canonical alias ---

/// The JEOD canonical quaternion type: scalar-first, left-transformation.
pub type JeodQuat = Quat<ScalarFirst, LeftTransform>;

// --- Conversions between layouts (same transformation convention) ---

impl<T: Transform> Quat<ScalarFirst, T> {
    /// Convert to scalar-last layout (e.g. to hand off to `glam::DQuat`).
    #[inline]
    pub fn to_scalar_last(self) -> Quat<ScalarLast, T> {
        Quat::from_array([self.data[1], self.data[2], self.data[3], self.data[0]])
    }
}

impl<T: Transform> Quat<ScalarLast, T> {
    /// Convert to scalar-first layout (JEOD canonical).
    #[inline]
    pub fn to_scalar_first(self) -> Quat<ScalarFirst, T> {
        Quat::from_array([self.data[3], self.data[0], self.data[1], self.data[2]])
    }
}

// --- glam bridging ---
//
// `glam::DQuat` stores `[x, y, z, w]` and applies its rotations under the
// left-transformation convention (`r' = q r q⁻¹`), which matches JEOD. The
// bridging impls below are restricted to `LeftTransform` so converting
// a `Quat<ScalarLast, RightTransform>` to/from `DQuat` is rejected at
// compile time rather than silently mislabeled.
//
// Callers who genuinely hold a RightTransform quaternion should conjugate
// explicitly (flip the sign of the vector part) before bridging.

impl Quat<ScalarLast, LeftTransform> {
    /// Interpret the quaternion as a `glam::DQuat`. Zero-cost: the layout
    /// is `[x, y, z, w]` and the transformation convention matches glam.
    #[inline]
    pub fn to_glam(self) -> DQuat {
        DQuat::from_xyzw(self.data[0], self.data[1], self.data[2], self.data[3])
    }
}

impl From<DQuat> for Quat<ScalarLast, LeftTransform> {
    #[inline]
    fn from(q: DQuat) -> Self {
        Self::from_array([q.x, q.y, q.z, q.w])
    }
}

/// Error returned by [`NormalizedQuat::new`] when the wrapped quaternion is
/// not close enough to unit norm.
#[derive(Debug, thiserror::Error)]
#[error("quaternion norm {norm} deviates from 1 by {deviation:.3e}, which exceeds tolerance {tolerance:.3e}")]
pub struct NotNormalized {
    pub norm: f64,
    pub deviation: f64,
    pub tolerance: f64,
}

/// A quaternion witnessed to have unit norm at construction time.
///
/// Witnesses are invalidated silently by arithmetic on the inner `data`, so
/// this type exposes *no* mutable accessor — all transformations that could
/// denormalize go through re-normalizing constructors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedQuat<L: Layout, T: Transform>(Quat<L, T>);

impl<L: Layout, T: Transform> NormalizedQuat<L, T> {
    /// Default tolerance: norm must lie within `1 ± 1e-12`.
    pub const DEFAULT_TOLERANCE: f64 = 1e-12;

    /// Wrap a quaternion iff its norm is within [`Self::DEFAULT_TOLERANCE`] of 1.
    #[inline]
    pub fn new(q: Quat<L, T>) -> Result<Self, NotNormalized> {
        Self::new_with_tolerance(q, Self::DEFAULT_TOLERANCE)
    }

    /// Wrap a quaternion iff its norm is within the given tolerance of 1.
    #[inline]
    pub fn new_with_tolerance(q: Quat<L, T>, tolerance: f64) -> Result<Self, NotNormalized> {
        let norm = q.norm();
        let deviation = (norm - 1.0).abs();
        if deviation <= tolerance {
            Ok(Self(q))
        } else {
            Err(NotNormalized {
                norm,
                deviation,
                tolerance,
            })
        }
    }

    /// Renormalize an arbitrary quaternion into this witness.
    ///
    /// Returns `None` if the norm is not finite and strictly positive —
    /// i.e. the input is all zeros, contains any NaN, or has components so
    /// large that `‖q‖` overflows to infinity. All three cases would
    /// otherwise produce a non-unit witness.
    #[inline]
    pub fn renormalize(q: Quat<L, T>) -> Option<Self> {
        let n = q.norm();
        if !(n.is_finite() && n > 0.0) {
            return None;
        }
        let inv = 1.0 / n;
        Some(Self(Quat::from_array([
            q.data[0] * inv,
            q.data[1] * inv,
            q.data[2] * inv,
            q.data[3] * inv,
        ])))
    }

    /// Read-only view of the underlying quaternion.
    #[inline]
    pub const fn inner(self) -> Quat<L, T> {
        self.0
    }
}

// ============================================================================
// JEOD canonical (ScalarFirst, LeftTransform) algebra, matrix conversions,
// axis-angle construction, vector transform.
//
// These impls live here (rather than in `jeod_math::quaternion`) because
// inherent impls may only appear in the defining crate. See
// `jeod_math::quaternion` for the JEOD-specific `normalize_integ` variant
// that preserves the scalar sign.
// ============================================================================

/// Threshold (as a scalar difference `1 - ‖q‖²`) below which the fast Padé
/// renormalization path is used. JEOD source:
/// `models/utils/quaternion/src/quaternion_normalize.cc`.
///
/// Single source of truth: `jeod_math::quaternion::NORM_LIMIT` re-exports
/// this constant so JEOD-specific consumers can reach it under their
/// expected path without duplicating the literal.
pub const NORM_LIMIT: f64 = 2.107_342e-8;

impl JeodQuat {
    /// Identity quaternion `[1, 0, 0, 0]`: no rotation.
    #[inline]
    pub const fn identity() -> Self {
        Self::from_array([1.0, 0.0, 0.0, 0.0])
    }

    /// Construct from explicit scalar-first components.
    #[inline]
    pub const fn new(scalar: f64, vx: f64, vy: f64, vz: f64) -> Self {
        Self::from_array([scalar, vx, vy, vz])
    }

    /// Scalar (real) part `q0`.
    #[inline]
    pub fn scalar(&self) -> f64 {
        self.data[0]
    }

    /// Vector (imaginary) part `[q1, q2, q3]`.
    #[inline]
    pub fn vector(&self) -> DVec3 {
        DVec3::new(self.data[1], self.data[2], self.data[3])
    }

    /// Squared norm (alias for [`Quat::norm_squared`] — kept for JEOD parity naming).
    #[inline]
    pub fn norm_sq(&self) -> f64 {
        self.norm_squared()
    }

    // ----------------------------------------------------------------
    // Conversions to/from glam::DQuat
    // ----------------------------------------------------------------

    /// Convert to a glam `DQuat`. glam stores `(x, y, z, w)` where `w` is the scalar.
    #[inline]
    pub fn to_glam(&self) -> DQuat {
        DQuat::from_xyzw(self.data[1], self.data[2], self.data[3], self.data[0])
    }

    /// Create from a glam `DQuat` (which uses `[x, y, z, w]`).
    #[inline]
    pub fn from_glam(q: DQuat) -> Self {
        Self::from_array([q.w, q.x, q.y, q.z])
    }

    // ----------------------------------------------------------------
    // Algebraic operations
    // ----------------------------------------------------------------

    /// Quaternion conjugate: `[s, -v]`.
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self::from_array([self.data[0], -self.data[1], -self.data[2], -self.data[3]])
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

        Self::from_array([ps, pv.x, pv.y, pv.z])
    }

    // ----------------------------------------------------------------
    // Normalization (JEOD fast-path with canonical hemisphere)
    // ----------------------------------------------------------------

    /// Normalize the quaternion in place, forcing `scalar >= 0`
    /// (canonical hemisphere).
    ///
    /// Uses JEOD's fast Padé approximation when the quaternion is already
    /// close to unit length.  See
    /// `models/utils/quaternion/src/quaternion_normalize.cc`.
    pub fn normalize(&mut self) {
        let qmagsq = self.norm_squared();
        // JEOD_INV: QT.03 — cannot normalize a zero quaternion
        assert!(qmagsq > 0.0, "cannot normalize a zero quaternion");

        // JEOD_INV: QT.01 — fast Padé path near unit magnitude, sqrt path otherwise
        let fact = if (1.0 - qmagsq).abs() < NORM_LIMIT {
            // Near-unit: first-order Padé approximant  2 / (1 + ||q||²)
            2.0 / (1.0 + qmagsq)
        } else {
            1.0 / qmagsq.sqrt()
        };

        for d in self.data.iter_mut() {
            *d *= fact;
        }

        // JEOD_INV: QT.02 — canonical hemisphere: force scalar non-negative
        if self.data[0] < 0.0 {
            for d in self.data.iter_mut() {
                *d = -*d;
            }
        }
    }

    // ----------------------------------------------------------------
    // Axis-angle construction
    // ----------------------------------------------------------------

    /// Construct a left-transform quaternion from an axis-angle rotation.
    ///
    /// JEOD convention: `scalar = cos(θ/2)`, `vector = -sin(θ/2) · axis`.
    /// `axis` must be a unit vector.
    pub fn left_quat_from_eigen_rotation(angle: f64, axis: DVec3) -> Self {
        let half = angle * 0.5;
        let s = half.cos();
        let v = -half.sin() * axis;
        let mut q = Self::from_array([s, v.x, v.y, v.z]);
        q.normalize();
        q
    }

    // ----------------------------------------------------------------
    // Matrix <-> quaternion (left-transformation convention)
    // ----------------------------------------------------------------

    // JEOD_INV: RF.09 — assumes quaternion is normalized (caller must normalize after integration)
    /// Build the 3x3 rotation (transformation) matrix from a left quaternion,
    /// **without** verifying unit norm.
    ///
    /// This is the historical JEOD-parity signature retained for migration
    /// purposes. New callers should prefer the [`NormalizedQuat`] witness
    /// version (see module docs of `jeod_math::quaternion`).
    ///
    /// Formula from JEOD `models/utils/quaternion/src/quaternion_to_matrix.cc`.
    ///
    /// ```text
    /// cost  = 2·qs² − 1
    /// T[i][i] = cost + 2·qv[i]²
    /// T[i][j] = 2·(qv[i]·qv[j] ∓ qs·qv[k])
    /// ```
    pub fn left_quat_to_transformation(&self) -> DMat3 {
        left_quat_to_transformation_impl(self)
    }

    /// Build a left quaternion from a transformation matrix.
    ///
    /// Robust method from JEOD
    /// `models/utils/quaternion/src/quaternion_from_matrix.cc`: selects
    /// among four extraction branches based on which of
    /// `{trace, T[0][0], T[1][1], T[2][2]}` is largest.
    pub fn left_quat_from_transformation(mat: &DMat3) -> Self {
        // Convenience accessor: T[i][j] = mat.col(j)[i]
        let t = |r: usize, c: usize| -> f64 { mat.col(c)[r] };

        let tr = t(0, 0) + t(1, 1) + t(2, 2);

        // Find maximum of (tr, T00, T11, T22)
        let vals = [tr, t(0, 0), t(1, 1), t(2, 2)];
        let max_idx = vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
            .unwrap()
            .0;

        let mut q = [0.0_f64; 4];

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

        let mut result = Self::from_array(q);
        result.normalize();
        result
    }

    // ----------------------------------------------------------------
    // Vector transformation (Rodrigues via quaternion)
    // ----------------------------------------------------------------

    /// Transform a vector using the quaternion without building a matrix.
    ///
    /// ```text
    /// t  = 2 · (qv × v)
    /// v' = v + qs·t + qv × t
    /// ```
    pub fn left_quat_transform(&self, v: DVec3) -> DVec3 {
        let qs = self.scalar();
        let qv = self.vector();

        let qv_cross_v = qv.cross(v);
        v + 2.0 * qs * qv_cross_v + 2.0 * qv.cross(qv_cross_v)
    }
}

// ----------------------------------------------------------------
// NormalizedQuat: JEOD witness-gated operations
// ----------------------------------------------------------------

impl NormalizedQuat<ScalarFirst, LeftTransform> {
    /// Build the 3x3 rotation (transformation) matrix from a witnessed
    /// unit-norm left quaternion.
    ///
    /// Preferred over `JeodQuat::left_quat_to_transformation` because the
    /// witness proves the unit-norm precondition at the type level.
    #[inline]
    pub fn left_quat_to_transformation(&self) -> DMat3 {
        left_quat_to_transformation_impl(&self.inner())
    }

    /// Transform a vector using a witnessed unit-norm left quaternion.
    #[inline]
    pub fn left_quat_transform(&self, v: DVec3) -> DVec3 {
        self.inner().left_quat_transform(v)
    }
}

// Shared inner implementation used by both the raw-`JeodQuat` and the
// witness-gated `NormalizedQuat<ScalarFirst, LeftTransform>` call sites.
#[inline]
fn left_quat_to_transformation_impl(q: &JeodQuat) -> DMat3 {
    let qs = q.data[0];
    let qv = [q.data[1], q.data[2], q.data[3]];

    let cost = 2.0 * qs * qs - 1.0;

    // Diagonal
    let t00 = cost + 2.0 * qv[0] * qv[0];
    let t11 = cost + 2.0 * qv[1] * qv[1];
    let t22 = cost + 2.0 * qv[2] * qv[2];

    // Off-diagonal:
    //   T[0][1] = 2·(qv0·qv1 − qs·qv2)
    //   T[1][0] = 2·(qv1·qv0 + qs·qv2)
    //   T[0][2] = 2·(qv0·qv2 + qs·qv1)
    //   T[2][0] = 2·(qv2·qv0 − qs·qv1)
    //   T[1][2] = 2·(qv1·qv2 − qs·qv0)
    //   T[2][1] = 2·(qv2·qv1 + qs·qv0)
    let t01 = 2.0 * (qv[0] * qv[1] - qs * qv[2]);
    let t10 = 2.0 * (qv[1] * qv[0] + qs * qv[2]);
    let t02 = 2.0 * (qv[0] * qv[2] + qs * qv[1]);
    let t20 = 2.0 * (qv[2] * qv[0] - qs * qv[1]);
    let t12 = 2.0 * (qv[1] * qv[2] - qs * qv[0]);
    let t21 = 2.0 * (qv[2] * qv[1] + qs * qv[0]);

    // glam stores column-major. Build columns from our row-indexed T.
    DMat3::from_cols(
        DVec3::new(t00, t10, t20), // col 0 -> T[0..3][0]
        DVec3::new(t01, t11, t21), // col 1 -> T[0..3][1]
        DVec3::new(t02, t12, t22), // col 2 -> T[0..3][2]
    )
}
