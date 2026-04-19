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

use glam::DQuat;

use crate::sealed::Sealed;

/// Compile-time quaternion storage layout marker.
pub trait Layout: Sealed + 'static {
    const NAME: &'static str;
}

/// Storage layout `[q0, q1, q2, q3]` where `q0` is the scalar part (JEOD).
#[derive(Debug, Clone, Copy)]
pub struct ScalarFirst;
impl Sealed for ScalarFirst {}
impl Layout for ScalarFirst {
    const NAME: &'static str = "ScalarFirst";
}

/// Storage layout `[x, y, z, w]` where `w` is the scalar part (glam).
#[derive(Debug, Clone, Copy)]
pub struct ScalarLast;
impl Sealed for ScalarLast {}
impl Layout for ScalarLast {
    const NAME: &'static str = "ScalarLast";
}

/// Compile-time quaternion transformation convention marker.
pub trait Transform: Sealed + 'static {
    const NAME: &'static str;
}

/// `r' = q r q⁻¹` — the JEOD convention.
#[derive(Debug, Clone, Copy)]
pub struct LeftTransform;
impl Sealed for LeftTransform {}
impl Transform for LeftTransform {
    const NAME: &'static str = "LeftTransform";
}

/// `r' = q⁻¹ r q` — the opposite of JEOD; common in many textbooks.
#[derive(Debug, Clone, Copy)]
pub struct RightTransform;
impl Sealed for RightTransform {}
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
