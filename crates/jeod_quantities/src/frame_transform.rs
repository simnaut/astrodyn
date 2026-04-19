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

impl<From: Frame, To: Frame> FrameTransform<From, To> {
    /// The identity transform (only well-typed when `From = To`, which Rust
    /// enforces at call sites).
    #[inline]
    pub fn identity() -> FrameTransform<From, From> {
        FrameTransform {
            quat: NormalizedQuat::new(JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]))
                .expect("identity quaternion is normalized"),
            matrix: DMat3::IDENTITY,
            _from: PhantomData,
            _to: PhantomData,
        }
    }

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
}

/// Compose two transforms: `(A→B) ∘ (B→C) = A→C`.
///
/// The compiler rejects compositions where the inner frames don't match.
impl<A: Frame, B: Frame, C: Frame> Mul<FrameTransform<B, C>> for FrameTransform<A, B> {
    type Output = FrameTransform<A, C>;

    #[inline]
    fn mul(self, rhs: FrameTransform<B, C>) -> Self::Output {
        // Composition of rotations: output matrix = rhs · self gives
        // (A→C)(v) = rhs(self(v)). We store the matrix that sends v_A to v_C.
        let m = rhs.matrix * self.matrix;
        // Derive the composed quaternion via matrix → quaternion round-trip.
        let g = glam::DQuat::from_mat3(&m);
        let composed = JeodQuat::from_array([g.w, g.x, g.y, g.z]);
        FrameTransform {
            quat: NormalizedQuat::renormalize(composed)
                .expect("composition of rotations yields a non-zero quaternion"),
            matrix: m,
            _from: PhantomData,
            _to: PhantomData,
        }
    }
}
