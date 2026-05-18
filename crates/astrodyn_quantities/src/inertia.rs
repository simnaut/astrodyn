//! `InertiaTensor<F>` — a frame-tagged 3×3 inertia tensor.
//!
//! JEOD's `MassProperties::inertia` is a `DMat3` carrying mass moment of
//! inertia (dimension `M·L²`, base SI unit `kg·m²`) about the mass body's
//! axes. This module wraps that storage with a phantom frame parameter so
//! `InertiaTensor<BodyFrame<V>>` and `InertiaTensor<StructuralFrame<V>>`
//! are distinct types — `Add` requires both operands to share the same
//! `F`, so cross-frame additions fail to compile.
//!
//! [`InertiaTensor::transform`] is the **numeric** similarity transform
//! `T^T · I · T`; it does **not** change the frame phantom. To re-express
//! an inertia tensor in a different frame, callers must explicitly retag
//! after rotating: `InertiaTensor::<NewFrame>::from_dmat3_unchecked(
//! original.transform(&rot).as_dmat3())`. A safer typed-`FrameTransform`
//! wrapper is left for a follow-up phase.
//!
//! Storage is `DMat3` (no `uom` integration) — the dimension is a phantom
//! annotation. Callers feed in / read out raw `DMat3` values whose entries
//! are interpreted as `kg·m²`.

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use glam::DMat3;

use crate::derive_utils::{
    impl_clone_phantom, impl_copy_phantom, impl_debug_phantom, impl_partial_eq_phantom,
};
use crate::frame::Frame;

/// Mass moment of inertia tensor (`kg·m²`) expressed in frame `F`.
///
/// Symmetric in physically-meaningful instances, but the type does not
/// enforce symmetry — composition via [`Self::transform`] preserves it
/// when the input is symmetric and the rotation is proper.
///
/// # Frame-mismatch is a compile error
///
/// Adding inertia tensors expressed in different compile-time frames
/// fails to typecheck rather than silently mis-summing them:
///
/// ```compile_fail
/// use astrodyn_quantities::frame::{Ecef, RootInertial};
/// use astrodyn_quantities::inertia::InertiaTensor;
///
/// let a = InertiaTensor::<RootInertial>::from_principal(1.0, 1.0, 1.0);
/// let b = InertiaTensor::<Ecef>::from_principal(1.0, 1.0, 1.0);
/// let _bad = a + b; // Add is only impl'd for matching frames
/// ```
#[repr(transparent)]
pub struct InertiaTensor<F: Frame> {
    value: DMat3,
    _f: PhantomData<F>,
}

impl<F: Frame> InertiaTensor<F> {
    /// Wrap a raw `DMat3` whose entries are interpreted as `kg·m²` in
    /// frame `F`. The caller is responsible for the unit and for any
    /// symmetry / positive-definite invariants the downstream physics
    /// expects.
    #[inline]
    pub const fn from_dmat3_unchecked(value: DMat3) -> Self {
        Self {
            value,
            _f: PhantomData,
        }
    }

    /// The zero tensor (useful as an accumulator seed).
    #[inline]
    pub fn zero() -> Self {
        Self::from_dmat3_unchecked(DMat3::ZERO)
    }

    /// Diagonal tensor with the given principal moments
    /// (`Ixx`, `Iyy`, `Izz`); off-diagonal products are zero.
    #[inline]
    pub fn from_principal(ixx: f64, iyy: f64, izz: f64) -> Self {
        Self::from_dmat3_unchecked(DMat3::from_diagonal(glam::DVec3::new(ixx, iyy, izz)))
    }

    /// Symmetric tensor built from JEOD's six-component layout:
    /// `Ixx, Iyy, Izz` on the diagonal; `Ixy, Ixz, Iyz` mirrored across.
    ///
    /// JEOD stores inertia products as `+Ixy = ∫ x·y dm` (no negation),
    /// matching the convention used in
    /// `models/dynamics/mass/src/mass_properties_inertia.cc`.
    #[inline]
    pub fn from_components(ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64) -> Self {
        // glam DMat3::from_cols is column-major: columns are x_axis, y_axis, z_axis.
        let m = DMat3::from_cols(
            glam::DVec3::new(ixx, ixy, ixz),
            glam::DVec3::new(ixy, iyy, iyz),
            glam::DVec3::new(ixz, iyz, izz),
        );
        Self::from_dmat3_unchecked(m)
    }

    /// Raw `DMat3` view in `kg·m²`. Value copy.
    #[inline]
    pub const fn as_dmat3(&self) -> DMat3 {
        self.value
    }

    /// Transpose. For a symmetric inertia tensor this is the identity.
    #[inline]
    pub fn transpose(&self) -> Self {
        Self::from_dmat3_unchecked(self.value.transpose())
    }

    /// Numeric similarity transform `T^T · I · T`. Matches JEOD's
    /// `transpose_transform_matrix` usage in `mass_properties.cc` for
    /// composing child inertia about a parent structural frame's axes.
    ///
    /// **The frame phantom `F` is intentionally unchanged.** This is
    /// the numeric kernel; frame-changing must go through an explicit
    /// retag (`InertiaTensor::<NewFrame>::from_dmat3_unchecked(...)`)
    /// or a future typed `FrameTransform`-style wrapper that updates
    /// the phantom coherently. Calling `transform` alone does not
    /// re-express the tensor in a different compile-time frame.
    #[inline]
    pub fn transform(&self, rot: &DMat3) -> Self {
        Self::from_dmat3_unchecked(rot.transpose() * self.value * *rot)
    }
}

// Phantom-only generic mirroring `Qty3` style — `#[derive]` would wrongly
// demand `F: Copy + Debug + PartialEq` even though `PhantomData<F>` is
// zero-sized regardless.
impl_copy_phantom!([F: Frame] InertiaTensor[F]);
impl_clone_phantom!([F: Frame] InertiaTensor[F]);
impl_debug_phantom!(
    [F: Frame] InertiaTensor[F],
    |this, f| write!(
        f,
        "InertiaTensor<{}>({:?})",
        core::any::type_name::<F>(),
        this.value
    )
);
impl_partial_eq_phantom!(
    [F: Frame] InertiaTensor[F],
    |this, other| this.value == other.value
);

impl<F: Frame> Default for InertiaTensor<F> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

// --- Arithmetic ----------------------------------------------------------

/// Component-wise addition. Used by parallel-axis composition: the caller
/// supplies the offset contribution (`m · ([d]² · I − d ⊗ d)`) as another
/// `InertiaTensor<F>` in the same frame and adds it.
impl<F: Frame> Add for InertiaTensor<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_dmat3_unchecked(self.value + rhs.value)
    }
}

impl<F: Frame> AddAssign for InertiaTensor<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<F: Frame> Sub for InertiaTensor<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_dmat3_unchecked(self.value - rhs.value)
    }
}

impl<F: Frame> SubAssign for InertiaTensor<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<F: Frame> Neg for InertiaTensor<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_dmat3_unchecked(-self.value)
    }
}

/// Scalar multiply on the right (`tensor * f64`). The left form
/// (`f64 * tensor`) is omitted — `Mul<InertiaTensor<F>> for f64` would
/// require an `F`-generic foreign-impl that the orphan rules permit but
/// which adds little ergonomic value for what's almost always a scalar
/// in user code.
impl<F: Frame> Mul<f64> for InertiaTensor<F> {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f64) -> Self {
        Self::from_dmat3_unchecked(self.value * scalar)
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "symmetry / diagonal-element tests assert bit-exact relations between literal-constructed components"
)]
mod tests {
    use super::*;
    use crate::frame::RootInertial;

    #[test]
    fn principal_constructor_is_diagonal() {
        let i = InertiaTensor::<RootInertial>::from_principal(2.0, 3.0, 5.0);
        let m = i.as_dmat3();
        assert_eq!(m.x_axis, glam::DVec3::new(2.0, 0.0, 0.0));
        assert_eq!(m.y_axis, glam::DVec3::new(0.0, 3.0, 0.0));
        assert_eq!(m.z_axis, glam::DVec3::new(0.0, 0.0, 5.0));
    }

    #[test]
    fn from_components_is_symmetric() {
        let i = InertiaTensor::<RootInertial>::from_components(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let m = i.as_dmat3();
        assert_eq!(m.col(0)[1], m.col(1)[0]);
        assert_eq!(m.col(0)[2], m.col(2)[0]);
        assert_eq!(m.col(1)[2], m.col(2)[1]);
    }

    #[test]
    fn add_is_componentwise() {
        let a = InertiaTensor::<RootInertial>::from_principal(1.0, 1.0, 1.0);
        let b = InertiaTensor::<RootInertial>::from_principal(2.0, 3.0, 4.0);
        assert_eq!(
            (a + b).as_dmat3(),
            DMat3::from_diagonal(glam::DVec3::new(3.0, 4.0, 5.0))
        );
    }

    #[test]
    fn scalar_mul_distributes() {
        let i = InertiaTensor::<RootInertial>::from_principal(1.0, 2.0, 3.0);
        let scaled = i * 2.0;
        assert_eq!(
            scaled.as_dmat3(),
            DMat3::from_diagonal(glam::DVec3::new(2.0, 4.0, 6.0))
        );
    }

    #[test]
    fn transform_with_identity_is_noop() {
        let i = InertiaTensor::<RootInertial>::from_components(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let rotated = i.transform(&DMat3::IDENTITY);
        assert_eq!(i, rotated);
    }
}
