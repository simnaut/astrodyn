//! Arithmetic operators on `Qty3`.
//!
//! Rules:
//! - `Add`/`Sub`/`Neg` require matching dimension **and** matching frame.
//! - `Mul`/`Div` by a scalar `Quantity` of any dimension `D2` produces
//!   `Qty3<D·D2, F>` / `Qty3<D÷D2, F>`, preserving the frame.
//! - `.dot(b)` on `Qty3<D1, F> · Qty3<D2, F>` returns a *scalar* of dimension
//!   `D1·D2`.
//! - `.cross(b)` returns `Qty3<D1·D2, F>`.
//! - `.magnitude()` returns the scalar magnitude of dimension `D`.

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

use uom::si::{Dimension, Quantity, ISQ, SI};
use uom::typenum::{Diff, Integer, Sum};

use crate::diagnostics::CompatibleFrames;
use crate::frame::Frame;
use crate::qty3::Qty3;

// ---- Add / Sub / Neg ----
//
// `Add` and `Sub` are parameterized over distinct LHS/RHS frames and then
// constrained by `(): CompatibleFrames<Fl, Fr>`, whose blanket impl only
// covers `CompatibleFrames<F, F>`. When the frames mismatch, that bound
// fails and the `#[diagnostic::on_unimplemented]` attribute on
// `CompatibleFrames` surfaces the tailored error message.

impl<D: ?Sized + Dimension, Fl: Frame, Fr: Frame> Add<Qty3<D, Fr>> for Qty3<D, Fl>
where
    (): CompatibleFrames<Fl, Fr>,
    Quantity<D, SI<f64>, f64>: Add<Output = Quantity<D, SI<f64>, f64>>,
{
    type Output = Qty3<D, Fl>;

    #[inline]
    fn add(self, rhs: Qty3<D, Fr>) -> Self::Output {
        // Fl = Fr at this impl site (enforced by the CompatibleFrames bound),
        // but the compiler still sees two nominally-distinct types. Go through
        // the raw-SI representation to sidestep that and keep this a safe,
        // unsafe-code-free path.
        Qty3::from_raw_si(self.raw_si() + rhs.raw_si())
    }
}

impl<D: ?Sized + Dimension, Fl: Frame, Fr: Frame> Sub<Qty3<D, Fr>> for Qty3<D, Fl>
where
    (): CompatibleFrames<Fl, Fr>,
    Quantity<D, SI<f64>, f64>: Sub<Output = Quantity<D, SI<f64>, f64>>,
{
    type Output = Qty3<D, Fl>;

    #[inline]
    fn sub(self, rhs: Qty3<D, Fr>) -> Self::Output {
        Qty3::from_raw_si(self.raw_si() - rhs.raw_si())
    }
}

impl<D: ?Sized + Dimension, F: Frame> Neg for Qty3<D, F>
where
    Quantity<D, SI<f64>, f64>: Neg<Output = Quantity<D, SI<f64>, f64>>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Qty3::new(-self.x, -self.y, -self.z)
    }
}

// ---- Scalar multiply / divide by f64 (dimension unchanged) ----

impl<D: ?Sized + Dimension, F: Frame> Mul<f64> for Qty3<D, F>
where
    Quantity<D, SI<f64>, f64>: Mul<f64, Output = Quantity<D, SI<f64>, f64>>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Qty3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl<D: ?Sized + Dimension, F: Frame> Div<f64> for Qty3<D, F>
where
    Quantity<D, SI<f64>, f64>: Div<f64, Output = Quantity<D, SI<f64>, f64>>,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Qty3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

// ---- Cross-dimension multiply / divide by scalar `Quantity` ----
//
// Multiplying a `Qty3<Dl, F>` by a scalar of dimension `Dr` yields a
// `Qty3<Dl · Dr, F>`. Division yields `Qty3<Dl / Dr, F>`. We express this
// by invoking uom's built-in dimensional arithmetic on the scalar
// components and letting type inference unify the output type. The output
// dimension is expressed via `typenum::Sum` (for multiply) or `Diff` (for
// divide) over each base-unit exponent.

impl<Ll, Ml, Tl, Il, Thl, Nl, Jl, Lr, Mr, Tr, Ir, Thr, Nr, Jr, F>
    Mul<Quantity<ISQ<Lr, Mr, Tr, Ir, Thr, Nr, Jr>, SI<f64>, f64>>
    for Qty3<ISQ<Ll, Ml, Tl, Il, Thl, Nl, Jl>, F>
where
    Ll: Integer + Add<Lr>,
    Ml: Integer + Add<Mr>,
    Tl: Integer + Add<Tr>,
    Il: Integer + Add<Ir>,
    Thl: Integer + Add<Thr>,
    Nl: Integer + Add<Nr>,
    Jl: Integer + Add<Jr>,
    Lr: Integer,
    Mr: Integer,
    Tr: Integer,
    Ir: Integer,
    Thr: Integer,
    Nr: Integer,
    Jr: Integer,
    Sum<Ll, Lr>: Integer,
    Sum<Ml, Mr>: Integer,
    Sum<Tl, Tr>: Integer,
    Sum<Il, Ir>: Integer,
    Sum<Thl, Thr>: Integer,
    Sum<Nl, Nr>: Integer,
    Sum<Jl, Jr>: Integer,
    F: Frame,
{
    type Output = Qty3<
        ISQ<
            Sum<Ll, Lr>,
            Sum<Ml, Mr>,
            Sum<Tl, Tr>,
            Sum<Il, Ir>,
            Sum<Thl, Thr>,
            Sum<Nl, Nr>,
            Sum<Jl, Jr>,
        >,
        F,
    >;

    #[inline]
    fn mul(self, rhs: Quantity<ISQ<Lr, Mr, Tr, Ir, Thr, Nr, Jr>, SI<f64>, f64>) -> Self::Output {
        // Multiply component-wise: uom's scalar Mul already yields the
        // right output dimension, which matches Self::Output's dimension
        // by construction.
        Qty3::<_, F>::from_raw_si(glam::DVec3::new(
            (self.x * rhs).value,
            (self.y * rhs).value,
            (self.z * rhs).value,
        ))
    }
}

// ---- Div<Quantity<ISQ<...>>>: Qty3<Dl> / Quantity<Dr> = Qty3<Dl - Dr> ----

impl<Ll, Ml, Tl, Il, Thl, Nl, Jl, Lr, Mr, Tr, Ir, Thr, Nr, Jr, F>
    Div<Quantity<ISQ<Lr, Mr, Tr, Ir, Thr, Nr, Jr>, SI<f64>, f64>>
    for Qty3<ISQ<Ll, Ml, Tl, Il, Thl, Nl, Jl>, F>
where
    Ll: Integer + core::ops::Sub<Lr>,
    Ml: Integer + core::ops::Sub<Mr>,
    Tl: Integer + core::ops::Sub<Tr>,
    Il: Integer + core::ops::Sub<Ir>,
    Thl: Integer + core::ops::Sub<Thr>,
    Nl: Integer + core::ops::Sub<Nr>,
    Jl: Integer + core::ops::Sub<Jr>,
    Lr: Integer,
    Mr: Integer,
    Tr: Integer,
    Ir: Integer,
    Thr: Integer,
    Nr: Integer,
    Jr: Integer,
    Diff<Ll, Lr>: Integer,
    Diff<Ml, Mr>: Integer,
    Diff<Tl, Tr>: Integer,
    Diff<Il, Ir>: Integer,
    Diff<Thl, Thr>: Integer,
    Diff<Nl, Nr>: Integer,
    Diff<Jl, Jr>: Integer,
    F: Frame,
{
    type Output = Qty3<
        ISQ<
            Diff<Ll, Lr>,
            Diff<Ml, Mr>,
            Diff<Tl, Tr>,
            Diff<Il, Ir>,
            Diff<Thl, Thr>,
            Diff<Nl, Nr>,
            Diff<Jl, Jr>,
        >,
        F,
    >;

    #[inline]
    fn div(self, rhs: Quantity<ISQ<Lr, Mr, Tr, Ir, Thr, Nr, Jr>, SI<f64>, f64>) -> Self::Output {
        Qty3::<_, F>::from_raw_si(glam::DVec3::new(
            (self.x / rhs).value,
            (self.y / rhs).value,
            (self.z / rhs).value,
        ))
    }
}

// ---- dot / cross / magnitude ----

impl<Ll, Ml, Tl, Il, Thl, Nl, Jl, F> Qty3<ISQ<Ll, Ml, Tl, Il, Thl, Nl, Jl>, F>
where
    Ll: Integer,
    Ml: Integer,
    Tl: Integer,
    Il: Integer,
    Thl: Integer,
    Nl: Integer,
    Jl: Integer,
    F: Frame,
{
    /// Dot product. Dimension of the output is the product of the two input
    /// dimensions: `Position · Velocity → Quantity<m²/s>` etc.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn dot<Lr, Mr, Tr, Ir, Thr, Nr, Jr>(
        &self,
        rhs: &Qty3<ISQ<Lr, Mr, Tr, Ir, Thr, Nr, Jr>, F>,
    ) -> Quantity<
        ISQ<
            Sum<Ll, Lr>,
            Sum<Ml, Mr>,
            Sum<Tl, Tr>,
            Sum<Il, Ir>,
            Sum<Thl, Thr>,
            Sum<Nl, Nr>,
            Sum<Jl, Jr>,
        >,
        SI<f64>,
        f64,
    >
    where
        Ll: Add<Lr>,
        Ml: Add<Mr>,
        Tl: Add<Tr>,
        Il: Add<Ir>,
        Thl: Add<Thr>,
        Nl: Add<Nr>,
        Jl: Add<Jr>,
        Lr: Integer,
        Mr: Integer,
        Tr: Integer,
        Ir: Integer,
        Thr: Integer,
        Nr: Integer,
        Jr: Integer,
        Sum<Ll, Lr>: Integer,
        Sum<Ml, Mr>: Integer,
        Sum<Tl, Tr>: Integer,
        Sum<Il, Ir>: Integer,
        Sum<Thl, Thr>: Integer,
        Sum<Nl, Nr>: Integer,
        Sum<Jl, Jr>: Integer,
    {
        Quantity {
            dimension: PhantomData,
            units: PhantomData,
            value: self.raw_si().dot(rhs.raw_si()),
        }
    }

    /// Cross product. Output is a 3-vector in the same frame with dimension
    /// equal to the product of the two input dimensions.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn cross<Lr, Mr, Tr, Ir, Thr, Nr, Jr>(
        &self,
        rhs: &Qty3<ISQ<Lr, Mr, Tr, Ir, Thr, Nr, Jr>, F>,
    ) -> Qty3<
        ISQ<
            Sum<Ll, Lr>,
            Sum<Ml, Mr>,
            Sum<Tl, Tr>,
            Sum<Il, Ir>,
            Sum<Thl, Thr>,
            Sum<Nl, Nr>,
            Sum<Jl, Jr>,
        >,
        F,
    >
    where
        Ll: Add<Lr>,
        Ml: Add<Mr>,
        Tl: Add<Tr>,
        Il: Add<Ir>,
        Thl: Add<Thr>,
        Nl: Add<Nr>,
        Jl: Add<Jr>,
        Lr: Integer,
        Mr: Integer,
        Tr: Integer,
        Ir: Integer,
        Thr: Integer,
        Nr: Integer,
        Jr: Integer,
        Sum<Ll, Lr>: Integer,
        Sum<Ml, Mr>: Integer,
        Sum<Tl, Tr>: Integer,
        Sum<Il, Ir>: Integer,
        Sum<Thl, Thr>: Integer,
        Sum<Nl, Nr>: Integer,
        Sum<Jl, Jr>: Integer,
    {
        Qty3::from_raw_si(self.raw_si().cross(rhs.raw_si()))
    }
}

impl<D: ?Sized + Dimension, F: Frame> Qty3<D, F> {
    /// Euclidean magnitude, same dimension as `self`.
    ///
    /// Implementation: sqrt over the raw SI value and rewrap with `D`. The
    /// squared sum is a `Quantity<D²>` internally, but we project back to
    /// `D` via the SI scalar (sqrt of `D²` = `D`), so no extra bound is
    /// needed at the trait level.
    #[inline]
    pub fn magnitude(&self) -> Quantity<D, SI<f64>, f64> {
        let v = self.raw_si();
        let raw = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
        Quantity {
            dimension: PhantomData,
            units: PhantomData,
            value: raw,
        }
    }
}

// ---- Velocity × Time → Position (and Acceleration × Time → Velocity) ----
//
// These follow automatically from the generic `Mul<Quantity<_>>` impl above,
// so no additional code is required here. See `tests/qty3_ops.rs` for
// verification.

#[cfg(test)]
mod tests {
    use crate::aliases::*;
    use crate::frame::Inertial;
    use uom::si::f64::Length;
    use uom::si::length::meter;

    fn pos_inertial(x: f64, y: f64, z: f64) -> Position<Inertial> {
        Position::<Inertial>::new(
            Length::new::<meter>(x),
            Length::new::<meter>(y),
            Length::new::<meter>(z),
        )
    }

    #[test]
    fn add_sub_neg() {
        let a = pos_inertial(1.0, 2.0, 3.0);
        let b = pos_inertial(4.0, 5.0, 6.0);
        let sum = a + b;
        let diff = a - b;
        let neg = -a;
        assert_eq!(sum.raw_si(), glam::DVec3::new(5.0, 7.0, 9.0));
        assert_eq!(diff.raw_si(), glam::DVec3::new(-3.0, -3.0, -3.0));
        assert_eq!(neg.raw_si(), glam::DVec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn scalar_mul_div() {
        let a = pos_inertial(2.0, 4.0, 6.0);
        assert_eq!((a * 2.5).raw_si(), glam::DVec3::new(5.0, 10.0, 15.0));
        assert_eq!((a / 2.0).raw_si(), glam::DVec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn magnitude_is_euclidean() {
        let a = pos_inertial(3.0, 4.0, 0.0);
        assert!((a.magnitude().value - 5.0).abs() < 1e-12);
    }
}
