//! `Qty3<D, F>` — a frame-tagged, dimension-typed 3-vector.

use core::marker::PhantomData;

use glam::DVec3;
use uom::si::{Dimension, Quantity, SI};

use crate::frame::Frame;

/// Componentwise 3-vector: each of `x`, `y`, `z` carries the dimension `D`,
/// and the whole vector carries the frame tag `F`.
///
/// Layout: `#[repr(C)]` with three contiguous `f64`s. `raw_si()` and
/// `from_raw_si` are layout-compatible with `DVec3`; see the `layout`
/// unit tests for the static size/align assertions.
#[repr(C)]
pub struct Qty3<D: ?Sized + Dimension, F: Frame> {
    /// X component, dimension `D`, in SI base units.
    pub x: Quantity<D, SI<f64>, f64>,
    /// Y component, dimension `D`, in SI base units.
    pub y: Quantity<D, SI<f64>, f64>,
    /// Z component, dimension `D`, in SI base units.
    pub z: Quantity<D, SI<f64>, f64>,
    _f: PhantomData<F>,
}

impl<D: ?Sized + Dimension, F: Frame> Qty3<D, F> {
    /// Construct from three scalar quantities of dimension `D`.
    #[inline]
    pub const fn new(
        x: Quantity<D, SI<f64>, f64>,
        y: Quantity<D, SI<f64>, f64>,
        z: Quantity<D, SI<f64>, f64>,
    ) -> Self {
        Self {
            x,
            y,
            z,
            _f: PhantomData,
        }
    }

    /// Raw SI-unit `DVec3` view.
    ///
    /// This is a value copy — but since each `Quantity<D, SI<f64>, f64>` is
    /// layout-equivalent to `f64` (all type parameters land in `PhantomData`
    /// fields which are zero-sized), the compiler lowers this to three field
    /// moves with no unit-conversion arithmetic. Inlined and monomorphic.
    #[inline(always)]
    pub fn raw_si(&self) -> DVec3 {
        DVec3::new(self.x.value, self.y.value, self.z.value)
    }

    /// Inverse of [`Self::raw_si`]: wrap a `DVec3` of SI-unit values as a typed
    /// [`Qty3`] in frame `F` and dimension `D`.
    ///
    /// The caller is responsible for providing values that are already in the
    /// dimension's base SI unit — this constructor performs no conversion.
    #[inline(always)]
    pub fn from_raw_si(v: DVec3) -> Self {
        Self {
            x: Quantity {
                dimension: PhantomData,
                units: PhantomData,
                value: v.x,
            },
            y: Quantity {
                dimension: PhantomData,
                units: PhantomData,
                value: v.y,
            },
            z: Quantity {
                dimension: PhantomData,
                units: PhantomData,
                value: v.z,
            },
            _f: PhantomData,
        }
    }

    /// Zero vector in frame `F`, dimension `D`.
    #[inline]
    pub fn zero() -> Self {
        Self::from_raw_si(DVec3::ZERO)
    }
}

// Manual Copy/Clone/PartialEq to avoid the derive macro demanding `D: Copy`.
impl<D: ?Sized + Dimension, F: Frame> Copy for Qty3<D, F> {}

impl<D: ?Sized + Dimension, F: Frame> Clone for Qty3<D, F> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<D: ?Sized + Dimension, F: Frame> core::fmt::Debug for Qty3<D, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `F::NAME` is the *kind* of the frame (e.g. "BodyFrame"). For
        // vehicle- or planet-parameterized frames that alone can't
        // distinguish `BodyFrame<Iss>` from `BodyFrame<Mir>`. Use
        // `type_name::<F>()` so the output carries the full phantom tag
        // — the cost is a single opaque string, no allocations.
        write!(
            f,
            "Qty3<{}>({}, {}, {})",
            core::any::type_name::<F>(),
            self.x.value,
            self.y.value,
            self.z.value
        )
    }
}

impl<D: ?Sized + Dimension, F: Frame> PartialEq for Qty3<D, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x.value == other.x.value
            && self.y.value == other.y.value
            && self.z.value == other.z.value
    }
}

/// `Default` returns the zero vector of dimension `D` in frame `F`. Manual
/// impl rather than derive, because `derive(Default)` would demand
/// `D: Default` (it isn't a real value, just a phantom dimension).
impl<D: ?Sized + Dimension, F: Frame> Default for Qty3<D, F> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}
