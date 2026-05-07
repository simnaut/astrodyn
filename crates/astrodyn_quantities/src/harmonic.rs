//! `HarmonicDegree` — semantic newtype over `usize` for spherical-harmonic
//! degree / order indices.
//!
//! JEOD's gravity controls track four ordinal indices —
//! `degree`, `order`, `gradient_degree`, `gradient_order` — that select
//! how many spherical-harmonic terms to include when evaluating the
//! Gottlieb algorithm. They are unitless integers, but they are
//! *not* interchangeable with angular `Degree` (the uom angle unit) or
//! `Ratio`. Using a tagged newtype lets the compiler distinguish
//! `HarmonicDegree(8)` from `8.0_f64.deg()` or `Ratio::new::<ratio>(8.0)`
//! at typecheck time.
//!
//! The underlying integer type is `usize` rather than `u16`, matching
//! the existing field types in `astrodyn_gravity::gravity_controls`. This
//! keeps the migration churn-free at every call site.
//!
//! Cross-type confusion (passing an `Angle` or `Ratio` where a
//! `HarmonicDegree` is expected) is already prevented by the standard
//! Rust type-mismatch error, since `HarmonicDegree` is a distinct
//! struct rather than a `uom` Quantity newtype. Wrap with
//! `HarmonicDegree::new(n)` or `HarmonicDegree::from(n)` at call
//! sites that have a raw `usize`.

use core::fmt;

/// Spherical-harmonic degree or order index. Unitless ordinal.
///
/// `HarmonicDegree::default()` is `HarmonicDegree(0)` for parity with
/// JEOD's `GravityControl::default()` (which initializes its degree
/// fields to zero before the user explicitly sets them).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HarmonicDegree(pub usize);

impl HarmonicDegree {
    /// Construct from a raw `usize`.
    #[inline]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Read the underlying ordinal value.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl Default for HarmonicDegree {
    #[inline]
    fn default() -> Self {
        Self(0)
    }
}

impl fmt::Debug for HarmonicDegree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HarmonicDegree({})", self.0)
    }
}

impl fmt::Display for HarmonicDegree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for HarmonicDegree {
    #[inline]
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<HarmonicDegree> for usize {
    #[inline]
    fn from(value: HarmonicDegree) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_zero() {
        assert_eq!(HarmonicDegree::default(), HarmonicDegree::new(0));
        assert_eq!(HarmonicDegree::default().get(), 0);
    }

    #[test]
    fn ordering_is_natural() {
        assert!(HarmonicDegree(3) < HarmonicDegree(8));
        assert!(HarmonicDegree(70) > HarmonicDegree(8));
        assert_eq!(HarmonicDegree(8), HarmonicDegree(8));
    }

    #[test]
    fn round_trip_through_usize() {
        let d = HarmonicDegree::from(8usize);
        let back: usize = d.into();
        assert_eq!(back, 8);
    }

    #[test]
    fn debug_and_display_show_inner_value() {
        let d = HarmonicDegree(70);
        assert_eq!(format!("{d}"), "70");
        assert_eq!(format!("{d:?}"), "HarmonicDegree(70)");
    }
}
