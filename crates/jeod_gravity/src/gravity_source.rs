use crate::spherical_harmonics_gravity_source::SphericalHarmonicsData;
use jeod_quantities::dims::GravParam;

#[derive(Debug, Clone)]
pub struct GravitySource {
    pub mu: f64, // gravitational parameter, m^3/s^2
    pub model: GravityModel,
}

#[derive(Debug, Clone)]
pub enum GravityModel {
    PointMass,
    SphericalHarmonics(Box<SphericalHarmonicsData>),
}

/// Typed sibling of [`GravitySource`].
///
/// `mu` carries the [`GravParam`] dimensional type (`m³/s²`) instead
/// of bare `f64`. The model variant is unchanged — `GravityModel` is
/// an enum whose data layout the type system has nothing to add to.
#[derive(Debug, Clone)]
pub struct GravitySourceTyped {
    /// Gravitational parameter μ.
    pub mu: GravParam,
    /// Gravity model variant.
    pub model: GravityModel,
}

impl GravitySourceTyped {
    /// Drop the dimension annotation and emit the untyped storage form.
    /// Numeric value (`m³/s²`) is preserved exactly.
    ///
    /// **Performance**: this clones [`GravityModel`], which for the
    /// `SphericalHarmonics(Box<…>)` variant deep-clones the
    /// coefficient and Gottlieb helper arrays (potentially MB-scale
    /// for high-degree models). For one-shot conversions, prefer
    /// [`Self::into_untyped`], which consumes `self` and moves the
    /// boxed model.
    #[inline]
    pub fn to_untyped(&self) -> GravitySource {
        GravitySource {
            mu: self.mu.value,
            model: self.model.clone(),
        }
    }

    /// Consuming sibling of [`Self::to_untyped`]: moves the inner
    /// [`GravityModel`] (including any `Box<SphericalHarmonicsData>`)
    /// without cloning the coefficient arrays.
    #[inline]
    pub fn into_untyped(self) -> GravitySource {
        GravitySource {
            mu: self.mu.value,
            model: self.model,
        }
    }

    /// Wrap an untyped [`GravitySource`] as typed. **The caller asserts**
    /// the `mu` field carries SI base units (m³/s²).
    ///
    /// Same performance caveat as [`Self::to_untyped`]: clones
    /// `s.model`. Use [`Self::from_untyped`] for a consuming
    /// conversion when `s` can be moved.
    #[inline]
    pub fn from_untyped_unchecked(s: &GravitySource) -> Self {
        Self {
            mu: GravParam {
                dimension: core::marker::PhantomData,
                units: core::marker::PhantomData,
                value: s.mu,
            },
            model: s.model.clone(),
        }
    }

    /// Consuming sibling of [`Self::from_untyped_unchecked`].
    #[inline]
    pub fn from_untyped(s: GravitySource) -> Self {
        Self {
            mu: GravParam {
                dimension: core::marker::PhantomData,
                units: core::marker::PhantomData,
                value: s.mu,
            },
            model: s.model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_quantities::ext::F64Ext;

    #[test]
    fn typed_round_trip_preserves_mu() {
        let earth_mu = 3.986_004_415e14;
        let untyped = GravitySource {
            mu: earth_mu,
            model: GravityModel::PointMass,
        };
        let typed = GravitySourceTyped::from_untyped_unchecked(&untyped);
        assert_eq!(typed.mu.value, earth_mu);

        let back = typed.to_untyped();
        assert_eq!(back.mu, earth_mu);
    }

    #[test]
    fn typed_constructor_via_f64_ext_works() {
        let earth_mu = 3.986_004_415e14;
        let typed = GravitySourceTyped {
            mu: earth_mu.m3_per_s2(),
            model: GravityModel::PointMass,
        };
        assert_eq!(typed.to_untyped().mu, earth_mu);
    }
}
