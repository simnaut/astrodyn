//! Reference-ellipsoid parameter block for a planetary body.
//!
//! Ports the `Planet` struct in
//! [`models/environment/planet/include/planet.hh`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/environment/planet/include/planet.hh)
//! from JEOD v5.4.0. Stores `name`, `mu` (m³/s²), `r_eq` (m), `r_pol` (m),
//! and `flat_coeff` along with derived helpers (`flat_inv`,
//! `e_ellipsoid`, `e_ellip_sq`) plus typed accessors that wrap the raw
//! `f64` fields as `jeod_quantities` quantities.

/// Planetary shape parameters (reference ellipsoid).
///
/// Mirrors JEOD's `Planet` struct from `planet.hh`. The `mu` field stores the
/// geodetic standard gravitational parameter (e.g., WGS84 for Earth). Gravity
/// models carry their own `mu` in `GravitySource` which may differ slightly.
#[derive(Debug, Clone, Copy)]
pub struct PlanetShape {
    /// Planet name.
    pub name: &'static str,
    /// Gravitational parameter (m^3/s^2).
    pub mu: f64,
    /// Mean equatorial radius (m).
    pub r_eq: f64,
    /// Mean polar radius (m).
    pub r_pol: f64,
    /// Flattening coefficient: f = (r_eq - r_pol) / r_eq.
    pub flat_coeff: f64,
}

impl PlanetShape {
    /// Inverse flattening (e.g., 298.257223563 for Earth).
    pub fn flat_inv(&self) -> f64 {
        1.0 / self.flat_coeff
    }

    /// Ellipsoid eccentricity: e = sqrt(2f - f^2).
    pub fn e_ellipsoid(&self) -> f64 {
        let f = self.flat_coeff;
        (2.0 * f - f * f).sqrt()
    }

    /// Square of ellipsoid eccentricity.
    pub fn e_ellip_sq(&self) -> f64 {
        let f = self.flat_coeff;
        2.0 * f - f * f
    }

    /// Gravitational parameter as a typed
    /// `GravParam<jeod_quantities::frame::SelfPlanet>` (m³/s²).
    ///
    /// Returns the planet-erased
    /// ([`jeod_quantities::frame::SelfPlanet`]) form because
    /// `PlanetShape` is the dynamic per-planet record carried by the
    /// runner — the planet identity is determined by which entity
    /// holds the shape, not by the static type. Mission code that
    /// knows the planet at compile time can either reach for the
    /// curated `mu_*()` constants in `jeod_sim::recipes::constants`
    /// (which return planet-pinned `GravParam<P>` values) or relabel
    /// at the call site via `mu_typed().relabel::<P>()`.
    pub fn mu_typed(&self) -> jeod_quantities::dims::GravParam<jeod_quantities::frame::SelfPlanet> {
        jeod_quantities::dims::GravParam::<jeod_quantities::frame::SelfPlanet>::from_si(self.mu)
    }

    /// Equatorial radius as a typed `uom::si::f64::Length` (meters).
    ///
    /// Additive typed accessor introduced by Phase 1 of the type-system
    /// refactor (issue #101). Wraps the existing `r_eq` f64 field; the
    /// underlying field is unchanged.
    pub fn r_eq_typed(&self) -> uom::si::f64::Length {
        use jeod_quantities::ext::F64Ext;
        self.r_eq.m()
    }

    /// Polar radius as a typed `uom::si::f64::Length` (meters).
    ///
    /// Additive typed accessor introduced by Phase 1 of the type-system
    /// refactor (issue #101). Wraps the existing `r_pol` f64 field; the
    /// underlying field is unchanged.
    pub fn r_pol_typed(&self) -> uom::si::f64::Length {
        use jeod_quantities::ext::F64Ext;
        self.r_pol.m()
    }
}

#[cfg(test)]
mod tests {
    use crate::presets::*;

    #[test]
    fn earth_mu_typed_matches_f64_field() {
        // `GravParam` stores value in base SI (m³/s²) via `.value`.
        let mu = EARTH.mu_typed();
        assert_eq!(mu.value, EARTH.mu);
    }

    #[test]
    fn earth_r_eq_typed_matches_f64_field() {
        use uom::si::length::meter;
        let r_eq = EARTH.r_eq_typed();
        assert_eq!(r_eq.get::<meter>(), EARTH.r_eq);
    }

    #[test]
    fn earth_r_pol_typed_matches_f64_field() {
        use uom::si::length::meter;
        let r_pol = EARTH.r_pol_typed();
        assert_eq!(r_pol.get::<meter>(), EARTH.r_pol);
    }

    #[test]
    fn polar_radius_consistent_with_flattening() {
        for planet in [EARTH, MOON, SUN, MARS] {
            let expected_r_pol = planet.r_eq * (1.0 - planet.flat_coeff);
            let err = (planet.r_pol - expected_r_pol).abs();
            assert!(
                err < 1.0, // < 1 m tolerance for rounding
                "{}: r_pol={} vs r_eq*(1-f)={}, err={}",
                planet.name,
                planet.r_pol,
                expected_r_pol,
                err
            );
        }
    }

    #[test]
    fn all_values_positive() {
        for planet in [EARTH, MOON, SUN, MARS] {
            assert!(planet.mu > 0.0, "{}: mu must be positive", planet.name);
            assert!(planet.r_eq > 0.0, "{}: r_eq must be positive", planet.name);
            assert!(
                planet.r_pol > 0.0,
                "{}: r_pol must be positive",
                planet.name
            );
            assert!(
                planet.flat_coeff > 0.0,
                "{}: flattening must be positive",
                planet.name
            );
            assert!(
                planet.flat_coeff < 1.0,
                "{}: flattening must be < 1",
                planet.name
            );
            assert!(
                planet.r_eq >= planet.r_pol,
                "{}: r_eq must be >= r_pol",
                planet.name
            );
        }
    }

    #[test]
    fn earth_inverse_flattening() {
        let inv_f = EARTH.flat_inv();
        assert!(
            (inv_f - 298.257223563).abs() < 1e-6,
            "Earth 1/f: expected 298.257223563, got {}",
            inv_f
        );
    }

    #[test]
    fn eccentricity_positive() {
        for planet in [EARTH, MOON, SUN, MARS] {
            let e = planet.e_ellipsoid();
            assert!(e > 0.0 && e < 1.0, "{}: e={}", planet.name, e);
        }
    }

    // -----------------------------------------------------------------
    // PR6 (#219): basic accessor coverage. The pre-existing tests above
    // cover typed accessors and aggregate sanity; the cases below pin
    // the field-level semantics of `PlanetShape` for a hand-constructed
    // value, so that downstream callers can rely on the struct surface
    // without depending on the `presets` module.
    // -----------------------------------------------------------------

    use crate::planet::PlanetShape;

    /// Hand-built minimal planet, so the test does not depend on preset
    /// constants (which are exercised separately).
    const TEST_SHAPE: PlanetShape = PlanetShape {
        name: "Testopia",
        mu: 1.5e14,
        r_eq: 6_400_000.0,
        r_pol: 6_400_000.0 * (1.0 - 1.0 / 300.0),
        flat_coeff: 1.0 / 300.0,
    };

    #[test]
    fn shape_struct_field_access_round_trip() {
        // The struct has no methods that mutate state — direct field
        // access must return the literal we stored.
        assert_eq!(TEST_SHAPE.name, "Testopia");
        assert_eq!(TEST_SHAPE.mu, 1.5e14);
        assert_eq!(TEST_SHAPE.r_eq, 6_400_000.0);
        assert_eq!(TEST_SHAPE.flat_coeff, 1.0 / 300.0);
    }

    #[test]
    fn radius_accessors_distinct_and_consistent() {
        // r_eq > r_pol for any oblate body (f > 0). The const above has
        // flat_coeff = 1/300 > 0, so the relation must hold for every
        // preset and our hand-built shape. We reach for the values via
        // a non-const accessor expression so clippy doesn't fold the
        // comparison to a literal.
        for planet in [TEST_SHAPE, EARTH, MOON, SUN, MARS] {
            assert!(
                planet.r_eq > planet.r_pol,
                "{}: r_eq={} not > r_pol={}",
                planet.name,
                planet.r_eq,
                planet.r_pol
            );
            let derived_pol = planet.r_eq * (1.0 - planet.flat_coeff);
            let err = (derived_pol - planet.r_pol).abs();
            assert!(
                err < 1.0,
                "{}: r_pol drifted by {err} m from r_eq*(1-f)",
                planet.name
            );
        }
    }

    #[test]
    fn flattening_inverse_round_trip() {
        // flat_inv() must invert flat_coeff exactly (within f64 ULP).
        let f = TEST_SHAPE.flat_coeff;
        let inv = TEST_SHAPE.flat_inv();
        assert!(
            (inv * f - 1.0).abs() < 1e-14,
            "flat_inv({f}) * f != 1: got {}",
            inv * f
        );
    }

    #[test]
    fn mu_accessor_matches_typed_and_preserves_value() {
        // The typed getter must round-trip the raw f64 mu without loss.
        // GravParam stores its value in base SI (m³/s²).
        for planet in [TEST_SHAPE, EARTH, MOON, SUN, MARS] {
            assert_eq!(planet.mu_typed().value, planet.mu);
        }
    }

    #[test]
    fn eccentricity_squared_matches_definition() {
        // e_ellip_sq = 2f - f^2; e_ellipsoid is its sqrt.
        for planet in [TEST_SHAPE, EARTH, MOON, SUN, MARS] {
            let f = planet.flat_coeff;
            let expected = 2.0 * f - f * f;
            assert!((planet.e_ellip_sq() - expected).abs() < 1e-15);
            let e = planet.e_ellipsoid();
            assert!((e * e - planet.e_ellip_sq()).abs() < 1e-15);
        }
    }

    #[test]
    fn shape_struct_is_copy() {
        // The Copy bound matters for ergonomic use as a const — assert
        // it implicitly via a copy through a function boundary.
        fn take_by_value(s: PlanetShape) -> f64 {
            s.r_eq
        }
        let s = TEST_SHAPE;
        assert_eq!(take_by_value(s), TEST_SHAPE.r_eq);
        // Verify the original is still usable post-copy.
        assert_eq!(s.name, "Testopia");
    }
}
