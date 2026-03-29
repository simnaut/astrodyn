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
}

#[cfg(test)]
mod tests {
    use crate::presets::*;

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
}
