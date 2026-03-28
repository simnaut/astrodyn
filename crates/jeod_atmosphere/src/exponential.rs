//! Exponential atmosphere model.
//!
//! Simple density model: `rho = rho_0 * exp(-(h - h_0) / H)`
//!
//! Useful as a fallback and for testing. Not physically accurate above ~100 km
//! where the atmosphere transitions from well-mixed to diffusive equilibrium.

use glam::DVec3;

use crate::AtmosphericState;

/// Exponential atmosphere model parameters.
#[derive(Debug, Clone, Copy)]
pub struct ExponentialAtmosphere {
    /// Reference density at reference altitude (kg/m^3).
    pub rho_0: f64,
    /// Reference altitude (m).
    pub h_0: f64,
    /// Scale height (m).
    pub scale_height: f64,
}

impl Default for ExponentialAtmosphere {
    /// Default: Earth sea-level values.
    fn default() -> Self {
        Self {
            rho_0: 1.225,
            h_0: 0.0,
            scale_height: 8500.0,
        }
    }
}

impl ExponentialAtmosphere {
    /// Compute atmospheric state at a given geodetic altitude.
    ///
    /// Temperature and pressure are set to zero (not modeled).
    /// Wind is set to zero (no co-rotation model).
    ///
    /// # Arguments
    /// * `altitude` - Geodetic altitude above the reference ellipsoid (m)
    pub fn density(&self, altitude: f64) -> AtmosphericState {
        // Cap altitude to avoid numerical overflow in exp() for deeply
        // sub-surface altitudes. Threshold accounts for h_0.
        let min_altitude = self.h_0 - self.scale_height;
        let effective_altitude = altitude.max(min_altitude);
        let density = self.rho_0 * (-(effective_altitude - self.h_0) / self.scale_height).exp();

        AtmosphericState {
            density,
            temperature: 0.0,
            pressure: 0.0,
            wind: DVec3::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sea_level_density() {
        let atmos = ExponentialAtmosphere::default();
        let state = atmos.density(0.0);
        assert!(
            (state.density - 1.225).abs() < 1e-10,
            "Sea level density should be 1.225 kg/m^3, got {}",
            state.density
        );
    }

    #[test]
    fn one_scale_height() {
        let atmos = ExponentialAtmosphere::default();
        let state = atmos.density(8500.0);
        let expected = 1.225 * (-1.0_f64).exp();
        assert!(
            (state.density - expected).abs() / expected < 1e-12,
            "Density at one scale height: expected {expected}, got {}",
            state.density
        );
    }

    #[test]
    fn hundred_km() {
        let atmos = ExponentialAtmosphere::default();
        let state = atmos.density(100_000.0);
        // At 100 km, exp(-100000/8500) ~ exp(-11.76) ~ 7.7e-6
        // rho ~ 1.225 * 7.7e-6 ~ 9.4e-6 kg/m^3
        // Order of magnitude: ~1e-5 to 1e-6
        assert!(
            state.density > 1e-7 && state.density < 1e-4,
            "Density at 100 km should be ~1e-5, got {}",
            state.density
        );
    }

    #[test]
    fn four_hundred_km() {
        let atmos = ExponentialAtmosphere::default();
        let state = atmos.density(400_000.0);
        // Exponential model gives very low density at 400 km
        // Real atmosphere is ~1e-12 at 400 km; exponential overpredicts decay
        assert!(
            state.density > 0.0 && state.density < 1e-10,
            "Density at 400 km should be very small, got {}",
            state.density
        );
    }

    #[test]
    fn density_decreases_with_altitude() {
        let atmos = ExponentialAtmosphere::default();
        let d0 = atmos.density(0.0).density;
        let d1 = atmos.density(10_000.0).density;
        let d2 = atmos.density(100_000.0).density;
        assert!(d0 > d1, "Density should decrease with altitude");
        assert!(d1 > d2, "Density should decrease with altitude");
    }

    #[test]
    fn wind_is_zero() {
        let atmos = ExponentialAtmosphere::default();
        assert_eq!(atmos.density(400_000.0).wind, DVec3::ZERO);
    }
}
