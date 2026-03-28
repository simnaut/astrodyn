pub mod exponential;
pub mod met;

use glam::DVec3;

/// Atmospheric state at a given position.
///
/// Output of an atmosphere model evaluation. All quantities are in SI units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereState {
    /// Atmospheric density in kg/m^3.
    pub density: f64,
    /// Temperature in K.
    pub temperature: f64,
    /// Pressure in N/m^2 (Pa).
    pub pressure: f64,
    /// Wind velocity in m/s, expressed in the inertial frame.
    pub wind: DVec3,
}

impl Default for AtmosphereState {
    fn default() -> Self {
        Self {
            density: 0.0,
            temperature: 0.0,
            pressure: 0.0,
            wind: DVec3::ZERO,
        }
    }
}
