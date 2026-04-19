pub use jeod_quantities::prelude::*;

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

/// Compute atmospheric co-rotation wind velocity in the inertial frame.
///
/// Port of JEOD `WindVelocity::update_wind()` with uniform omega scale.
/// Wind is the cross product of the planet's angular velocity vector (Z-axis)
/// with the vehicle's inertial position: `wind = omega × r`.
///
/// For Earth's Z-axis rotation this simplifies to:
///   wind = [-omega * y, omega * x, 0]
///
/// # Arguments
/// * `omega` - Planet angular velocity in rad/s (Earth: 7.292115146706388e-5)
/// * `inertial_pos` - Vehicle position in the inertial frame (m)
// JEOD_INV: AT.04 — wind velocity computed as omega × position (co-rotation)
pub fn compute_corotation_wind(omega: f64, inertial_pos: DVec3) -> DVec3 {
    DVec3::new(-omega * inertial_pos.y, omega * inertial_pos.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corotation_wind_at_equator() {
        let omega = 7.292115146706388e-5;
        let r = 6_778_137.0; // ~400 km altitude
        let pos = DVec3::new(r, 0.0, 0.0); // on equator, X-axis

        let wind = compute_corotation_wind(omega, pos);

        // omega × [r, 0, 0] = [0, omega*r, 0]
        assert!(wind.x.abs() < 1e-10);
        assert!((wind.y - omega * r).abs() < 1e-6);
        assert!(wind.z.abs() < 1e-10);

        // ~494 m/s at 400 km
        assert!(wind.length() > 400.0 && wind.length() < 600.0);
    }

    #[test]
    fn corotation_wind_at_pole() {
        let omega = 7.292115146706388e-5;
        let pos = DVec3::new(0.0, 0.0, 6_778_137.0); // north pole

        let wind = compute_corotation_wind(omega, pos);

        // omega × [0, 0, r] = [0, 0, 0] for Z-axis rotation
        assert!(wind.length() < 1e-10);
    }

    #[test]
    fn corotation_wind_zero_omega() {
        let pos = DVec3::new(7e6, 3e6, 1e6);
        let wind = compute_corotation_wind(0.0, pos);
        assert_eq!(wind, DVec3::ZERO);
    }
}
