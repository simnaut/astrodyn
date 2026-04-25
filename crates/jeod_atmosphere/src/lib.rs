pub use jeod_quantities::prelude::*;

pub mod exponential;
pub mod met;

use glam::DVec3;
use uom::si::f64::{MassDensity, Pressure, ThermodynamicTemperature};
use uom::si::mass_density::kilogram_per_cubic_meter;
use uom::si::pressure::pascal;
use uom::si::thermodynamic_temperature::kelvin;

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

impl AtmosphereState {
    /// Typed accessor: atmospheric density as `uom::si::f64::MassDensity` (kg/m^3).
    #[inline]
    pub fn density_typed(&self) -> MassDensity {
        MassDensity::new::<kilogram_per_cubic_meter>(self.density)
    }

    /// Typed accessor: atmospheric temperature as
    /// `uom::si::f64::ThermodynamicTemperature` (kelvin).
    #[inline]
    pub fn temperature_typed(&self) -> ThermodynamicTemperature {
        ThermodynamicTemperature::new::<kelvin>(self.temperature)
    }

    /// Typed accessor: atmospheric pressure as `uom::si::f64::Pressure` (pascal).
    #[inline]
    pub fn pressure_typed(&self) -> Pressure {
        Pressure::new::<pascal>(self.pressure)
    }

    /// Typed accessor: wind velocity as a frame-tagged
    /// `Velocity<Inertial>` (m/s).
    #[inline]
    pub fn wind_typed(&self) -> Velocity<Inertial> {
        self.wind.m_per_s_at::<Inertial>()
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

/// Typed variant of [`compute_corotation_wind`].
///
/// Wraps the same computation with dimension-typed inputs/outputs. Delegates
/// to the raw f64/DVec3 implementation internally so behavior is bit-identical.
#[inline]
pub fn compute_corotation_wind_typed(
    omega: uom::si::f64::AngularVelocity,
    pos: Position<Inertial>,
) -> Velocity<Inertial> {
    use uom::si::angular_velocity::radian_per_second;
    let w = omega.get::<radian_per_second>();
    let p = pos.raw_si();
    compute_corotation_wind(w, p).m_per_s_at::<Inertial>()
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

    #[test]
    fn typed_accessors_roundtrip_bit_identical() {
        let state = AtmosphereState {
            density: 1.225e-12,
            temperature: 288.15,
            pressure: 2.537e-10,
            wind: DVec3::new(-359.7, 123.4, 0.5),
        };

        // Values extracted through typed accessors must equal the raw f64
        // inputs bit-for-bit — no unit conversion or round-trip loss.
        assert_eq!(
            state.density_typed().get::<kilogram_per_cubic_meter>(),
            state.density
        );
        assert_eq!(state.temperature_typed().get::<kelvin>(), state.temperature);
        assert_eq!(state.pressure_typed().get::<pascal>(), state.pressure);
        assert_eq!(state.wind_typed().raw_si(), state.wind);
    }

    #[test]
    fn typed_accessors_default_state() {
        let state = AtmosphereState::default();
        assert_eq!(state.density_typed().get::<kilogram_per_cubic_meter>(), 0.0);
        assert_eq!(state.temperature_typed().get::<kelvin>(), 0.0);
        assert_eq!(state.pressure_typed().get::<pascal>(), 0.0);
        assert_eq!(state.wind_typed().raw_si(), DVec3::ZERO);
    }

    #[test]
    fn corotation_wind_typed_matches_raw() {
        let omega_raw = 7.292115146706388e-5;
        let pos_raw = DVec3::new(7.0e6, 3.0e6, 1.0e6);

        let typed_out =
            compute_corotation_wind_typed(omega_raw.rad_per_s(), pos_raw.m_at::<Inertial>());
        let raw_out = compute_corotation_wind(omega_raw, pos_raw);

        // Typed path must be bit-identical to the raw f64 path.
        assert_eq!(typed_out.raw_si(), raw_out);
    }

    #[test]
    fn corotation_wind_typed_zero_omega() {
        let pos = DVec3::new(7e6, 3e6, 1e6).m_at::<Inertial>();
        let wind = compute_corotation_wind_typed(0.0.rad_per_s(), pos);
        assert_eq!(wind.raw_si(), DVec3::ZERO);
    }
}
