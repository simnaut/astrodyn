//! Atmosphere model recipes.
//!
//! ```
//! use jeod_sim::recipes::atmosphere;
//! let a = atmosphere::met_solar_mean();
//! assert!(matches!(a.model, jeod_sim::AtmosphereModel::Met(_)));
//! ```

use jeod_atmosphere::exponential::ExponentialAtmosphere;
use jeod_atmosphere::met as met_atmos;

use crate::atmosphere::AtmosphereConfig;
use crate::AtmosphereModel;
use crate::EARTH;

/// MET (Marshall Engineering Thermosphere) atmosphere at solar-mean
/// conditions, configured for Earth (WGS84 ellipsoid + JEOD sidereal
/// angular velocity).
pub fn met_solar_mean() -> AtmosphereConfig {
    AtmosphereConfig::from_planet(AtmosphereModel::Met(met_atmos::SOLAR_MEAN), &EARTH)
}

/// MET atmosphere at solar-min conditions.
pub fn met_solar_min() -> AtmosphereConfig {
    AtmosphereConfig::from_planet(AtmosphereModel::Met(met_atmos::SOLAR_MIN), &EARTH)
}

/// MET atmosphere at solar-max conditions.
pub fn met_solar_max() -> AtmosphereConfig {
    AtmosphereConfig::from_planet(AtmosphereModel::Met(met_atmos::SOLAR_MAX), &EARTH)
}

/// Default exponential atmosphere (`rho = rho_0 * exp(-(h - h_0) / H)`).
/// No latitude / longitude / time dependence.
pub fn exponential_default() -> AtmosphereConfig {
    AtmosphereConfig::from_planet(
        AtmosphereModel::Exponential(ExponentialAtmosphere::default()),
        &EARTH,
    )
}
