use bevy::prelude::*;
use jeod_atmosphere::exponential::ExponentialAtmosphere;
use jeod_math::geodetic::cartesian_to_geodetic;

use bevy_jeod_dynamics::{
    AtmosphericStateC, PlanetFixedRotationC, TranslationalStateC,
};

/// Resource holding the atmosphere model configuration.
///
/// Currently supports only the exponential atmosphere model. The MET model is
/// implemented in `jeod_atmosphere` but is not yet wired into `AtmosphereModelR`.
#[derive(Resource, Debug, Clone)]
pub struct AtmosphereModelR {
    pub model: ExponentialAtmosphere,
    /// Equatorial radius of the planet (m). Used for geodetic conversion.
    pub r_eq: f64,
    /// Polar radius of the planet (m). Used for geodetic conversion.
    pub r_pol: f64,
    /// Entity of the planet whose atmosphere this is (for finding PlanetFixedRotationC).
    pub planet_entity: Option<Entity>,
}

// JEOD_INV: AT.01 — active flag gates computation (no AtmosphericStateC component = no computation)
// JEOD_INV: AT.02 — atmosphere model pointer non-null for update (AtmosphereModelR resource checked)
/// Update atmospheric state for entities that have AtmosphericStateC.
///
/// Converts the vehicle's inertial position to planet-fixed coordinates,
/// computes geodetic altitude, then evaluates the atmosphere model.
///
/// Placed in `JeodSet::Environment`.
pub fn atmosphere_update_system(
    atmos_model: Option<Res<AtmosphereModelR>>,
    planet_query: Query<&PlanetFixedRotationC>,
    mut query: Query<(&TranslationalStateC, &mut AtmosphericStateC)>,
) {
    // JEOD_INV: AT.02 — early return if no atmosphere model resource (non-null check)
    let Some(model) = atmos_model else {
        return; // No atmosphere model configured
    };

    // JEOD_INV: AT.03 — planet-fixed position required for geodetic altitude
    // JEOD's AtmosphereState::update_state() requires a PlanetFixedPosition pointer
    // that is always set during initialization. Missing it is a configuration error.
    let t_inertial_pfix = if let Some(entity) = model.planet_entity {
        let Ok(r) = planet_query.get(entity) else {
            panic!(
                "AtmosphereModelR.planet_entity is set ({entity:?}) but entity has no \
                 PlanetFixedRotationC. In JEOD, the planet-fixed frame is always \
                 available for atmosphere computation. Add PlanetFixedRotationC to \
                 the planet entity or set planet_entity to None for spherical fallback."
            );
        };
        Some(r.0)
    } else {
        None
    };

    for (state, mut atmos) in &mut query {
        // Convert inertial position to planet-fixed
        let pos_pfix = if let Some(rot) = t_inertial_pfix {
            rot * state.position
        } else {
            state.position // No rotation available → assume already in PCPF
        };

        // Compute geodetic altitude
        let geodetic = cartesian_to_geodetic(pos_pfix, model.r_eq, model.r_pol);

        // Evaluate atmosphere model
        let result = model.model.density(geodetic.altitude);

        atmos.density = result.density;
        atmos.temperature = result.temperature;
        atmos.pressure = result.pressure;
        atmos.wind = result.wind;
    }
}
