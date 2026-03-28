use bevy::prelude::*;
use jeod_atmosphere::exponential::ExponentialAtmosphere;
use jeod_atmosphere::met::MetAtmosphere;
use jeod_math::geodetic::cartesian_to_geodetic;

use bevy_jeod_dynamics::{
    AtmosphericStateC, PlanetFixedRotationC, TranslationalStateC,
};
use bevy_jeod_time::SimulationTimeR;

/// Selectable atmosphere model.
#[derive(Debug, Clone)]
pub enum AtmosphereModel {
    /// Simple exponential: `rho = rho_0 * exp(-(h - h_0) / H)`.
    /// No time, latitude, or longitude dependence.
    Exponential(ExponentialAtmosphere),
    /// Marshall Engineering Thermosphere (Jacchia 1970/1971).
    /// Full altitude/latitude/longitude/time/solar-activity dependence.
    Met(MetAtmosphere),
}

/// Resource holding the atmosphere model configuration.
#[derive(Resource, Debug, Clone)]
pub struct AtmosphereModelR {
    /// The atmosphere model to evaluate.
    pub model: AtmosphereModel,
    /// Equatorial radius of the planet (m). Used for geodetic conversion.
    pub r_eq: f64,
    /// Polar radius of the planet (m). Used for geodetic conversion.
    pub r_pol: f64,
    /// Entity of the planet whose atmosphere this is (for finding PlanetFixedRotationC).
    pub planet_entity: Option<Entity>,
    /// Planet angular velocity in rad/s for atmospheric co-rotation wind.
    /// Set to 0.0 to disable wind computation.
    /// Earth: 7.292115146706388e-5 rad/s (from JEOD RNPJ2000 data).
    /// Port of JEOD `WindVelocity::omega`.
    pub planet_omega: f64,
}

// JEOD_INV: AT.01 — active flag gates computation (no AtmosphericStateC component = no computation)
// JEOD_INV: AT.02 — atmosphere model pointer non-null for update (AtmosphereModelR resource checked)
/// Update atmospheric state for entities that have AtmosphericStateC.
///
/// Converts the vehicle's inertial position to planet-fixed coordinates,
/// computes geodetic altitude/latitude/longitude, then evaluates the
/// atmosphere model.
///
/// For the MET model, also reads `SimulationTimeR` to get TAI TJT (truncated
/// Julian time) for solar angle and seasonal variation computation.
///
/// Placed in `JeodSet::Environment`.
pub fn atmosphere_update_system(
    atmos_model: Option<Res<AtmosphereModelR>>,
    sim_time: Option<Res<SimulationTimeR>>,
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

        // Compute geodetic coordinates
        let geodetic = cartesian_to_geodetic(pos_pfix, model.r_eq, model.r_pol);

        // Evaluate atmosphere model
        let result = match &model.model {
            AtmosphereModel::Exponential(exp) => exp.density(geodetic.altitude),
            AtmosphereModel::Met(met) => {
                let tjt = sim_time
                    .as_ref()
                    .expect(
                        "MET atmosphere requires SimulationTimeR resource for TJT. \
                         Add JeodTimePlugin before JeodAtmospherePlugin."
                    )
                    .tai_tjt;

                met.density(
                    geodetic.altitude / 1000.0, // MET expects altitude in km
                    geodetic.latitude,
                    geodetic.longitude,
                    tjt,
                )
            }
        };

        atmos.density = result.density;
        atmos.temperature = result.temperature;
        atmos.pressure = result.pressure;

        // JEOD_INV: AT.04 — wind velocity computed as omega × position (co-rotation)
        // Port of JEOD WindVelocity::update_wind() with uniform omega scale.
        // Wind uses the vehicle's inertial position (matching JEOD's
        // dyn_body.composite_body.state.trans.position input).
        atmos.wind = if model.planet_omega != 0.0 {
            jeod_atmosphere::compute_corotation_wind(model.planet_omega, state.position)
        } else {
            result.wind
        };
    }
}
