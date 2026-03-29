use bevy::prelude::*;

use bevy_jeod_dynamics::{AtmosphericStateC, PlanetFixedRotationC, TranslationalStateC};
use bevy_jeod_time::SimulationTimeR;

// Re-export jeod_sim types that form the public API of this crate.
pub use jeod_sim::atmosphere::{AtmosphereConfig, AtmosphereModel};

/// Bevy resource wrapping [`AtmosphereConfig`] with an entity reference for
/// the planet whose rotation matrix is used for geodetic conversion.
#[derive(Resource, Debug, Clone)]
pub struct AtmosphereModelR {
    /// ECS-agnostic atmosphere configuration (model, radii, wind).
    pub config: AtmosphereConfig,
    /// Entity of the planet whose `PlanetFixedRotationC` is used.
    /// `None` means no rotation (position assumed planet-fixed).
    pub planet_entity: Option<Entity>,
}

// JEOD_INV: AT.01 — active flag gates computation (no AtmosphericStateC component = no computation)
// JEOD_INV: AT.02 — atmosphere model pointer non-null for update (AtmosphereModelR resource checked)
/// Update atmospheric state for entities that have `AtmosphericStateC`.
///
/// Delegates to [`jeod_sim::evaluate_atmosphere`] for the per-body evaluation
/// pipeline (planet-fixed rotation, geodetic conversion, model dispatch, wind).
///
/// Placed in `JeodSet::Environment`.
pub fn atmosphere_update_system(
    atmos_model: Option<Res<AtmosphereModelR>>,
    sim_time: Option<Res<SimulationTimeR>>,
    planet_query: Query<&PlanetFixedRotationC>,
    mut query: Query<(&TranslationalStateC, &mut AtmosphericStateC)>,
) {
    // JEOD_INV: AT.02 — early return if no atmosphere model resource
    let Some(model) = atmos_model else {
        return;
    };

    // JEOD_INV: AT.03 — planet-fixed position required for geodetic altitude
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

    let tai_tjt = sim_time.as_ref().map(|t| t.tai_tjt);
    // MET atmosphere requires time for seasonal variation. If SimulationTimeR
    // is missing, evaluate_atmosphere will panic — provide a clear hint.
    if tai_tjt.is_none() {
        if let jeod_sim::AtmosphereModel::Met(_) = &model.config.model {
            panic!(
                "MET atmosphere requires SimulationTimeR resource for TJT. \
                 Add JeodTimePlugin before JeodAtmospherePlugin."
            );
        }
    }

    for (state, mut atmos) in &mut query {
        **atmos = jeod_sim::evaluate_atmosphere(
            &model.config,
            state.position,
            t_inertial_pfix.as_ref(),
            tai_tjt,
        );
    }
}
