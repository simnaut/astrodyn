use bevy::prelude::*;
use bevy_jeod_dynamics::{
    GravityAccelerationC, GravityControlsC, GravitySourceC, PlanetFixedRotationC,
    TranslationalStateC,
};

/// Pre-computes gravity for each dynamic body. The result in
/// `GravityAccelerationC` is used by both `force_collection_system` (for
/// frame derivatives) and `integration_system` (held constant across RK4
/// stages, matching JEOD's `DynamicsIntegrationGroup::gravitation`).
///
/// For each body that has `GravityControlsC`, the system iterates over its
/// control entries, looks up the corresponding `GravitySourceC` entity, and
/// accumulates the acceleration, gradient, and potential.
///
/// Delegates to [`jeod_sim::accumulate_gravity`] for the per-body accumulation
/// loop, providing a closure that resolves Bevy entity references.
pub fn gravity_computation_system(
    mut bodies: Query<(
        Entity,
        &TranslationalStateC,
        &GravityControlsC,
        &mut GravityAccelerationC,
    )>,
    sources: Query<(&GravitySourceC, Option<&PlanetFixedRotationC>)>,
) {
    for (entity, state, controls, mut accel) in &mut bodies {
        accel.0 = jeod_sim::accumulate_gravity(
            state.position,
            &controls.0,
            |source_entity| {
                match sources.get(source_entity) {
                    Ok((source, rot)) => Some((&source.0, rot.map(|r| &r.0))),
                    Err(_) => {
                        // Let accumulate_gravity handle the panic with a descriptive message
                        None
                    }
                }
            },
        );
        let _ = entity; // Entity available for future per-entity diagnostics
    }
}
