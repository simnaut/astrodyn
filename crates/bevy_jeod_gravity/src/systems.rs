use bevy::prelude::*;
use bevy_jeod_dynamics::{
    GravityAccelerationC, GravityControlsC, GravitySourceC, PlanetFixedRotationC,
    TranslationalStateC,
};

/// Pre-computes gravity for each dynamic body.
///
/// Gravity is precomputed here in the Environment stage but is recomputed at
/// each RK4 stage by the integration system for 4th-order accuracy.
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
        accel.0 =
            jeod_sim::accumulate_gravity(
                state.position,
                &controls.0,
                |source_entity| match sources.get(source_entity) {
                    Ok((source, rot)) => Some((&source.0, rot.map(|r| &r.0))),
                    Err(_) => {
                        panic!(
                            "Entity {entity:?}: GravityControl references source \
                         {source_entity:?} which does not exist or has no GravitySourceC."
                        );
                    }
                },
            );
    }
}
