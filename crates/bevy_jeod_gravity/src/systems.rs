use bevy::prelude::*;
use bevy_jeod_dynamics::{GravityAccelerationC, GravityControlsC, GravitySourceC, TranslationalStateC};
use jeod_dynamics::GravityAcceleration;

/// Computes gravitational acceleration on each body from all its referenced
/// gravity sources.
///
/// For each body that has `GravityControlsC`, the system iterates over its
/// control entries, looks up the corresponding `GravitySourceC` entity, and
/// accumulates the acceleration, gradient, and potential.
///
/// **Phase 1 assumption**: gravity sources are at the origin of the integration
/// frame (body position is relative to the source center). In Phase 2, source
/// positions will be obtained from `TranslationalStateC` on the source entity
/// and subtracted to get relative position.
pub fn gravity_computation_system(
    mut bodies: Query<(
        &TranslationalStateC,
        &GravityControlsC,
        &mut GravityAccelerationC,
    )>,
    sources: Query<&GravitySourceC>,
) {
    for (state, controls, mut accel) in &mut bodies {
        let mut total = GravityAcceleration::default();
        for ctrl in &controls.0.controls {
            let Ok(source) = sources.get(ctrl.source_id) else {
                warn!(
                    "GravityControl references entity {:?} which has no GravitySourceC",
                    ctrl.source_id
                );
                continue;
            };
            let result = jeod_gravity::compute_gravity(&source.0, state.position);
            total.accel += result.accel;
            if ctrl.compute_gradient {
                total.gradient = glam::DMat3::from_cols(
                    total.gradient.col(0) + result.gradient.col(0),
                    total.gradient.col(1) + result.gradient.col(1),
                    total.gradient.col(2) + result.gradient.col(2),
                );
            }
            total.potential += result.potential;
        }
        accel.0 = total;
    }
}
