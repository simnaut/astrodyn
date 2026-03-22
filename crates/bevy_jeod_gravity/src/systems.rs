use bevy::prelude::*;
use bevy_jeod_dynamics::{GravityAccelerationC, GravityControlsC, GravitySourceC, TranslationalStateC};
use jeod_dynamics::GravityAcceleration;

/// Computes gravitational acceleration on each body from all its referenced
/// gravity sources.
///
/// For each body that has `GravityControlsC`, the system iterates over its
/// control entries, looks up the corresponding `GravitySourceC` entity, and
/// accumulates the acceleration, gradient, and potential.
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
            if let Ok(source) = sources.get(ctrl.source_id) {
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
        }
        accel.0 = total;
    }
}
