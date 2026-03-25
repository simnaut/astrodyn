use bevy::prelude::*;
use bevy_jeod_dynamics::{
    GravityAccelerationC, GravityControlsC, GravitySourceC, PlanetFixedRotationC,
    TranslationalStateC,
};
use jeod_dynamics::GravityAcceleration;

/// Phase 2 scaffolding: stores pre-computed gravity for use by
/// `force_collection_system`. The `integration_system` independently recomputes
/// gravity at each RK4 stage for 4th-order accuracy; this stored value is not
/// used for integration.
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
    sources: Query<(&GravitySourceC, Option<&PlanetFixedRotationC>)>,
) {
    for (state, controls, mut accel) in &mut bodies {
        let mut total = GravityAcceleration::default();
        for ctrl in &controls.0.controls {
            let Ok((source, rot)) = sources.get(ctrl.source_name) else {
                warn!(
                    "GravityControl references entity {:?} which has no GravitySourceC",
                    ctrl.source_name
                );
                continue;
            };
            let t_parent_this = rot.map_or(glam::DMat3::IDENTITY, |r| r.0);
            let result = jeod_gravity::gravitation(
                &source.0, state.position, &t_parent_this,
                ctrl.degree, ctrl.order, ctrl.perturbing_only,
                ctrl.gradient,
                ctrl.gradient_degree,
                ctrl.gradient_order,
            );
            total.grav_accel += result.grav_accel;
            if ctrl.gradient {
                total.grav_grad += result.grav_grad;
            }
            total.grav_pot += result.grav_pot;
        }
        accel.0 = total;
    }
}
