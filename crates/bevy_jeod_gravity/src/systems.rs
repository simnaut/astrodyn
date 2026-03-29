use bevy::prelude::*;
use bevy_jeod_dynamics::{
    GravityAccelerationC, GravityControlsC, GravitySourceC, PlanetFixedRotationC,
    TranslationalStateC,
};
use jeod_dynamics::GravityAcceleration;

/// Pre-computes gravity for each dynamic body. The result in
/// `GravityAccelerationC` is used by both `force_collection_system` (for
/// frame derivatives) and `integration_system` (held constant across RK4
/// stages, matching JEOD's `DynamicsIntegrationGroup::gravitation`).
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
        Entity,
        &TranslationalStateC,
        &GravityControlsC,
        &mut GravityAccelerationC,
    )>,
    sources: Query<(&GravitySourceC, Option<&PlanetFixedRotationC>)>,
) {
    for (entity, state, controls, mut accel) in &mut bodies {
        let mut total = GravityAcceleration::default();
        for ctrl in &controls.0.controls {
            // JEOD_INV: GV.12 — gravity source must exist for control.
            // JEOD: GravityControls::initialize_control() calls MessageHandler::error()
            // (non-fatal, severity 0) when find_grav_source() returns nullptr, then
            // returns — leaving the control uninitialized so gravity from that source
            // is silently skipped. We escalate to a panic because silently omitting a
            // gravity source would produce incorrect physics. Our validation system
            // already panics for this at startup; if we reach here, an entity was
            // despawned after validation — that's a lifecycle bug, not recoverable.
            let Ok((source, rot)) = sources.get(ctrl.source_name) else {
                panic!(
                    "Entity {entity:?}: GravityControl references entity {:?} which \
                     does not exist or has no GravitySourceC. JEOD logs a non-fatal \
                     error and skips; we panic to prevent silently wrong physics.",
                    ctrl.source_name
                );
            };

            // Pre-check: provide entity context before delegating to evaluate()
            if ctrl.is_nonspherical() && rot.is_none() {
                panic!(
                    "Entity {entity:?}: non-spherical GravityControl (degree={}, order={}) \
                     references source {:?} which is missing PlanetFixedRotationC",
                    ctrl.degree, ctrl.order, ctrl.source_name
                );
            }
            let result = ctrl.evaluate(&source.0, state.position, rot.map(|r| &r.0));
            total.grav_accel += result.grav_accel;
            if ctrl.gradient {
                total.grav_grad += result.grav_grad;
            }
            total.grav_pot += result.grav_pot;
        }
        accel.0 = total;
    }
}
