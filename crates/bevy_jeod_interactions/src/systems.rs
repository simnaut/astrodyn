use bevy::prelude::*;
use jeod_interactions::{compute_ballistic_drag, compute_gravity_torque, compute_srp_force};

use bevy_jeod_dynamics::{
    AerodynamicForceC, AtmosphericStateC, GravityAccelerationC, GravityTorqueC,
    MassPropertiesC, RadiationForceC, RotationalStateC, StructuralTransformC,
    TranslationalStateC,
};
use crate::components::{DragConfigC, SrpConfigC};

/// Compute aerodynamic drag for entities with all required components:
/// `DragConfigC`, `AtmosphericStateC`, `TranslationalStateC`, `RotationalStateC`,
/// and `AerodynamicForceC`.
///
/// Placed in `JeodSet::Interaction` (after Environment, before ForceCollection).
// JEOD_INV: IN.03 — AerodynamicDrag.active gates computation (structural: no DragConfigC -> no drag)
#[allow(clippy::type_complexity)]
pub fn aero_drag_system(
    mut query: Query<(
        &DragConfigC,
        &AtmosphericStateC,
        &TranslationalStateC,
        &RotationalStateC,
        Option<&StructuralTransformC>,
        &mut AerodynamicForceC,
    )>,
) {
    for (drag_config, atmos, state, rot, struct_xform, mut aero_force) in &mut query {
        // JEOD passes T_inertial_struct (inertial→structural), not T_inertial_body.
        let t_inertial_body = rot.quaternion.left_quat_to_transformation();
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| s.0);
        let t_inertial_struct = jeod_dynamics::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);

        let result = compute_ballistic_drag(
            &drag_config.0,
            &atmos,  // AtmosphericStateC derefs to AtmosphereState
            state.velocity,
            &t_inertial_struct,
        );

        aero_force.force = result.force;
        aero_force.torque = result.torque;
    }
}

/// Compute gravity gradient torque for entities with GravityAccelerationC +
/// RotationalStateC + MassPropertiesC.
///
/// Placed in `JeodSet::Interaction`.
// JEOD_INV: IN.01 — GravityTorque.subject_body required (structural: query requires all components)
// JEOD_INV: IN.02 — GravityTorque.active gates computation (structural: no GravityTorqueC -> no torque)
pub fn gravity_torque_system(
    mut query: Query<(
        &GravityAccelerationC,
        &RotationalStateC,
        &MassPropertiesC,
        &mut GravityTorqueC,
    )>,
) {
    for (grav, rot, mass, mut torque) in &mut query {
        let t_parent_this = rot.quaternion.left_quat_to_transformation();

        torque.0 = compute_gravity_torque(
            &grav.grav_grad,
            &t_parent_this,
            &mass.inertia,
        );
    }
}

/// Compute solar radiation pressure for entities with SrpConfigC.
///
/// Requires a Sun entity with `TranslationalStateC` and `SunMarker` to query
/// the Sun position. Shadow detection is not yet wired in — `illum_factor`
/// is hard-coded to 1.0 (full sun). To add eclipse support, the system would
/// need Earth entity position and radius to call `compute_shadow_fraction`.
///
/// Placed in `JeodSet::Interaction`.
// JEOD_INV: IN.06 — RadiationPressure.active gates computation (structural: no SrpConfigC -> no SRP)
// JEOD_INV: IN.09 — RadiationSource planet must exist (partial: SunMarker required)
pub fn radiation_pressure_system(
    mut query: Query<(
        &SrpConfigC,
        &TranslationalStateC,
        &mut RadiationForceC,
    ), Without<SunMarker>>,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
) {
    // JEOD_INV: IN.09 — RadiationSource planet must be found by DynManager
    // JEOD's RadiationSource::initialize() fatally fails if the source planet
    // is not found. Zero suns = SRP not configured (silent return); multiple = panic.
    let sun_state = match sun_query.single() {
        Ok(s) => s,
        Err(bevy::ecs::query::QuerySingleError::NoEntities(_)) => return,
        Err(bevy::ecs::query::QuerySingleError::MultipleEntities(_)) => {
            panic!(
                "Multiple entities with SunMarker found. In JEOD, RadiationPressure \
                 has exactly one RadiationSource (value member). Ensure exactly one \
                 Sun entity exists."
            );
        }
    };

    for (srp_config, state, mut srp_force) in &mut query {
        // For now, no shadow detection (Phase 4 Tier 3 will add it)
        let illum_factor = 1.0;

        let result = compute_srp_force(
            &srp_config.0,
            sun_state.position,
            state.position,
            illum_factor,
        );

        srp_force.force = result.force;
        srp_force.torque = result.torque;
    }
}

/// Marker component for the Sun entity (used by SRP system to find Sun position).
#[derive(Component)]
pub struct SunMarker;
