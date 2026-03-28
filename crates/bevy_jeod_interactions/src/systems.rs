use bevy::prelude::*;
use jeod_interactions::{compute_ballistic_drag, compute_gravity_torque, compute_srp_force};

use bevy_jeod_dynamics::{
    AerodynamicForceC, AtmosphericStateC, GravityAccelerationC, GravityTorqueC,
    MassPropertiesC, RadiationForceC, RotationalStateC, TranslationalStateC,
};
use crate::components::{DragConfigC, SrpConfigC};

/// Compute aerodynamic drag for entities with DragConfigC + AtmosphericStateC.
///
/// Placed in `JeodSet::Interaction` (after Environment, before ForceCollection).
pub fn aero_drag_system(
    mut query: Query<(
        &DragConfigC,
        &AtmosphericStateC,
        &TranslationalStateC,
        &RotationalStateC,
        &mut AerodynamicForceC,
    )>,
) {
    for (drag_config, atmos, state, rot, mut aero_force) in &mut query {
        let atmos_state = jeod_atmosphere::AtmosphericState {
            density: atmos.density,
            temperature: atmos.temperature,
            pressure: atmos.pressure,
            wind: atmos.wind,
        };

        let t_inertial_body = rot.quaternion.left_quat_to_transformation();

        let result = compute_ballistic_drag(
            &drag_config.0,
            &atmos_state,
            state.velocity,
            &t_inertial_body,
        );

        aero_force.force = result.force;
        aero_force.torque = result.torque;
    }
}

/// Compute gravity gradient torque for entities with GravityAccelerationC +
/// RotationalStateC + MassPropertiesC.
///
/// Placed in `JeodSet::Interaction`.
pub fn gravity_torque_system(
    mut query: Query<(
        &GravityAccelerationC,
        &RotationalStateC,
        &MassPropertiesC,
        &mut GravityTorqueC,
    )>,
) {
    for (grav, rot, mass, mut torque) in &mut query {
        let t_inertial_body = rot.quaternion.left_quat_to_transformation();

        torque.0 = compute_gravity_torque(
            &grav.grav_grad,
            &t_inertial_body,
            &mass.inertia,
        );
    }
}

/// Compute solar radiation pressure for entities with SrpConfigC.
///
/// Requires a Sun entity with `TranslationalStateC` to query the Sun position.
/// Shadow detection uses Earth radius from `jeod_interactions::SOLAR_RADIUS`.
///
/// Placed in `JeodSet::Interaction`.
pub fn radiation_pressure_system(
    mut query: Query<(
        &SrpConfigC,
        &TranslationalStateC,
        &mut RadiationForceC,
    ), Without<SunMarker>>,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
) {
    let Ok(sun_state) = sun_query.single() else {
        return; // No Sun entity → no SRP
    };

    for (srp_config, state, mut srp_force) in &mut query {
        // For now, no shadow detection (Phase 4 Tier 3 will add it)
        let shadow_fraction = 1.0;

        let result = compute_srp_force(
            &srp_config.0,
            sun_state.position,
            state.position,
            shadow_fraction,
        );

        srp_force.force = result.force;
        srp_force.torque = result.torque;
    }
}

/// Marker component for the Sun entity (used by SRP system to find Sun position).
#[derive(Component)]
pub struct SunMarker;
