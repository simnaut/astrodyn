use bevy::prelude::*;
use glam::DVec3;

use crate::components::{DragConfigC, FlatPlateConfigC, ShadowBodyC};
use bevy_jeod_dynamics::{
    AerodynamicForceC, AtmosphericStateC, GravityAccelerationC, GravityTorqueC, MassPropertiesC,
    RadiationForceC, RotationalStateC, StructuralTransformC, TranslationalStateC,
};

/// Compute aerodynamic drag for entities with all required components.
///
/// Placed in `JeodSet::Interaction`.
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
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| s.0);

        let result = jeod_sim::compute_drag(
            &drag_config.0,
            atmos,
            state.velocity,
            Some(&rot.0),
            t_struct_body,
        );

        aero_force.force = result.force;
        aero_force.torque = result.torque;
    }
}

/// Compute gravity gradient torque.
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
        torque.0 = jeod_sim::compute_gravity_torque(&grav.grav_grad, &rot.0, &mass.inertia);
    }
}

/// Compute illumination factor from all shadow-casting bodies.
fn compute_illum_factor(
    vehicle_pos: DVec3,
    sun_pos: DVec3,
    shadow_bodies: &Query<(&TranslationalStateC, &ShadowBodyC), Without<SunMarker>>,
) -> f64 {
    let mut illum = 1.0_f64;
    for (body_state, shadow) in shadow_bodies.iter() {
        let factor = jeod_sim::compute_shadow_fraction(
            vehicle_pos,
            sun_pos,
            body_state.position,
            shadow.radius,
            jeod_sim::SOLAR_RADIUS,
        );
        illum = illum.min(factor);
    }
    illum
}

/// Compute flat-plate SRP with thermal emission and shadow detection.
///
// JEOD_INV: IN.06 — RadiationPressure.active gates computation (structural: no FlatPlateConfigC → no SRP)
// JEOD_INV: IN.09 — RadiationSource planet must exist (SunMarker required; panics on multiple)
/// For entities with `FlatPlateConfigC`. Handles:
/// - Solar flux at vehicle distance
/// - Conical shadow from `ShadowBodyC` entities
/// - Per-plate absorption, diffuse/specular reflection, thermal emission
/// - Temperature integration (forward Euler)
/// - Force is rotated from structural to inertial by this system before writing `RadiationForceC`
///
/// Placed in `JeodSet::Interaction`.
#[allow(clippy::type_complexity)]
pub fn flat_plate_srp_system(
    mut query: Query<
        (
            &mut FlatPlateConfigC,
            &TranslationalStateC,
            Option<&RotationalStateC>,
            Option<&MassPropertiesC>,
            Option<&StructuralTransformC>,
            &mut RadiationForceC,
        ),
        Without<SunMarker>,
    >,
    sun_query: Query<&TranslationalStateC, With<SunMarker>>,
    shadow_bodies: Query<(&TranslationalStateC, &ShadowBodyC), Without<SunMarker>>,
    time: Res<Time<Fixed>>,
) {
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

    let dt = time.delta_secs_f64();

    for (mut flat_config, state, rot, mass, struct_xform, mut srp_force) in &mut query {
        let sun_to_vehicle = state.position - sun_state.position;
        let distance = sun_to_vehicle.length();
        if distance < 1.0 {
            continue;
        }
        let flux_inertial_hat = sun_to_vehicle / distance;
        let flux_mag = jeod_sim::solar_flux_at_distance(distance);

        // Shadow fraction
        let illum_factor = compute_illum_factor(state.position, sun_state.position, &shadow_bodies);

        // Rotate flux to structural frame
        let t_inertial_body = rot.map_or(glam::DMat3::IDENTITY, |r| {
            r.quaternion.left_quat_to_transformation()
        });
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| s.0);
        let t_inertial_struct =
            jeod_sim::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);
        let flux_struct_hat = t_inertial_struct * flux_inertial_hat;

        let center_grav = mass.map_or(DVec3::ZERO, |m| m.position);

        let srp_result = jeod_sim::compute_flat_plate_srp_thermal(
            &flat_config.plates,
            &flat_config.t_pow4_cached,
            flux_struct_hat,
            flux_mag,
            center_grav,
            illum_factor,
        );

        // Force: rotate from structural to inertial. Torque: stays structural.
        let force_inertial = t_inertial_struct.transpose() * srp_result.force;
        srp_force.force = force_inertial;
        srp_force.torque = srp_result.torque;

        // Integrate plate temperatures (forward Euler, matching Simulation runner)
        if dt > 0.0 {
            for (i, temp) in flat_config.temperatures.iter_mut().enumerate() {
                *temp += srp_result.temp_dots[i] * dt;
                if *temp < 0.0 {
                    *temp = 0.0;
                }
            }
            flat_config.t_pow4_cached =
                flat_config.temperatures.iter().map(|t| t.powi(4)).collect();
        }
    }
}

/// Marker component for the Sun entity (used by SRP system to find Sun position).
#[derive(Component)]
pub struct SunMarker;
