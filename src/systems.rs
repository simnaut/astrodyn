use bevy::prelude::*;
use glam::DVec3;

use crate::components::*;
use crate::AtmosphereModelR;
use crate::SimulationTimeR;

// ── Time ──

// JEOD_INV: TM.03 — time types updated in dependency order (delegates to SimulationTime::advance)
pub fn time_advance_system(mut sim_time: ResMut<SimulationTimeR>, time: Res<Time<Fixed>>) {
    let dt = time.delta_secs_f64();
    sim_time.advance(dt);
}

// ── Ephemeris / Frames ──

/// Computes the inertial-to-planet-fixed rotation matrix (RNP) for each entity
/// that carries a `PlanetFixedRotationC` component.
///
/// This replaces the `DMat3::IDENTITY` placeholder so that spherical-harmonic
/// gravity evaluation uses the correct body-fixed coordinates.
pub fn planet_fixed_rotation_system(
    sim_time: Res<SimulationTimeR>,
    mut query: Query<&mut PlanetFixedRotationC>,
) {
    let rotation =
        jeod_sim::compute_t_parent_this_from_tjt(sim_time.gmst_seconds, sim_time.tt_tjt());
    for mut rot in &mut query {
        rot.0 = rotation;
    }
}

// ── Dynamics ──

/// Recompute derived mass quantities (`inverse_mass`, `inverse_inertia`) each step.
///
/// Port of JEOD's `(DYNAMICS, "scheduled") dyn_body.mass.update_mass_properties()`.
/// JEOD runs this every timestep so that runtime mass changes (fuel burn,
/// staging, attach/detach) are reflected in the dynamics before the next
/// derivative computation.
///
/// Placed before `JeodSet::EphemerisUpdate` so gravity and force collection
/// see current mass properties.
pub fn mass_update_system(mut query: Query<&mut MassPropertiesC>) {
    for mut mass in &mut query {
        mass.recompute_derived();
    }
}

/// Collects non-gravity forces and all torques into `TotalForceC`.
///
/// Delegates to [`jeod_sim::collect_and_resolve_forces`] for frame-aware
/// force/torque aggregation and frame derivative computation.
///
/// Gravity is intentionally **excluded** because the integration system
/// recomputes it at each RK4 stage for 4th-order accuracy. Non-gravity
/// forces (aero, SRP) are approximately constant over one timestep and
/// are added to the per-stage gravity inside the integrator.
// JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
// JEOD_INV: DB.29 — torques collected in structural frame, rotated to body at root
#[allow(clippy::type_complexity)]
pub fn force_collection_system(
    mut query: Query<(
        &mut TotalForceC,
        Option<&mut FrameDerivativesC>,
        Option<&GravityAccelerationC>,
        Option<&RotationalStateC>,
        Option<&MassPropertiesC>,
        Option<&AerodynamicForceC>,
        Option<&RadiationForceC>,
        Option<&GravityTorqueC>,
        Option<&StructuralTransformC>,
    )>,
) {
    for (mut total, derivs, grav, rot_state, mass, aero, srp, grav_torque, struct_xform) in
        &mut query
    {
        let t_struct_body = struct_xform.map_or(glam::DMat3::IDENTITY, |s| s.0);
        let grav_accel = grav.map_or(DVec3::ZERO, |g| g.grav_accel);

        // Map Bevy component references to jeod_interactions types for jeod_sim.
        let aero_ref = aero.map(|a| jeod_sim::AerodynamicForce {
            force: a.force,
            torque: a.torque,
        });
        let srp_ref = srp.map(|s| jeod_sim::RadiationForce {
            force: s.force,
            torque: s.torque,
        });
        let gravity_torque_val = grav_torque.map(|gt| gt.0);

        let (collected, frame_derivs) = jeod_sim::collect_and_resolve_forces(
            aero_ref.as_ref(),
            srp_ref.as_ref(),
            gravity_torque_val,
            rot_state.map(|r| &r.0),
            t_struct_body,
            mass.map(|m| &m.0),
            grav_accel,
        );

        total.force = collected.force;
        total.torque = collected.torque;

        if let Some(mut derivs) = derivs {
            **derivs = frame_derivs;
        }
    }
}

/// Advances translational (and optionally rotational) state via RK4 integration.
///
/// Delegates to [`jeod_sim::integrate_body`] for 6-DOF/3-DOF routing and
/// RK4 stepping. Gravity is recomputed at each RK4 intermediate state
/// for proper 4th-order accuracy, matching JEOD's `DynamicsIntegrationGroup`
/// where the derivative function recomputes gravity at every stage.
#[allow(clippy::type_complexity)]
pub fn integration_system(
    mut bodies: Query<(
        Entity,
        &DynamicsConfigC,
        &mut TranslationalStateC,
        Option<&mut RotationalStateC>,
        Option<&MassPropertiesC>,
        &GravityControlsC,
        &TotalForceC,
    )>,
    sources: Query<(&GravitySourceC, Option<&PlanetFixedRotationC>)>,
    time: Res<Time<Fixed>>,
) {
    let dt = time.delta_secs_f64();
    if dt == 0.0 {
        return;
    }

    for (entity, config, mut state, mut rot_state, mass, controls, total_force) in &mut bodies {
        let _ = entity; // available for panic context if integrate_body fails
        jeod_sim::integrate_body(
            config,
            &mut state.0,
            rot_state.as_mut().map(|r| &mut r.0),
            mass.map(|m| &m.0),
            |pos| {
                jeod_sim::accumulate_gravity(pos, &controls.0, |source_entity| {
                    sources
                        .get(source_entity)
                        .ok()
                        .map(|(s, r)| (&s.0, r.map(|r| &r.0)))
                })
                .grav_accel
            },
            total_force.force,
            total_force.torque,
            dt,
        );
    }
}

// ── Gravity ──

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

// ── Atmosphere ──

// JEOD_INV: AT.01 — active flag gates computation (no AtmosphericStateC component = no computation)
// JEOD_INV: AT.02 — atmosphere model pointer non-null for update (AtmosphereModelR resource checked)
/// Update atmospheric state for entities that have `AtmosphericStateC`.
///
/// Delegates to [`jeod_sim::evaluate_atmosphere`] for the per-body evaluation
/// pipeline (planet-fixed rotation, geodetic conversion, model dispatch, wind).
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
    for (state, mut atmos) in &mut query {
        // MET atmosphere requires time for seasonal variation. Check only when
        // entities with AtmosphericStateC actually exist (avoids panic when MET
        // is configured but no bodies need atmosphere yet).
        if tai_tjt.is_none() {
            if let jeod_sim::AtmosphereModel::Met(_) = &model.config.model {
                panic!(
                    "MET atmosphere requires SimulationTimeR resource for TJT. \
                     Ensure JeodPlugin is added (it provides SimulationTimeR)."
                );
            }
        }
        **atmos = jeod_sim::evaluate_atmosphere(
            &model.config,
            state.position,
            t_inertial_pfix.as_ref(),
            tai_tjt,
        );
    }
}

// ── Interactions ──

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
