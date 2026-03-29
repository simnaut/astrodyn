use bevy::prelude::*;
use glam::DVec3;

use crate::components::{
    AerodynamicForceC, DynamicsConfigC, FrameDerivativesC, GravityAccelerationC, GravityTorqueC,
    MassPropertiesC, RadiationForceC, RotationalStateC, StructuralTransformC, TotalForceC,
    TranslationalStateC,
};

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
        let aero_ref = aero.map(|a| jeod_interactions::AerodynamicForce {
            force: a.force,
            torque: a.torque,
        });
        let srp_ref = srp.map(|s| jeod_interactions::RadiationForce {
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
/// RK4 stepping. Gravity is held constant across all RK4 stages (matching
/// JEOD's `DynamicsIntegrationGroup`).
#[allow(clippy::type_complexity)]
pub fn integration_system(
    mut bodies: Query<(
        Entity,
        &DynamicsConfigC,
        &mut TranslationalStateC,
        Option<&mut RotationalStateC>,
        Option<&MassPropertiesC>,
        &GravityAccelerationC,
        &TotalForceC,
    )>,
    time: Res<Time<Fixed>>,
) {
    let dt = time.delta_secs_f64();
    if dt == 0.0 {
        return;
    }

    for (_entity, config, mut state, mut rot_state, mass, grav, total_force) in &mut bodies {
        jeod_sim::integrate_body(
            config,
            &mut state.0,
            rot_state.as_mut().map(|r| &mut r.0),
            mass.map(|m| &m.0),
            grav.grav_accel,
            total_force.force,
            total_force.torque,
            dt,
        );
    }
}
