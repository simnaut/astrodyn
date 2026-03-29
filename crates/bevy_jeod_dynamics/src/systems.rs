use bevy::prelude::*;
use glam::{DMat3, DVec3};
use jeod_dynamics::{ForceContributions, SixDofState};

use crate::components::{
    AerodynamicForceC, DynamicsConfigC, FrameDerivativesC, GravityAccelerationC,
    GravityControlsC, GravitySourceC, GravityTorqueC, MassPropertiesC,
    PlanetFixedRotationC, RadiationForceC, RotationalStateC, StructuralTransformC,
    TotalForceC, TranslationalStateC,
};

/// Collects non-gravity forces and all torques into `TotalForceC`.
///
/// Gravity is intentionally **excluded** because the integration system
/// recomputes it at each RK4 stage for 4th-order accuracy. Non-gravity
/// forces (aero, SRP) are approximately constant over one timestep and
/// are added to the per-stage gravity inside the integrator.
///
/// Frame pipeline (matching JEOD `dyn_body_collect.cc`):
///
/// **Force** (collected in structural frame, rotated to inertial):
///   `T_inertial_struct^T * (aero_force_struct + ...)` + SRP (spherical, already inertial)
///
/// **Torque** (collected in body frame):
///   `T_struct_body * aero_torque_struct` + gravity_torque (already body) +
///   `T_struct_body * srp_torque_struct`
///
/// `T_inertial_struct = T_struct_body^T * T_inertial_body` where
/// `T_struct_body` is from `StructuralTransformC` (defaults to identity).
///
/// All interaction components are optional — entities without them
/// contribute zero from those terms.
// Force collection and frame derivative physics delegated to jeod_dynamics::collect_forces
// and jeod_dynamics::compute_frame_derivatives (DB.28, DB.29, FD.01, FD.02).
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
    // Note: JEOD gates force/torque collection on translational/rotational_dynamics flags (DB.07/DB.08).
    // We collect unconditionally here; gating is enforced in integration_system.
    for (mut total, derivs, grav, rot_state, mass, aero, srp, grav_torque, struct_xform) in &mut query {
        // Structural-to-body transform. Identity when absent (structure = body).
        // JEOD: mass.composite_properties.T_parent_this
        let t_struct_body = struct_xform.map_or(DMat3::IDENTITY, |s| s.0);

        // Build force contributions from optional interaction components.
        let mut contributions = ForceContributions::default();

        if let Some(aero) = aero {
            // JEOD_INV: IN.15 — aero drag requires body orientation (T_inertial_struct)
            if aero.force != DVec3::ZERO && rot_state.is_none() {
                panic!(
                    "AerodynamicForceC has non-zero force but RotationalStateC is missing. \
                     In JEOD, T_inertial_struct is a mandatory parameter of aero_drag(). \
                     Add RotationalStateC to any entity with aerodynamic forces."
                );
            }
            contributions.aero_force_struct = aero.force;
            contributions.aero_torque_struct = aero.torque;
        }

        if let Some(srp) = srp {
            contributions.srp_force_inertial = srp.force;
            contributions.srp_torque_struct = srp.torque;
        }

        if let Some(gt) = grav_torque {
            contributions.gravity_torque_body = gt.0;
        }

        // Rotation matrix from attitude quaternion (identity if no rotational state).
        let t_inertial_body = rot_state
            .as_ref()
            .map_or(DMat3::IDENTITY, |r| r.quaternion.left_quat_to_transformation());

        // Delegate frame-aware force/torque collection to jeod_dynamics (DB.28, DB.29).
        let collected = jeod_dynamics::collect_forces(&contributions, &t_struct_body, &t_inertial_body);
        total.force = collected.force;
        total.torque = collected.torque;

        // Delegate frame derivative computation to jeod_dynamics (FD.01, FD.02).
        if let Some(mut derivs) = derivs {
            let grav_accel = grav.map_or(DVec3::ZERO, |g| g.grav_accel);

            if let (Some(rot), Some(m)) = (rot_state, mass) {
                **derivs = jeod_dynamics::compute_frame_derivatives(
                    &collected,
                    m.mass,
                    grav_accel,
                    &m.inertia,
                    &m.inverse_inertia,
                    rot.ang_vel_body,
                );
            } else {
                // No rotational state or mass: translational-only derivatives.
                // Delegate to jeod_dynamics (FD.01).
                let mass_val = mass.map_or(0.0, |m| m.mass);
                **derivs = jeod_dynamics::compute_translational_derivatives(
                    collected.force,
                    mass_val,
                    grav_accel,
                );
            }
        }
    }
}

/// Advances translational (and optionally rotational) state via RK4 integration
/// with gravity re-evaluation at each stage.
///
/// Gravity is recomputed at each of the four RK4 intermediate positions for
/// 4th-order accuracy. Non-gravity accelerations from `TotalForceC` (aero, SRP)
/// are held constant over the timestep and added to the per-stage gravity.
///
/// Torques from `TotalForceC` (gravity gradient, aero torque) are similarly
/// held constant and passed to the 6-DOF integrator.
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
        // JEOD_INV: DB.07 — translational_dynamics gates integration (collection is unconditional; see force_collection_system)
        if !config.translational_dynamics {
            continue;
        }

        // Non-gravity translational acceleration (constant over one RK4 step).
        // TotalForceC.force holds only non-gravity forces (aero + SRP), already
        // in inertial frame. Delegate F=ma to jeod_dynamics (DB.18, MA.02).
        let non_grav_accel = if total_force.force == DVec3::ZERO {
            DVec3::ZERO
        } else if let Some(m) = mass {
            // JEOD_INV: MA.01 — MassBody always present on DynBody (partial: only checked when force != 0)
            jeod_dynamics::compute_translational_acceleration(total_force.force, m.mass)
        } else {
            panic!(
                "Entity {entity:?}: non-zero TotalForceC ({:?}) but no MassPropertiesC. \
                 In JEOD, DynBody always has mass. Add MassPropertiesC to any entity \
                 with interaction forces (drag, SRP).",
                total_force.force
            );
        };

        // Closure: compute gravitational acceleration at a given position.
        let compute_grav_accel = |position: DVec3| -> DVec3 {
            let mut accel = DVec3::ZERO;
            for ctrl in &controls.0.controls {
                // JEOD_INV: DM.08 — gravitation requires gravity source (source existence checked; "initialized" gate not enforced)
                // JEOD_INV: GV.12 — gravity source must exist for control (runtime panic)
                let Ok((source, rot)) = sources.get(ctrl.source_name) else {
                    panic!(
                        "Entity {entity:?}: GravityControl references entity {:?} which has no \
                         GravitySourceC. In JEOD, gravity source resolution is fatal. \
                         Ensure the source entity exists and has GravitySourceC before \
                         the first FixedUpdate tick.",
                        ctrl.source_name
                    );
                };

                // Pre-check: provide entity context before delegating to evaluate()
                // (GV.13, GV.17 enforced inside evaluate())
                if ctrl.is_nonspherical() && rot.is_none() {
                    panic!(
                        "Entity {entity:?}: non-spherical GravityControl references source {:?} \
                         which is missing PlanetFixedRotationC",
                        ctrl.source_name
                    );
                }
                accel += ctrl.evaluate(&source.0, position, rot.map(|r| &r.0)).grav_accel;
            }
            accel
        };

        // JEOD_INV: DB.08 — rotational_dynamics gates integration (collection is unconditional; see force_collection_system)
        // 6-DOF path: rotational dynamics enabled AND entity has components
        if config.rotational_dynamics {
            if let (Some(ref mut rot), Some(mass_props)) = (&mut rot_state, &mass) {
                let six_state = SixDofState {
                    trans: state.0,
                    rot: rot.0,
                };

                // Torque: constant over one RK4 step (gravity gradient + aero torque)
                let constant_torque = total_force.torque;

                let new_state = jeod_dynamics::rk4_sixdof_step(
                    &six_state,
                    |s| compute_grav_accel(s.trans.position) + non_grav_accel,
                    |_s| constant_torque,
                    &mass_props.0,
                    dt,
                );
                state.0 = new_state.trans;
                rot.0 = new_state.rot;
                continue;
            }
            // JEOD_INV: DB.04 — DynBody always has three frames (structure, composite_body, core_body)
            // and mass properties. Missing rotational state or mass with rotational_dynamics=true
            // is a configuration error, not a graceful fallback scenario.
            panic!(
                "Entity {entity:?} has rotational_dynamics=true but is missing RotationalStateC \
                 and/or MassPropertiesC. In JEOD, DynBody always has all three reference frames \
                 and mass properties. Add these components or set rotational_dynamics=false."
            );
        }

        // 3-DOF path: translational only
        let new_state = jeod_dynamics::rk4_translational_step(
            &state.0,
            |s| compute_grav_accel(s.position) + non_grav_accel,
            dt,
        );
        state.0 = new_state;
    }
}
