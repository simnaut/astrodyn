use bevy::prelude::*;
use glam::DVec3;
use jeod_dynamics::SixDofState;

use glam::DMat3;
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
    // Note: JEOD gates force/torque collection on translational/rotational_dynamics flags (DB.07/DB.08).
    // We collect unconditionally here; gating is enforced in integration_system.
    for (mut total, derivs, grav, rot_state, mass, aero, srp, grav_torque, struct_xform) in &mut query {
        let mut force = DVec3::ZERO;
        let mut torque = DVec3::ZERO;

        // Structural-to-body transform. Identity when absent (structure = body).
        // JEOD: mass.composite_properties.T_parent_this
        let t_struct_body = struct_xform.map_or(DMat3::IDENTITY, |s| s.0);

        // JEOD_INV: IN.15 — aero drag requires body orientation (T_inertial_struct)
        // JEOD's aero_drag() takes T_inertial_struct as a mandatory function parameter.
        // The rotation matrix is always available because DynBody always has three frames.
        //
        // Aero force is in structural frame (from compute_ballistic_drag).
        // JEOD dyn_body_collect.cc lines 219-221: structural→inertial via
        //   T_inertial_struct^T = (T_struct_body^T * T_inertial_body)^T
        //                       = T_inertial_body^T * T_struct_body
        if let Some(aero) = aero {
            if let Some(rot) = rot_state {
                let t_inertial_body = rot.quaternion.left_quat_to_transformation();
                let t_inertial_struct = t_struct_body.transpose() * t_inertial_body;
                force += t_inertial_struct.transpose() * aero.force;
            } else if aero.force != DVec3::ZERO {
                panic!(
                    "AerodynamicForceC has non-zero force but RotationalStateC is missing. \
                     In JEOD, T_inertial_struct is a mandatory parameter of aero_drag(). \
                     Add RotationalStateC to any entity with aerodynamic forces."
                );
            }
            // Aero torque is structural frame; convert to body.
            // JEOD dyn_body_collect.cc line 250: T_struct_body * torq_struct → torq_body
            torque += t_struct_body * aero.torque;
        }

        // SRP force: spherical model is already inertial (no rotation needed).
        // When flat-plate SRP is wired, force will be in structural frame and
        // must be rotated like aero above.
        if let Some(srp) = srp {
            force += srp.force;
            // SRP torque is structural frame; convert to body.
            torque += t_struct_body * srp.torque;
        }

        // Gravity gradient torque (already in body frame from compute_gravity_torque)
        if let Some(gt) = grav_torque {
            torque += gt.0;
        }

        total.force = force;
        total.torque = torque;

        // JEOD_INV: FD.01 — trans_accel = non_grav_accel + grav_accel
        // JEOD_INV: FD.02 — rot_accel = I^-1 * (tau - omega x I*omega)
        // Matches JEOD dyn_body_collect.cc lines 224-264:
        //   non_grav_accel = F_non_grav / m
        //   trans_accel = non_grav_accel + grav_accel
        //   rot_accel = I^-1 * (tau - omega x I*omega)
        if let Some(mut derivs) = derivs {
            let non_grav_accel = if let Some(m) = mass {
                if m.mass > 0.0 { force / m.mass } else { DVec3::ZERO }
            } else {
                DVec3::ZERO
            };

            let grav_accel = grav.map_or(DVec3::ZERO, |g| g.grav_accel);
            derivs.trans_accel = non_grav_accel + grav_accel;

            // Rotational acceleration from Euler's equation:
            // alpha = I^-1 * (tau - omega x (I * omega))
            derivs.rot_accel = if let (Some(rot), Some(m)) = (rot_state, mass) {
                jeod_dynamics::rotational::compute_rotational_acceleration(
                    &m.inertia,
                    &m.inverse_inertia,
                    rot.ang_vel_body,
                    torque,
                )
            } else {
                DVec3::ZERO
            };
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

        // JEOD_INV: DB.18 — F=ma (JEOD precomputes inverse_mass; we divide by mass at runtime)
        // JEOD_INV: MA.01 — MassBody always present on DynBody (partial: only checked when force != 0)
        // JEOD_INV: MA.02 — mass > 0 for meaningful dynamics (asserted before division)
        // Non-gravity translational acceleration (constant over one RK4 step).
        // TotalForceC.force holds only non-gravity forces (aero + SRP), already
        // in inertial frame. Divide by mass to get acceleration.
        let non_grav_accel = if total_force.force == DVec3::ZERO {
            DVec3::ZERO
        } else if let Some(m) = mass {
            assert!(
                m.mass > 0.0,
                "Entity {entity:?}: MassPropertiesC.mass must be positive for F=ma, got {}",
                m.mass
            );
            total_force.force / m.mass
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

                if ctrl.is_nonspherical() {
                    // JEOD_INV: GV.13 — gravity source must have inertial frame (PlanetFixedRotationC as proxy)
                    // JEOD_INV: GV.17 — active nonspherical controls subscribe to planet-fixed frame
                    let Some(r) = rot else {
                        panic!(
                            "Entity {entity:?}: GravityControl for source {:?} requests \
                             non-spherical gravity (degree={}/order={}) but source has no \
                             PlanetFixedRotationC. In JEOD, the planet-fixed frame is always \
                             subscribed for non-spherical gravity.",
                            ctrl.source_name, ctrl.degree, ctrl.order
                        );
                    };
                    accel += jeod_gravity::gravitation(
                        &source.0, position, &r.0,
                        ctrl.degree, ctrl.order, ctrl.perturbing_only,
                        false, 0, 0,
                    ).grav_accel;
                } else {
                    accel += jeod_gravity::gravitation(
                        &source.0, position, &glam::DMat3::IDENTITY,
                        0, 0, ctrl.perturbing_only,
                        false, 0, 0,
                    ).grav_accel;
                }
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
