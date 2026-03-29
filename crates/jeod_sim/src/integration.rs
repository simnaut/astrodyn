use glam::DVec3;
use jeod_dynamics::{
    DynamicsConfig, MassProperties, RotationalState, SixDofState, TranslationalState,
};

/// Integrate a single body's state forward by one timestep.
///
/// Handles 6-DOF vs 3-DOF routing based on configuration flags and
/// available state. Gravity and non-gravity accelerations are held
/// constant across all RK4 stages (matching JEOD's
/// `DynamicsIntegrationGroup` behavior).
///
/// # Arguments
/// - `config`: dynamics flags (translational/rotational/three_dof)
/// - `trans`: translational state (mutated in place)
/// - `rot`: optional rotational state (mutated in place if 6-DOF)
/// - `mass`: mass properties (required for non-zero forces and 6-DOF)
/// - `gravity_accel`: pre-computed gravitational acceleration (constant over step)
/// - `non_grav_force`: total non-gravity force in inertial frame (from force collection)
/// - `torque`: total torque in body frame (from force collection)
/// - `dt`: timestep in seconds
///
/// # Panics
/// - Non-zero force without mass properties (JEOD_INV: MA.01)
/// - `rotational_dynamics=true` without `RotationalState` or `MassProperties` (JEOD_INV: DB.04)
// JEOD_INV: DB.07 — translational_dynamics gates integration
// JEOD_INV: DB.08 — rotational_dynamics gates integration
// JEOD_INV: MA.01 — MassBody always present on DynBody (partial: checked when force != 0)
// JEOD_INV: DB.04 — DynBody always has three frames and mass properties
#[allow(clippy::too_many_arguments)]
pub fn integrate_body(
    config: &DynamicsConfig,
    trans: &mut TranslationalState,
    rot: Option<&mut RotationalState>,
    mass: Option<&MassProperties>,
    gravity_accel: DVec3,
    non_grav_force: DVec3,
    torque: DVec3,
    dt: f64,
) {
    // JEOD_INV: DB.07 — translational_dynamics gates integration
    if !config.translational_dynamics {
        return;
    }

    // Non-gravity translational acceleration (constant over one RK4 step).
    // JEOD_INV: DB.18 — force to acceleration via inverse mass
    let non_grav_accel = if non_grav_force == DVec3::ZERO {
        DVec3::ZERO
    } else if let Some(m) = mass {
        // JEOD_INV: MA.01 — MassBody always present on DynBody (partial: only checked when force != 0)
        jeod_dynamics::compute_translational_acceleration(non_grav_force, m.inverse_mass)
    } else {
        panic!(
            "Non-zero force ({non_grav_force:?}) but no MassProperties. \
             In JEOD, DynBody always has mass. Provide MassProperties for \
             any body with interaction forces (drag, SRP)."
        );
    };

    // Gravitational acceleration: pre-computed once per step, held constant
    // across all RK4 stages (matching JEOD DynamicsIntegrationGroup).
    let total_accel = gravity_accel + non_grav_accel;

    // JEOD_INV: DB.08 — rotational_dynamics gates integration
    // 6-DOF path: rotational dynamics enabled AND components present
    if config.rotational_dynamics {
        if let (Some(rot), Some(mass_props)) = (rot, mass) {
            let six_state = SixDofState {
                trans: *trans,
                rot: *rot,
            };

            let constant_torque = torque;
            let new_state = jeod_dynamics::rk4_sixdof_step(
                &six_state,
                |_s| total_accel,
                |_s| constant_torque,
                mass_props,
                dt,
            );
            *trans = new_state.trans;
            *rot = new_state.rot;
            return;
        }
        // JEOD_INV: DB.04 — DynBody always has three frames and mass properties
        panic!(
            "rotational_dynamics=true but RotationalState and/or MassProperties \
             missing. In JEOD, DynBody always has all three reference frames and \
             mass properties. Provide these or set rotational_dynamics=false."
        );
    }

    // 3-DOF path: translational only
    let new_trans = jeod_dynamics::rk4_translational_step(trans, |_s| total_accel, dt);
    *trans = new_trans;
}
