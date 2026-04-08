use glam::DVec3;
use jeod_dynamics::{
    DynamicsConfig, IntegratorType, MassProperties, RotationalState, SixDofState,
    TranslationalState,
};

/// Integrate a single body's state forward by one timestep.
///
/// Handles 6-DOF vs 3-DOF routing based on configuration flags and
/// available state. The gravity function is called at each integrator
/// intermediate state for proper multi-stage accuracy, matching JEOD's
/// `DynamicsIntegrationGroup` behavior where the derivative function
/// recomputes gravity at every stage.
///
/// Non-gravity forces and torques are held constant across stages (they
/// change negligibly over one timestep).
///
/// # Arguments
/// - `config`: dynamics flags (translational/rotational/three_dof)
/// - `trans`: translational state (mutated in place)
/// - `rot`: optional rotational state (mutated in place if 6-DOF)
/// - `mass`: mass properties (required for non-zero forces and 6-DOF)
/// - `gravity_fn`: computes gravitational acceleration from position (called per integrator stage)
/// - `non_grav_force`: total non-gravity force in inertial frame (constant over step)
/// - `torque`: total torque in body frame (constant over step)
/// - `dt`: timestep in seconds
/// - `integrator`: integration method to use
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
    gravity_fn: impl Fn(DVec3) -> DVec3,
    non_grav_force: DVec3,
    torque: DVec3,
    dt: f64,
    integrator: IntegratorType,
    gj_state: Option<&mut jeod_dynamics::GaussJacksonState>,
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

    // JEOD_INV: DB.08 — rotational_dynamics gates integration
    // 6-DOF path: rotational dynamics enabled AND components present
    if config.rotational_dynamics {
        if let (Some(rot), Some(mass_props)) = (rot, mass) {
            let six_state = SixDofState {
                trans: *trans,
                rot: *rot,
            };

            let constant_torque = torque;
            // Gravity recomputed at each integrator intermediate state for
            // multi-stage accuracy. Non-gravity acceleration held constant
            // (negligible change over one step).
            let accel = |s: &SixDofState| gravity_fn(s.trans.position) + non_grav_accel;
            let torque_fn = |_s: &SixDofState| constant_torque;
            let new_state = match integrator {
                IntegratorType::Rk4 => {
                    jeod_dynamics::rk4_sixdof_step(&six_state, accel, torque_fn, mass_props, dt)
                }
                IntegratorType::Rkf45 => {
                    jeod_dynamics::rkf45_sixdof_step(&six_state, accel, torque_fn, mass_props, dt)
                }
                IntegratorType::GaussJackson(..) => {
                    panic!(
                        "GaussJackson 6-DOF integration not yet supported. \
                         Set rotational_dynamics=false for GJ bodies."
                    );
                }
            };
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
    let accel = |s: &TranslationalState| gravity_fn(s.position) + non_grav_accel;
    match integrator {
        IntegratorType::Rk4 => {
            *trans = jeod_dynamics::rk4_translational_step(trans, accel, dt);
        }
        IntegratorType::Rkf45 => {
            *trans = jeod_dynamics::rkf45_translational_step(trans, accel, dt);
        }
        IntegratorType::GaussJackson(..) => {
            let gj = gj_state.expect(
                "GaussJackson integrator requires gj_state. \
                 Set SimBody::gj_state or call Simulation::validate() first.",
            );
            // Integration loop matching JEOD's IntegrationControls.
            // Stages are managed internally by the integrator.
            // Gravity is recomputed between stages at the predicted position.
            //
            // Stage cap prevents infinite loops if the FSM gets stuck.
            // JEOD's max is ~order*max_correction_iterations bootstrap edits
            // + 4 primer stages + 2 GJ stages; 1000 is generous.
            const MAX_STAGES: usize = 1000;
            for _ in 0..MAX_STAGES {
                let acc = gravity_fn(trans.position) + non_grav_accel;
                let result = gj.integrate(dt, acc, trans);
                if result.time_scale > 0.0 {
                    if !result.passed {
                        log::warn!(
                            "GaussJackson integration step did not converge \
                             (position may be degraded)"
                        );
                    }
                    break;
                }
            }
        }
    }
}
