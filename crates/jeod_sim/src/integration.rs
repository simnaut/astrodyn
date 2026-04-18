use glam::DVec3;
use jeod_dynamics::{
    DynamicsConfig, IntegratorType, MassProperties, RotationalState, SixDofState,
    TranslationalState,
};
use jeod_math::JeodQuat;

use crate::interactions::FlatPlateState;

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
/// - `gravity_fn`: computes gravitational acceleration from (position, velocity) per integrator stage
/// - `non_grav_force`: total non-gravity force in inertial frame (constant over step)
/// - `torque`: total torque in body frame (constant over step)
/// - `dt`: simulation timestep in seconds (JEOD: `sim_dt`)
/// - `time_scale_factor`: ratio of dynamic time to simulation time
///   (JEOD: `TimeDyn::scale_factor`). Applied uniformly to all integrators:
///   RK4/RKF45/ABM4 use `integ_dyndt = dt * time_scale_factor`, Gauss-Jackson
///   uses `cycle_dyndt = dt * cycle_scale * time_scale_factor`.
/// - `integrator`: integration method to use
/// - `gj_state`: persistent Gauss-Jackson state, required (and only used)
///   when `integrator == IntegratorType::GaussJackson`. Panics otherwise if
///   absent. Caller retains one per body; see `Simulation::validate` /
///   `GaussJacksonStateC` for the runner/Bevy wiring.
/// - `abm4_state`: persistent ABM4 history, required (and only used) when
///   `integrator == IntegratorType::Abm4`. Panics otherwise if absent. Caller
///   retains one per body; see `Simulation::validate` / `Abm4StateC` for the
///   runner/Bevy wiring.
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
    gravity_fn: impl Fn(DVec3, DVec3, f64) -> DVec3,
    non_grav_force: DVec3,
    torque: DVec3,
    dt: f64,
    time_scale_factor: f64,
    integrator: IntegratorType,
    gj_state: Option<&mut jeod_dynamics::GaussJacksonState>,
    abm4_state: Option<&mut jeod_dynamics::Abm4State>,
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

    // JEOD: integ_dyndt = sim_dt * time_scale_factor
    // (single_cycle_integration_controls.cc:63-65, standard_integration_controls.cc:80-82)
    let integ_dyndt = dt * time_scale_factor;

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
            let accel = |s: &SixDofState, time_frac: f64| {
                gravity_fn(s.trans.position, s.trans.velocity, time_frac) + non_grav_accel
            };
            let torque_fn = |_s: &SixDofState| constant_torque;
            let new_state = match integrator {
                IntegratorType::Rk4 => jeod_dynamics::rk4_sixdof_step(
                    &six_state,
                    accel,
                    torque_fn,
                    mass_props,
                    integ_dyndt,
                ),
                IntegratorType::Rkf45 => jeod_dynamics::rkf45_sixdof_step(
                    &six_state,
                    accel,
                    torque_fn,
                    mass_props,
                    integ_dyndt,
                ),
                IntegratorType::GaussJackson(..) => {
                    panic!(
                        "GaussJackson 6-DOF integration not yet supported. \
                         Set rotational_dynamics=false for GJ bodies."
                    );
                }
                IntegratorType::Abm4 => {
                    panic!(
                        "ABM4 6-DOF integration not yet supported. \
                         Set rotational_dynamics=false for ABM4 bodies."
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
    let accel = |s: &TranslationalState, time_frac: f64| {
        gravity_fn(s.position, s.velocity, time_frac) + non_grav_accel
    };
    match integrator {
        IntegratorType::Rk4 => {
            *trans = jeod_dynamics::rk4_translational_step(trans, accel, integ_dyndt);
        }
        IntegratorType::Rkf45 => {
            *trans = jeod_dynamics::rkf45_translational_step(trans, accel, integ_dyndt);
        }
        IntegratorType::Abm4 => {
            let abm = abm4_state.expect(
                "ABM4 integrator requires a persistent Abm4State passed in via abm4_state. \
                 Runner: call Simulation::validate(); Bevy: add Abm4StateC.",
            );
            *trans = jeod_dynamics::abm4_translational_step(trans, accel, integ_dyndt, abm);
        }
        IntegratorType::GaussJackson(cfg) => {
            let gj = gj_state.expect(
                "GaussJackson integrator requires gj_state. \
                 Set SimBody::gj_state or call Simulation::validate() first.",
            );
            debug_assert_eq!(
                gj.config(),
                &cfg,
                "GaussJacksonState config does not match IntegratorType config. \
                 Recreate the state from the same config or call Simulation::validate()."
            );
            // Integration loop matching JEOD's IntegrationControls.
            // Stages are managed internally by the integrator.
            // Gravity is recomputed between stages at the predicted position.
            //
            // Stage cap from the state's actual config (not the IntegratorType
            // config, which could differ if constructed manually).
            // Worst case per step: primer (4 stages) + bootstrap edit
            // (order * max_correction_iterations) + GJ predict/correct (2).
            let max_stages = {
                let cfg = gj.config();
                // Cap ndoubling_steps to avoid huge tour_count (JEOD default: 4).
                let capped_ndoubling = cfg.ndoubling_steps.min(10);
                let tour_count = 1usize << capped_ndoubling;
                let edits = cfg
                    .final_order
                    .saturating_mul(cfg.max_correction_iterations + 1);
                edits
                    .saturating_add(10)
                    .saturating_mul(tour_count)
                    .clamp(100, 10_000_000) // hard cap: prevent runaway loops
            };
            let mut completed = false;
            for _ in 0..max_stages {
                let acc = gravity_fn(trans.position, trans.velocity, 0.0) + non_grav_accel;
                let result = gj.integrate(dt, time_scale_factor, acc, trans);
                if result.time_scale > 0.0 {
                    if !result.passed {
                        log::warn!(
                            "GaussJackson integration step did not converge \
                             (position may be degraded)"
                        );
                    }
                    completed = true;
                    break;
                }
            }
            assert!(
                completed,
                "GaussJackson integration did not complete within {max_stages} stages. \
                 The FSM may be stuck. Reset the integrator or check configuration."
            );
        }
    }
}

/// Evaluation results at one RK4 stage, returned by the stage closure
/// passed to [`integrate_body_coupled`].
///
/// The closure recomputes gravity, SRP force/torque, and thermal derivatives
/// at each intermediate state, coupling the thermal and orbital ODEs.
#[derive(Debug, Clone)]
pub struct CoupledStageEval {
    /// Gravitational acceleration (m/s²) at the intermediate position.
    pub gravity_accel: DVec3,
    /// Total non-gravity force (N) in inertial frame, including SRP with
    /// thermal emission computed from the intermediate temperature state.
    pub non_grav_force: DVec3,
    /// Total torque (N·m) in body frame.
    pub torque: DVec3,
    /// Per-plate temperature derivatives (K/s) at the intermediate state.
    /// Same length as `FlatPlateState::temperatures`.
    pub temp_dots: Vec<f64>,
}

/// Integrate orbital + thermal state through a coupled RK4 step.
///
/// Port of JEOD's `DynamicsIntegrationGroup::integrate_bodies()` which drives
/// both `DynBody` (orbital) and `ThermalIntegrableObject` (thermal) through the
/// same RK4 stage loop. At each stage the `stage_fn` closure is called with the
/// intermediate orbital and thermal state, returning derivatives for both.
///
/// This ensures that:
/// - Temperature derivatives use the intermediate orbital position (solar flux
///   direction changes across stages)
/// - SRP force uses the intermediate temperatures (thermal emission coupling)
/// - Both ODEs see the same RK4 stage structure
///
/// JEOD's `DynamicsIntegrationGroup` is integrator-agnostic (RK4, RKF45, GJ,
/// LSODE). This implementation covers RK4 only — the only integrator for which
/// JEOD verification sims exercise thermal coupling.
///
/// # Arguments
/// - `config`: dynamics flags (translational/rotational)
/// - `trans`: translational state (mutated in place)
/// - `rot`: optional rotational state (mutated in place if 6-DOF)
/// - `mass`: mass properties (required for non-zero forces and 6-DOF)
/// - `stage_fn`: closure that evaluates derivatives at an intermediate state.
///   Called 4 times (once per RK4 stage). Must recompute gravity, SRP
///   force/torque, and thermal derivatives at the given intermediate state.
/// - `thermal`: flat-plate thermal state (temperatures mutated in place)
/// - `dt`: simulation timestep in seconds
/// - `time_scale_factor`: ratio of dynamic time to simulation time
///
/// # Panics
/// - Non-zero force without mass properties (JEOD_INV: MA.01)
/// - `rotational_dynamics=true` without `RotationalState` or `MassProperties` (JEOD_INV: DB.04)
// JEOD_INV: DB.07 — translational_dynamics gates integration
// JEOD_INV: DB.08 — rotational_dynamics gates integration
#[allow(clippy::too_many_arguments)]
pub fn integrate_body_coupled(
    config: &DynamicsConfig,
    trans: &mut TranslationalState,
    rot: Option<&mut RotationalState>,
    mass: Option<&MassProperties>,
    mut stage_fn: impl FnMut(
        &TranslationalState,
        Option<&RotationalState>,
        &FlatPlateState,
    ) -> CoupledStageEval,
    thermal: &mut FlatPlateState,
    dt: f64,
    time_scale_factor: f64,
) {
    // JEOD_INV: DB.07 — translational_dynamics gates integration
    if !config.translational_dynamics {
        return;
    }

    let integ_dyndt = dt * time_scale_factor;
    let n_plates = thermal.temperatures.len();

    // Dispatch to 6-DOF or 3-DOF coupled path.
    if config.rotational_dynamics {
        if let (Some(rot_state), Some(mass_props)) = (rot, mass) {
            integrate_coupled_sixdof(
                trans,
                rot_state,
                mass_props,
                &mut stage_fn,
                thermal,
                integ_dyndt,
                n_plates,
            );
            return;
        }
        // JEOD_INV: DB.04 — rotational integration requires all three frames (composite/structure/core);
        // here that maps to needing both RotationalState and MassProperties present together.
        panic!(
            "rotational_dynamics=true but RotationalState and/or MassProperties \
             missing. Provide these or set rotational_dynamics=false."
        );
    }

    // ── 3-DOF coupled RK4 (translational + thermal) ──

    let pos0 = trans.position;
    let vel0 = trans.velocity;
    let temps0: Vec<f64> = thermal.temperatures.clone();
    let t_pow4_0: Vec<f64> = thermal.t_pow4_cached.clone();

    // Stage 1: evaluate at current state
    let eval1 = stage_fn(trans, None, thermal);
    debug_assert_eq!(
        eval1.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 1"
    );
    let k1_accel = compute_total_accel(&eval1, mass);
    let k1_v = vel0;
    let k1_tdots = &eval1.temp_dots;

    // Stage 2: evaluate at t + dt/2 using k1
    let half_dt = integ_dyndt * 0.5;
    let s2_trans = TranslationalState {
        position: pos0 + k1_v * half_dt,
        velocity: vel0 + k1_accel * half_dt,
    };
    advance_thermal_intermediate(thermal, &temps0, &t_pow4_0, k1_tdots, half_dt);
    let eval2 = stage_fn(&s2_trans, None, thermal);
    debug_assert_eq!(
        eval2.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 2"
    );
    let k2_accel = compute_total_accel(&eval2, mass);
    let k2_v = s2_trans.velocity;
    let k2_tdots = eval2.temp_dots.clone();

    // Stage 3: evaluate at t + dt/2 using k2
    let s3_trans = TranslationalState {
        position: pos0 + k2_v * half_dt,
        velocity: vel0 + k2_accel * half_dt,
    };
    advance_thermal_intermediate(thermal, &temps0, &t_pow4_0, &k2_tdots, half_dt);
    let eval3 = stage_fn(&s3_trans, None, thermal);
    debug_assert_eq!(
        eval3.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 3"
    );
    let k3_accel = compute_total_accel(&eval3, mass);
    let k3_v = s3_trans.velocity;
    let k3_tdots = eval3.temp_dots.clone();

    // Stage 4: evaluate at t + dt using k3
    let s4_trans = TranslationalState {
        position: pos0 + k3_v * integ_dyndt,
        velocity: vel0 + k3_accel * integ_dyndt,
    };
    advance_thermal_intermediate(thermal, &temps0, &t_pow4_0, &k3_tdots, integ_dyndt);
    let eval4 = stage_fn(&s4_trans, None, thermal);
    debug_assert_eq!(
        eval4.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 4"
    );
    let k4_accel = compute_total_accel(&eval4, mass);
    let k4_v = s4_trans.velocity;

    // Combine orbital state
    let sixth_dt = integ_dyndt / 6.0;
    trans.position = pos0 + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * sixth_dt;
    trans.velocity = vel0 + (k1_accel + k2_accel * 2.0 + k3_accel * 2.0 + k4_accel) * sixth_dt;

    // Combine thermal state with overshoot clamping
    thermal.finalize_rk4_temperatures(
        &temps0,
        &t_pow4_0,
        k1_tdots,
        &k2_tdots,
        &k3_tdots,
        &eval4.temp_dots,
        integ_dyndt,
        n_plates,
    );
}

/// 6-DOF coupled RK4: translational + rotational + thermal.
#[allow(clippy::too_many_arguments)]
fn integrate_coupled_sixdof(
    trans: &mut TranslationalState,
    rot: &mut RotationalState,
    mass_props: &MassProperties,
    stage_fn: &mut impl FnMut(
        &TranslationalState,
        Option<&RotationalState>,
        &FlatPlateState,
    ) -> CoupledStageEval,
    thermal: &mut FlatPlateState,
    integ_dyndt: f64,
    n_plates: usize,
) {
    let pos0 = trans.position;
    let vel0 = trans.velocity;
    let q0 = rot.quaternion.data;
    let omega0 = rot.ang_vel_body;
    let temps0: Vec<f64> = thermal.temperatures.clone();
    let t_pow4_0: Vec<f64> = thermal.t_pow4_cached.clone();

    let make_rot = |q: [f64; 4], omega: DVec3| RotationalState {
        quaternion: JeodQuat::new(q[0], q[1], q[2], q[3]),
        ang_vel_body: omega,
    };

    let step_q = |q_base: [f64; 4], k_qdot: [f64; 4], h: f64| -> [f64; 4] {
        [
            q_base[0] + k_qdot[0] * h,
            q_base[1] + k_qdot[1] * h,
            q_base[2] + k_qdot[2] * h,
            q_base[3] + k_qdot[3] * h,
        ]
    };

    // Helper: compute orbital + rotational derivatives from eval
    let eval_rot_derivs = |eval: &CoupledStageEval,
                           rot_s: &RotationalState|
     -> (DVec3, [f64; 4], DVec3) {
        let accel = compute_total_accel(eval, Some(mass_props));
        let k_qdot = jeod_dynamics::compute_left_quat_deriv(&rot_s.quaternion, rot_s.ang_vel_body);
        let k_alpha = jeod_dynamics::compute_rotational_acceleration(
            &mass_props.inertia,
            &mass_props.inverse_inertia,
            rot_s.ang_vel_body,
            eval.torque,
        );
        (accel, k_qdot, k_alpha)
    };

    // Stage 1
    let eval1 = stage_fn(trans, Some(rot), thermal);
    debug_assert_eq!(
        eval1.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 1"
    );
    let (k1_accel, k1_qdot, k1_alpha) = eval_rot_derivs(&eval1, rot);
    let k1_v = vel0;
    let k1_tdots = &eval1.temp_dots;

    // Stage 2
    let half_dt = integ_dyndt * 0.5;
    let s2_trans = TranslationalState {
        position: pos0 + k1_v * half_dt,
        velocity: vel0 + k1_accel * half_dt,
    };
    let s2_rot = make_rot(step_q(q0, k1_qdot, half_dt), omega0 + k1_alpha * half_dt);
    advance_thermal_intermediate(thermal, &temps0, &t_pow4_0, k1_tdots, half_dt);
    let eval2 = stage_fn(&s2_trans, Some(&s2_rot), thermal);
    debug_assert_eq!(
        eval2.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 2"
    );
    let (k2_accel, k2_qdot, k2_alpha) = eval_rot_derivs(&eval2, &s2_rot);
    let k2_v = s2_trans.velocity;
    let k2_tdots = eval2.temp_dots.clone();

    // Stage 3
    let s3_trans = TranslationalState {
        position: pos0 + k2_v * half_dt,
        velocity: vel0 + k2_accel * half_dt,
    };
    let s3_rot = make_rot(step_q(q0, k2_qdot, half_dt), omega0 + k2_alpha * half_dt);
    advance_thermal_intermediate(thermal, &temps0, &t_pow4_0, &k2_tdots, half_dt);
    let eval3 = stage_fn(&s3_trans, Some(&s3_rot), thermal);
    debug_assert_eq!(
        eval3.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 3"
    );
    let (k3_accel, k3_qdot, k3_alpha) = eval_rot_derivs(&eval3, &s3_rot);
    let k3_v = s3_trans.velocity;
    let k3_tdots = eval3.temp_dots.clone();

    // Stage 4
    let s4_trans = TranslationalState {
        position: pos0 + k3_v * integ_dyndt,
        velocity: vel0 + k3_accel * integ_dyndt,
    };
    let s4_rot = make_rot(
        step_q(q0, k3_qdot, integ_dyndt),
        omega0 + k3_alpha * integ_dyndt,
    );
    advance_thermal_intermediate(thermal, &temps0, &t_pow4_0, &k3_tdots, integ_dyndt);
    let eval4 = stage_fn(&s4_trans, Some(&s4_rot), thermal);
    debug_assert_eq!(
        eval4.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 4"
    );
    let (k4_accel, k4_qdot, k4_alpha) = eval_rot_derivs(&eval4, &s4_rot);
    let k4_v = s4_trans.velocity;

    // Combine orbital state
    let sixth_dt = integ_dyndt / 6.0;
    trans.position = pos0 + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * sixth_dt;
    trans.velocity = vel0 + (k1_accel + k2_accel * 2.0 + k3_accel * 2.0 + k4_accel) * sixth_dt;

    // Combine rotational state
    rot.ang_vel_body = omega0 + (k1_alpha + k2_alpha * 2.0 + k3_alpha * 2.0 + k4_alpha) * sixth_dt;
    let final_q = [
        q0[0] + (k1_qdot[0] + 2.0 * k2_qdot[0] + 2.0 * k3_qdot[0] + k4_qdot[0]) * sixth_dt,
        q0[1] + (k1_qdot[1] + 2.0 * k2_qdot[1] + 2.0 * k3_qdot[1] + k4_qdot[1]) * sixth_dt,
        q0[2] + (k1_qdot[2] + 2.0 * k2_qdot[2] + 2.0 * k3_qdot[2] + k4_qdot[2]) * sixth_dt,
        q0[3] + (k1_qdot[3] + 2.0 * k2_qdot[3] + 2.0 * k3_qdot[3] + k4_qdot[3]) * sixth_dt,
    ];
    // JEOD_INV: DB.09 — quaternion normalized after every integration step
    rot.quaternion = JeodQuat::new(final_q[0], final_q[1], final_q[2], final_q[3]);
    jeod_dynamics::normalize_integ(&mut rot.quaternion);

    // Combine thermal state with overshoot clamping
    thermal.finalize_rk4_temperatures(
        &temps0,
        &t_pow4_0,
        k1_tdots,
        &k2_tdots,
        &k3_tdots,
        &eval4.temp_dots,
        integ_dyndt,
        n_plates,
    );
}

/// Compute total translational acceleration from a stage evaluation.
fn compute_total_accel(eval: &CoupledStageEval, mass: Option<&MassProperties>) -> DVec3 {
    let non_grav_accel = if eval.non_grav_force == DVec3::ZERO {
        DVec3::ZERO
    } else if let Some(m) = mass {
        // JEOD_INV: DB.18 — force to acceleration via inverse mass
        jeod_dynamics::compute_translational_acceleration(eval.non_grav_force, m.inverse_mass)
    } else {
        panic!(
            "Non-zero force ({:?}) but no MassProperties. \
             Provide MassProperties for any body with interaction forces.",
            eval.non_grav_force
        );
    };
    eval.gravity_accel + non_grav_accel
}

/// Advance thermal state to an intermediate RK4 position.
///
/// Sets temperatures to `temps0 + k_tdots * h` and updates cached T^4.
/// Used between RK4 stages so the next `stage_fn` call sees intermediate
/// temperatures for thermal emission force coupling.
fn advance_thermal_intermediate(
    thermal: &mut FlatPlateState,
    temps0: &[f64],
    _t_pow4_0: &[f64],
    k_tdots: &[f64],
    h: f64,
) {
    for i in 0..thermal.temperatures.len() {
        let t = (temps0[i] + k_tdots[i] * h).max(0.0);
        thermal.temperatures[i] = t;
        thermal.t_pow4_cached[i] = t * t * t * t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interactions::FlatPlateState;

    /// Verify coupled path produces identical results to standard path
    /// when forces are held constant (no thermal coupling).
    #[test]
    fn coupled_matches_standard_constant_force() {
        let config = DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        };
        let mass = jeod_dynamics::MassProperties::new(500.0);
        let dt = 1.0;
        let tsf = 1.0;
        let mu = 3.986004418e14;

        let pos0 = DVec3::new(6.778e6, 0.0, 0.0);
        let vel0 = DVec3::new(0.0, 7672.0, 0.0);
        let srp_force = DVec3::new(1e-4, 2e-4, 3e-4);

        // Standard path
        let mut trans1 = TranslationalState {
            position: pos0,
            velocity: vel0,
        };
        integrate_body(
            &config,
            &mut trans1,
            None,
            Some(&mass),
            |pos, _vel, _time_frac| -mu / pos.length().powi(3) * pos,
            srp_force,
            DVec3::ZERO,
            dt,
            tsf,
            jeod_dynamics::IntegratorType::Rk4,
            None,
            None,
        );

        // Coupled path with constant forces (no thermal state)
        let mut trans2 = TranslationalState {
            position: pos0,
            velocity: vel0,
        };
        let mut thermal = FlatPlateState {
            plates: vec![],
            temperatures: vec![],
            t_pow4_cached: vec![],
        };
        integrate_body_coupled(
            &config,
            &mut trans2,
            None,
            Some(&mass),
            |inter_trans, _inter_rot, _inter_thermal| CoupledStageEval {
                gravity_accel: {
                    let pos = inter_trans.position;
                    -mu / pos.length().powi(3) * pos
                },
                non_grav_force: srp_force,
                torque: DVec3::ZERO,
                temp_dots: vec![],
            },
            &mut thermal,
            dt,
            tsf,
        );

        let pos_diff = (trans1.position - trans2.position).length();
        let vel_diff = (trans1.velocity - trans2.velocity).length();
        eprintln!("Standard pos: {:?}", trans1.position);
        eprintln!("Coupled  pos: {:?}", trans2.position);
        eprintln!("Position diff: {:.6e} m", pos_diff);
        eprintln!("Velocity diff: {:.6e} m/s", vel_diff);

        assert!(pos_diff < 1e-10, "Position mismatch: {pos_diff:.6e} m");
        assert!(vel_diff < 1e-10, "Velocity mismatch: {vel_diff:.6e} m/s");
    }
}
