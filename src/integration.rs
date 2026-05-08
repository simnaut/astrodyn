// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Integration stage: stepping per-body translational + rotational state
//! by one timestep using the configured integrator (RK4, ABM4, or
//! Gauss-Jackson). Includes the contact-coupled multi-body RK4 entry
//! points used when contact pairs are registered.

use astrodyn_dynamics::state::TranslationalStateTyped;
use astrodyn_dynamics::{
    DynamicsConfig, MassProperties, RotationalState, SixDofState, TranslationalState,
};
use glam::DVec3;

use crate::integrator::IntegratorType;
use astrodyn_math::JeodQuat;
use astrodyn_quantities::aliases::{Acceleration, Force, Position, Torque, Velocity};
use astrodyn_quantities::frame::{BodyFrame, Frame, Vehicle};
use uom::si::f64::Time;

use crate::integrable::IntegrableObject;
use crate::interactions::FlatPlateState;

/// Per-body state for multi-body coupled RK4 integration.
///
/// Used by [`integrate_bodies_contact_coupled`] to pass per-body inputs through
/// the RK4 stages. Each body's per-stage gravity acceleration is recomputed
/// via `gravity_fn`; non-gravity (non-contact) force and body-frame torque are
/// held constant across stages (matching JEOD where aero/SRP/gravity-torque
/// are "scheduled" jobs evaluated once per step).
pub struct CoupledBodyInput<'a> {
    /// Translational state to advance (mutated in place).
    pub trans: &'a mut TranslationalState,
    /// Rotational state to advance (mutated in place). 6-DOF only.
    pub rot: &'a mut RotationalState,
    /// Mass properties.
    pub mass: &'a MassProperties,
    /// Constant non-gravity, non-contact inertial force over the step (aero,
    /// SRP, external force). Added to gravity + contact in the accel function.
    pub non_grav_non_contact_force: DVec3,
    /// Constant body-frame torque from non-contact sources (gravity gradient,
    /// SRP torque, aero torque, external torque).
    pub non_contact_torque_body: DVec3,
}

/// Reusable scratch buffers for [`integrate_bodies_contact_coupled`].
///
/// Owned by the caller and reused across integration steps so the inner
/// RK4 loop performs no heap allocations once the body count has
/// stabilized. All fields are resized to `n_bodies` on entry and the
/// backing storage is retained between calls.
#[derive(Default)]
pub struct CoupledIntegScratch {
    // Initial-state snapshots (one per body).
    pos0: Vec<DVec3>,
    vel0: Vec<DVec3>,
    q0: Vec<[f64; 4]>,
    omega0: Vec<DVec3>,
    // Per-stage intermediate state arrays (stage 0 is unused — stage 1
    // reads directly from the snapshots above). Indices 1..=3 hold the
    // inputs for stages 2, 3, 4 respectively.
    stage_pos: [Vec<DVec3>; 4],
    stage_vel: [Vec<DVec3>; 4],
    stage_q: [Vec<[f64; 4]>; 4],
    stage_omega: [Vec<DVec3>; 4],
    // Per-stage derivatives (k1..k4).
    k_v: [Vec<DVec3>; 4],
    k_a: [Vec<DVec3>; 4],
    k_qdot: [Vec<[f64; 4]>; 4],
    k_alpha: [Vec<DVec3>; 4],
    // Scratch for assembling TranslationalState / RotationalState slices
    // consumed by `contact_eval`.
    stage_trans: Vec<TranslationalState>,
    stage_rot: Vec<RotationalState>,
    // Scratch for per-body contact force/torque outputs populated by
    // `contact_eval` each stage.
    contact_out: Vec<(DVec3, DVec3)>,
}

impl CoupledIntegScratch {
    /// Create a fresh scratch (all buffers empty; grow on first use).
    pub fn new() -> Self {
        Self::default()
    }

    fn resize(&mut self, n: usize) {
        self.pos0.resize(n, DVec3::ZERO);
        self.vel0.resize(n, DVec3::ZERO);
        self.q0.resize(n, [0.0; 4]);
        self.omega0.resize(n, DVec3::ZERO);
        for buf in &mut self.stage_pos {
            buf.resize(n, DVec3::ZERO);
        }
        for buf in &mut self.stage_vel {
            buf.resize(n, DVec3::ZERO);
        }
        for buf in &mut self.stage_q {
            buf.resize(n, [0.0; 4]);
        }
        for buf in &mut self.stage_omega {
            buf.resize(n, DVec3::ZERO);
        }
        for buf in &mut self.k_v {
            buf.resize(n, DVec3::ZERO);
        }
        for buf in &mut self.k_a {
            buf.resize(n, DVec3::ZERO);
        }
        for buf in &mut self.k_qdot {
            buf.resize(n, [0.0; 4]);
        }
        for buf in &mut self.k_alpha {
            buf.resize(n, DVec3::ZERO);
        }
        self.stage_trans.resize(
            n,
            TranslationalState {
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            },
        );
        self.stage_rot.resize(
            n,
            RotationalState {
                quaternion: JeodQuat::new(1.0, 0.0, 0.0, 0.0),
                ang_vel_body: DVec3::ZERO,
            },
        );
        self.contact_out.resize(n, (DVec3::ZERO, DVec3::ZERO));
    }
}

/// Multi-body coupled RK4 step where contact forces between bodies are
/// recomputed at each of the four stages.
///
/// This matches JEOD's `IntegLoop sim_integ_loop(DYNAMICS) dynamics, contact,
/// veh1_dyn, veh2_dyn;` pattern where `contact.check_contact()` is a derivative
/// class job — i.e., the contact force is evaluated at every derivative
/// evaluation within the RK4 stages, not once per outer step.
///
/// - `bodies`: one per body, in the same indexing as used by `contact_eval`.
/// - `scratch`: preallocated working buffers reused across calls; no
///   allocations occur in the inner loop once the body count stabilizes.
/// - `gravity_fn(body_idx, position, velocity, time_frac) -> accel`: per-body
///   gravity at an intermediate state.
/// - `contact_eval(stage_trans, stage_rot, out)`: given the intermediate
///   states of ALL bodies at the current RK4 stage, populate `out[i]`
///   with the inertial force and body-frame torque on body `i`.
/// - `dt`: integration timestep in dynamic seconds (`sim_dt * time_scale_factor`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn integrate_bodies_contact_coupled(
    bodies: &mut [CoupledBodyInput<'_>],
    scratch: &mut CoupledIntegScratch,
    mut gravity_fn: impl FnMut(usize, DVec3, DVec3, f64) -> DVec3,
    mut contact_eval: impl FnMut(&[TranslationalState], &[RotationalState], &mut [(DVec3, DVec3)]),
    dt: f64,
) {
    let n = bodies.len();
    if n == 0 {
        return;
    }

    scratch.resize(n);

    // Snapshot initial states into reusable buffers.
    for (i, body) in bodies.iter().enumerate() {
        scratch.pos0[i] = body.trans.position;
        scratch.vel0[i] = body.trans.velocity;
        scratch.q0[i] = body.rot.quaternion.data;
        scratch.omega0[i] = body.rot.ang_vel_body;
    }

    // Stage 1 (t=0): initial state is the snapshot itself.
    eval_stage(
        &scratch.pos0,
        &scratch.vel0,
        &scratch.q0,
        &scratch.omega0,
        0.0,
        &mut gravity_fn,
        &mut contact_eval,
        bodies,
        &mut scratch.stage_trans,
        &mut scratch.stage_rot,
        &mut scratch.contact_out,
        &mut scratch.k_v[0],
        &mut scratch.k_a[0],
        &mut scratch.k_qdot[0],
        &mut scratch.k_alpha[0],
    );

    // Build stage 2 state, then evaluate.
    let half = dt * 0.5;
    fill_stage_state(
        n,
        &scratch.pos0,
        &scratch.vel0,
        &scratch.q0,
        &scratch.omega0,
        &scratch.k_v[0],
        &scratch.k_a[0],
        &scratch.k_qdot[0],
        &scratch.k_alpha[0],
        half,
        &mut scratch.stage_pos[1],
        &mut scratch.stage_vel[1],
        &mut scratch.stage_q[1],
        &mut scratch.stage_omega[1],
    );
    eval_stage(
        &scratch.stage_pos[1],
        &scratch.stage_vel[1],
        &scratch.stage_q[1],
        &scratch.stage_omega[1],
        0.5,
        &mut gravity_fn,
        &mut contact_eval,
        bodies,
        &mut scratch.stage_trans,
        &mut scratch.stage_rot,
        &mut scratch.contact_out,
        &mut scratch.k_v[1],
        &mut scratch.k_a[1],
        &mut scratch.k_qdot[1],
        &mut scratch.k_alpha[1],
    );

    // Build stage 3 state, then evaluate.
    fill_stage_state(
        n,
        &scratch.pos0,
        &scratch.vel0,
        &scratch.q0,
        &scratch.omega0,
        &scratch.k_v[1],
        &scratch.k_a[1],
        &scratch.k_qdot[1],
        &scratch.k_alpha[1],
        half,
        &mut scratch.stage_pos[2],
        &mut scratch.stage_vel[2],
        &mut scratch.stage_q[2],
        &mut scratch.stage_omega[2],
    );
    eval_stage(
        &scratch.stage_pos[2],
        &scratch.stage_vel[2],
        &scratch.stage_q[2],
        &scratch.stage_omega[2],
        0.5,
        &mut gravity_fn,
        &mut contact_eval,
        bodies,
        &mut scratch.stage_trans,
        &mut scratch.stage_rot,
        &mut scratch.contact_out,
        &mut scratch.k_v[2],
        &mut scratch.k_a[2],
        &mut scratch.k_qdot[2],
        &mut scratch.k_alpha[2],
    );

    // Build stage 4 state (full step using k3), then evaluate.
    fill_stage_state(
        n,
        &scratch.pos0,
        &scratch.vel0,
        &scratch.q0,
        &scratch.omega0,
        &scratch.k_v[2],
        &scratch.k_a[2],
        &scratch.k_qdot[2],
        &scratch.k_alpha[2],
        dt,
        &mut scratch.stage_pos[3],
        &mut scratch.stage_vel[3],
        &mut scratch.stage_q[3],
        &mut scratch.stage_omega[3],
    );
    eval_stage(
        &scratch.stage_pos[3],
        &scratch.stage_vel[3],
        &scratch.stage_q[3],
        &scratch.stage_omega[3],
        1.0,
        &mut gravity_fn,
        &mut contact_eval,
        bodies,
        &mut scratch.stage_trans,
        &mut scratch.stage_rot,
        &mut scratch.contact_out,
        &mut scratch.k_v[3],
        &mut scratch.k_a[3],
        &mut scratch.k_qdot[3],
        &mut scratch.k_alpha[3],
    );

    // Combine k1..k4 into the final state per body.
    let sixth = dt / 6.0;
    for (i, body) in bodies.iter_mut().enumerate() {
        let (kv1, kv2, kv3, kv4) = (
            scratch.k_v[0][i],
            scratch.k_v[1][i],
            scratch.k_v[2][i],
            scratch.k_v[3][i],
        );
        let (ka1, ka2, ka3, ka4) = (
            scratch.k_a[0][i],
            scratch.k_a[1][i],
            scratch.k_a[2][i],
            scratch.k_a[3][i],
        );
        let (kal1, kal2, kal3, kal4) = (
            scratch.k_alpha[0][i],
            scratch.k_alpha[1][i],
            scratch.k_alpha[2][i],
            scratch.k_alpha[3][i],
        );
        let (kq1, kq2, kq3, kq4) = (
            scratch.k_qdot[0][i],
            scratch.k_qdot[1][i],
            scratch.k_qdot[2][i],
            scratch.k_qdot[3][i],
        );

        body.trans.position = scratch.pos0[i] + (kv1 + kv2 * 2.0 + kv3 * 2.0 + kv4) * sixth;
        body.trans.velocity = scratch.vel0[i] + (ka1 + ka2 * 2.0 + ka3 * 2.0 + ka4) * sixth;
        body.rot.ang_vel_body = scratch.omega0[i] + (kal1 + kal2 * 2.0 + kal3 * 2.0 + kal4) * sixth;
        let q0_i = scratch.q0[i];
        let qfinal = [
            q0_i[0] + (kq1[0] + 2.0 * kq2[0] + 2.0 * kq3[0] + kq4[0]) * sixth,
            q0_i[1] + (kq1[1] + 2.0 * kq2[1] + 2.0 * kq3[1] + kq4[1]) * sixth,
            q0_i[2] + (kq1[2] + 2.0 * kq2[2] + 2.0 * kq3[2] + kq4[2]) * sixth,
            q0_i[3] + (kq1[3] + 2.0 * kq2[3] + 2.0 * kq3[3] + kq4[3]) * sixth,
        ];
        body.rot.quaternion = JeodQuat::new(qfinal[0], qfinal[1], qfinal[2], qfinal[3]);
        // JEOD_INV: DB.09 — quaternion normalized after every integration step
        astrodyn_dynamics::normalize_integ(&mut body.rot.quaternion);
    }
}

/// Populate one intermediate RK4 stage state from a base state and
/// derivative step of size `h`, reusing caller-owned buffers.
#[allow(clippy::too_many_arguments)]
fn fill_stage_state(
    n: usize,
    pos0: &[DVec3],
    vel0: &[DVec3],
    q0: &[[f64; 4]],
    omega0: &[DVec3],
    k_v: &[DVec3],
    k_a: &[DVec3],
    k_qdot: &[[f64; 4]],
    k_alpha: &[DVec3],
    h: f64,
    stage_pos: &mut [DVec3],
    stage_vel: &mut [DVec3],
    stage_q: &mut [[f64; 4]],
    stage_omega: &mut [DVec3],
) {
    for i in 0..n {
        stage_pos[i] = pos0[i] + k_v[i] * h;
        stage_vel[i] = vel0[i] + k_a[i] * h;
        stage_q[i] = step_q_arr(q0[i], k_qdot[i], h);
        stage_omega[i] = omega0[i] + k_alpha[i] * h;
    }
}

/// Evaluate per-body RK4 derivatives at a given stage state.
///
/// Writes outputs into the caller-owned `k_*` buffers instead of
/// allocating. `stage_trans_buf` and `stage_rot_buf` are reused for
/// assembling the input slices passed to `contact_eval`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn eval_stage(
    stage_pos: &[DVec3],
    stage_vel: &[DVec3],
    stage_q: &[[f64; 4]],
    stage_omega: &[DVec3],
    time_frac: f64,
    gravity_fn: &mut dyn FnMut(usize, DVec3, DVec3, f64) -> DVec3,
    contact_eval: &mut dyn FnMut(&[TranslationalState], &[RotationalState], &mut [(DVec3, DVec3)]),
    bodies: &[CoupledBodyInput<'_>],
    stage_trans_buf: &mut [TranslationalState],
    stage_rot_buf: &mut [RotationalState],
    contact_out: &mut [(DVec3, DVec3)],
    k_v: &mut [DVec3],
    k_a: &mut [DVec3],
    k_qdot: &mut [[f64; 4]],
    k_alpha: &mut [DVec3],
) {
    let n = bodies.len();
    for i in 0..n {
        stage_trans_buf[i] = TranslationalState {
            position: stage_pos[i],
            velocity: stage_vel[i],
        };
        // Build a NORMALIZED quaternion for `contact_eval` only: JEOD's
        // `left_quat_to_transformation` (and our port) assumes a
        // normalized quaternion (see `JEOD_INV: RF.09`), so feeding it
        // the raw RK4 intermediate would yield a slightly non-orthonormal
        // rotation matrix in the contact closure. Using the
        // integration-safe `normalize_integ` preserves the scalar sign.
        //
        // The `qdot = 0.5 · ω ⊗ q` derivative used below is computed from
        // the *raw* stage quaternion, matching `rk4_sixdof_step` and
        // `integrate_body_coupled`, which also do not renormalize at
        // intermediate stages — renormalization only happens once at
        // step end. The normalization here is strictly a boundary
        // correction for the contact callback.
        let mut normalized_quat =
            JeodQuat::new(stage_q[i][0], stage_q[i][1], stage_q[i][2], stage_q[i][3]);
        astrodyn_dynamics::normalize_integ(&mut normalized_quat);
        stage_rot_buf[i] = RotationalState {
            quaternion: normalized_quat,
            ang_vel_body: stage_omega[i],
        };
    }

    // Reset per-stage contact accumulators before handing off to the
    // callback. The scratch buffer is reused across stages and steps;
    // if a future `contact_eval` implementation only writes bodies that
    // are in contact, stale entries from a prior stage would otherwise
    // be silently applied.
    for entry in contact_out.iter_mut() {
        *entry = (DVec3::ZERO, DVec3::ZERO);
    }

    contact_eval(stage_trans_buf, stage_rot_buf, contact_out);

    for (i, body) in bodies.iter().enumerate() {
        let (contact_force, contact_torque_body) = contact_out[i];
        let grav_accel = gravity_fn(i, stage_pos[i], stage_vel[i], time_frac);
        let total_force = body.non_grav_non_contact_force + contact_force;
        let accel = grav_accel
            + if total_force == DVec3::ZERO {
                DVec3::ZERO
            } else {
                astrodyn_dynamics::compute_translational_acceleration(
                    total_force,
                    body.mass.inverse_mass,
                )
            };
        let total_torque = body.non_contact_torque_body + contact_torque_body;
        // qdot is computed from the *raw* stage quaternion (not the
        // normalized copy in `stage_rot_buf`) to match the rest of the
        // RK4 integration paths: intermediate stages are not
        // renormalized — only the final combined quaternion is.
        let raw_quat = JeodQuat::new(stage_q[i][0], stage_q[i][1], stage_q[i][2], stage_q[i][3]);
        let qdot = astrodyn_dynamics::compute_left_quat_deriv(&raw_quat, stage_omega[i]);
        let alpha = astrodyn_dynamics::compute_rotational_acceleration(
            &body.mass.inertia,
            &body.mass.inverse_inertia,
            stage_omega[i],
            total_torque,
        );
        k_v[i] = stage_vel[i];
        k_a[i] = accel;
        k_qdot[i] = qdot;
        k_alpha[i] = alpha;
    }
}

#[inline]
fn step_q_arr(q_base: [f64; 4], k_qdot: [f64; 4], h: f64) -> [f64; 4] {
    [
        q_base[0] + k_qdot[0] * h,
        q_base[1] + k_qdot[1] * h,
        q_base[2] + k_qdot[2] * h,
        q_base[3] + k_qdot[3] * h,
    ]
}

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
    gj_state: Option<&mut astrodyn_dynamics::GaussJacksonState>,
    abm4_state: Option<&mut astrodyn_dynamics::Abm4State>,
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
        astrodyn_dynamics::compute_translational_acceleration(non_grav_force, m.inverse_mass)
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
                IntegratorType::Rk4 => astrodyn_dynamics::rk4_sixdof_step(
                    &six_state,
                    accel,
                    torque_fn,
                    mass_props,
                    integ_dyndt,
                ),
                IntegratorType::Rkf45 => astrodyn_dynamics::rkf45_sixdof_step(
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
            *trans = astrodyn_dynamics::rk4_translational_step(trans, accel, integ_dyndt);
        }
        IntegratorType::Rkf45 => {
            *trans = astrodyn_dynamics::rkf45_translational_step(trans, accel, integ_dyndt);
        }
        IntegratorType::Abm4 => {
            let abm = abm4_state.expect(
                "ABM4 integrator requires a persistent Abm4State passed in via abm4_state. \
                 Runner: call Simulation::validate(); Bevy: add Abm4StateC.",
            );
            *trans = astrodyn_dynamics::abm4_translational_step(trans, accel, integ_dyndt, abm);
        }
        IntegratorType::GaussJackson(cfg) => {
            let gj = gj_state.expect(
                "GaussJackson integrator requires gj_state. \
                 Set SimBody::gj_state or call Simulation::validate() first.",
            );
            // Convert the wrapper config once for the equality check; the
            // raw kernel below operates on the underlying astrodyn_dynamics
            // config that gj.config() returns.
            let raw_cfg: astrodyn_dynamics::GaussJacksonConfig = cfg.into();
            debug_assert_eq!(
                gj.config(),
                &raw_cfg,
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
            let unconverged_before = gj.bootstrap_unconverged_iterations();
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
            let unconverged_after = gj.bootstrap_unconverged_iterations();
            if unconverged_after > unconverged_before && unconverged_before == 0 {
                log::warn!(
                    "GaussJackson bootstrap edit accepted a non-converged correction \
                     ({unconverged_after} iteration(s) total — JEOD-faithful behavior, \
                     but long missions where bootstrap error compounds may want to \
                     review the integration setup)."
                );
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
///   The `f64` argument is `time_frac` — the fraction of `dt` elapsed at
///   this stage (0.0 for stage 1, 0.5 for stages 2 and 3, 1.0 for stage 4),
///   matching `integrate_body`'s gravity-closure convention so callers can
///   reuse the same ephemeris-interpolation logic.
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
pub fn integrate_body_coupled<V: astrodyn_quantities::frame::Vehicle>(
    config: &DynamicsConfig,
    trans: &mut TranslationalState,
    rot: Option<&mut RotationalState>,
    mass: Option<&MassProperties>,
    mut stage_fn: impl FnMut(
        &TranslationalState,
        Option<&RotationalState>,
        &FlatPlateState<V>,
        f64,
    ) -> CoupledStageEval,
    thermal: &mut FlatPlateState<V>,
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
    // JEOD_INV: IN.32 — snapshot thermal state at step start before stage 1.
    thermal.snapshot();

    // Stage 1: evaluate at current state (time_frac = 0.0)
    let eval1 = stage_fn(trans, None, thermal, 0.0);
    debug_assert_eq!(
        eval1.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 1"
    );
    let k1_accel = compute_total_accel(&eval1, mass);
    let k1_v = vel0;
    // Move `temp_dots` out of `eval1`; we've already read everything we
    // need from `eval1` via `compute_total_accel` above, and the Vec
    // allocation is large enough to matter when the plate count is high.
    let k1_tdots = eval1.temp_dots;

    // Stage 2: evaluate at t + dt/2 using k1
    let half_dt = integ_dyndt * 0.5;
    let s2_trans = TranslationalState {
        position: pos0 + k1_v * half_dt,
        velocity: vel0 + k1_accel * half_dt,
    };
    thermal.advance_intermediate(&k1_tdots, half_dt);
    let eval2 = stage_fn(&s2_trans, None, thermal, 0.5);
    debug_assert_eq!(
        eval2.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 2"
    );
    let k2_accel = compute_total_accel(&eval2, mass);
    let k2_v = s2_trans.velocity;
    let k2_tdots = eval2.temp_dots;

    // Stage 3: evaluate at t + dt/2 using k2
    let s3_trans = TranslationalState {
        position: pos0 + k2_v * half_dt,
        velocity: vel0 + k2_accel * half_dt,
    };
    thermal.advance_intermediate(&k2_tdots, half_dt);
    let eval3 = stage_fn(&s3_trans, None, thermal, 0.5);
    debug_assert_eq!(
        eval3.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 3"
    );
    let k3_accel = compute_total_accel(&eval3, mass);
    let k3_v = s3_trans.velocity;
    let k3_tdots = eval3.temp_dots;

    // Stage 4: evaluate at t + dt using k3
    let s4_trans = TranslationalState {
        position: pos0 + k3_v * integ_dyndt,
        velocity: vel0 + k3_accel * integ_dyndt,
    };
    thermal.advance_intermediate(&k3_tdots, integ_dyndt);
    let eval4 = stage_fn(&s4_trans, None, thermal, 1.0);
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

    // Combine thermal state with overshoot clamping.
    thermal.finalize_rk4(
        &k1_tdots,
        &k2_tdots,
        &k3_tdots,
        &eval4.temp_dots,
        integ_dyndt,
    );
}

/// 6-DOF coupled RK4: translational + rotational + thermal.
///
/// # Quaternion drift across stages
///
/// As in [`astrodyn_dynamics::integration::rk4_sixdof_step`], the four RK4 stages
/// stack `qdot` increments on the un-normalized stage quaternions and run a
/// single `normalize_integ` after the final combination. For the dt regime our
/// Tier 3 suite exercises (60 s LEO, 300 s GEO, 100–200 s translunar, with
/// `|ω| ≲ 1 rad/s`), per-stage `|q| − 1` drift stays below `1e-4` and post-step
/// normalization absorbs it while preserving JEOD parity. Callers integrating
/// fast tumblers or with much larger dt should renormalize between stages 2
/// and 3.
#[allow(clippy::too_many_arguments)]
fn integrate_coupled_sixdof<V: astrodyn_quantities::frame::Vehicle>(
    trans: &mut TranslationalState,
    rot: &mut RotationalState,
    mass_props: &MassProperties,
    stage_fn: &mut impl FnMut(
        &TranslationalState,
        Option<&RotationalState>,
        &FlatPlateState<V>,
        f64,
    ) -> CoupledStageEval,
    thermal: &mut FlatPlateState<V>,
    integ_dyndt: f64,
    n_plates: usize,
) {
    let pos0 = trans.position;
    let vel0 = trans.velocity;
    let q0 = rot.quaternion.data;
    let omega0 = rot.ang_vel_body;
    // JEOD_INV: IN.32 — snapshot thermal state at step start before stage 1.
    thermal.snapshot();

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
    let eval_rot_derivs =
        |eval: &CoupledStageEval, rot_s: &RotationalState| -> (DVec3, [f64; 4], DVec3) {
            let accel = compute_total_accel(eval, Some(mass_props));
            let k_qdot =
                astrodyn_dynamics::compute_left_quat_deriv(&rot_s.quaternion, rot_s.ang_vel_body);
            let k_alpha = astrodyn_dynamics::compute_rotational_acceleration(
                &mass_props.inertia,
                &mass_props.inverse_inertia,
                rot_s.ang_vel_body,
                eval.torque,
            );
            (accel, k_qdot, k_alpha)
        };

    // Stage 1 (time_frac = 0.0)
    let eval1 = stage_fn(trans, Some(rot), thermal, 0.0);
    debug_assert_eq!(
        eval1.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 1"
    );
    let (k1_accel, k1_qdot, k1_alpha) = eval_rot_derivs(&eval1, rot);
    let k1_v = vel0;
    // Move temp_dots out of eval1 rather than cloning — `eval_rot_derivs`
    // has already consumed everything we need from the borrow above.
    let k1_tdots = eval1.temp_dots;

    // Stage 2
    let half_dt = integ_dyndt * 0.5;
    let s2_trans = TranslationalState {
        position: pos0 + k1_v * half_dt,
        velocity: vel0 + k1_accel * half_dt,
    };
    let s2_rot = make_rot(step_q(q0, k1_qdot, half_dt), omega0 + k1_alpha * half_dt);
    thermal.advance_intermediate(&k1_tdots, half_dt);
    let eval2 = stage_fn(&s2_trans, Some(&s2_rot), thermal, 0.5);
    debug_assert_eq!(
        eval2.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 2"
    );
    let (k2_accel, k2_qdot, k2_alpha) = eval_rot_derivs(&eval2, &s2_rot);
    let k2_v = s2_trans.velocity;
    let k2_tdots = eval2.temp_dots;

    // Stage 3
    let s3_trans = TranslationalState {
        position: pos0 + k2_v * half_dt,
        velocity: vel0 + k2_accel * half_dt,
    };
    let s3_rot = make_rot(step_q(q0, k2_qdot, half_dt), omega0 + k2_alpha * half_dt);
    thermal.advance_intermediate(&k2_tdots, half_dt);
    let eval3 = stage_fn(&s3_trans, Some(&s3_rot), thermal, 0.5);
    debug_assert_eq!(
        eval3.temp_dots.len(),
        n_plates,
        "stage_fn returned wrong temp_dots length at stage 3"
    );
    let (k3_accel, k3_qdot, k3_alpha) = eval_rot_derivs(&eval3, &s3_rot);
    let k3_v = s3_trans.velocity;
    let k3_tdots = eval3.temp_dots;

    // Stage 4
    let s4_trans = TranslationalState {
        position: pos0 + k3_v * integ_dyndt,
        velocity: vel0 + k3_accel * integ_dyndt,
    };
    let s4_rot = make_rot(
        step_q(q0, k3_qdot, integ_dyndt),
        omega0 + k3_alpha * integ_dyndt,
    );
    thermal.advance_intermediate(&k3_tdots, integ_dyndt);
    let eval4 = stage_fn(&s4_trans, Some(&s4_rot), thermal, 1.0);
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
    astrodyn_dynamics::normalize_integ(&mut rot.quaternion);

    // Combine thermal state with overshoot clamping.
    thermal.finalize_rk4(
        &k1_tdots,
        &k2_tdots,
        &k3_tdots,
        &eval4.temp_dots,
        integ_dyndt,
    );
}

/// Typed sibling of [`integrate_body`].
///
/// Identical kernel — wraps the entry boundary so callers pass typed
/// quantities. The `trans` parameter takes a mutable
/// [`TranslationalStateTyped<F>`] so the integration-frame constraint
/// is enforced at compile time; the wrapper unwraps to the raw kernel
/// storage on entry and writes the integrated state back at exit. The
/// `gravity_fn` closure is also typed: it receives an intermediate
/// position / velocity in `F` and returns an `Acceleration<F>`. No new
/// arithmetic — only `.raw_si()` / `from_raw_si` at the edges.
///
/// `dt` becomes [`uom::si::f64::Time`]. The dimensionless
/// `time_scale_factor` (JEOD's `TimeDyn::scale_factor`) stays an
/// `f64` ratio per Phase 5's design note (#107).
///
/// Generic over the integration frame `F` and the vehicle phantom `V`
/// — production callers integrate in `RootInertial` (the gateway
/// stage); the runner integrates in `IntegrationFrame` (per-body
/// integration frame). The torque parameter ties to
/// [`BodyFrame<V>`] regardless of `F`.
#[allow(clippy::too_many_arguments)]
pub fn integrate_body_typed<V: Vehicle, F: Frame>(
    config: &DynamicsConfig,
    trans: &mut TranslationalStateTyped<F>,
    rot: Option<&mut RotationalState>,
    mass: Option<&MassProperties>,
    gravity_fn: impl Fn(Position<F>, Velocity<F>, f64) -> Acceleration<F>,
    non_grav_force: Force<F>,
    torque: Torque<BodyFrame<V>>,
    dt: Time,
    time_scale_factor: f64,
    integrator: IntegratorType,
    gj_state: Option<&mut astrodyn_dynamics::GaussJacksonState>,
    abm4_state: Option<&mut astrodyn_dynamics::Abm4State>,
) {
    use uom::si::time::second;
    // allowed: typed-sibling boundary at the gateway-owned `IntegratorType`
    // wrapper. The integrator orchestration (`integrate_body`,
    // `integrate_body_coupled`, multi-stage RK4/Gauss-Jackson/ABM4) is
    // gateway-resident because it composes gateway-only types
    // (`crate::integrator::IntegratorType` wrapper, `FlatPlateState`,
    // `IntegrableObject`) and surfaces Bevy/runner-aware diagnostics. The
    // typed sibling unwraps `Position`/`Velocity` for the caller-supplied
    // `gravity_fn` closure on each integrator stage; lifting the raw
    // intermediate position is unavoidable here. See issue #388 — the
    // kernel itself is the documented orchestration boundary, not a
    // candidate for relocation into `astrodyn_dynamics`.
    let raw_gravity_fn = |pos: DVec3, vel: DVec3, time_frac: f64| -> DVec3 {
        gravity_fn(
            Position::<F>::from_raw_si(pos), // allowed: integrator-stage boundary, see note above
            Velocity::<F>::from_raw_si(vel), // allowed: integrator-stage boundary, see note above
            time_frac,
        )
        .raw_si()
    };
    let mut raw_trans = trans.to_untyped();
    integrate_body(
        config,
        &mut raw_trans,
        rot,
        mass,
        raw_gravity_fn,
        non_grav_force.raw_si(),
        torque.raw_si(),
        dt.get::<second>(),
        time_scale_factor,
        integrator,
        gj_state,
        abm4_state,
    );
    // allowed: typed-sibling boundary writing the raw integrator output
    // back into the typed `trans`. See note above.
    *trans = TranslationalStateTyped::<F>::from_untyped_unchecked(&raw_trans);
}

/// Compute total translational acceleration from a stage evaluation.
fn compute_total_accel(eval: &CoupledStageEval, mass: Option<&MassProperties>) -> DVec3 {
    let non_grav_accel = if eval.non_grav_force == DVec3::ZERO {
        DVec3::ZERO
    } else if let Some(m) = mass {
        // JEOD_INV: DB.18 — force to acceleration via inverse mass
        astrodyn_dynamics::compute_translational_acceleration(eval.non_grav_force, m.inverse_mass)
    } else {
        panic!(
            "Non-zero force ({:?}) but no MassProperties. \
             Provide MassProperties for any body with interaction forces.",
            eval.non_grav_force
        );
    };
    eval.gravity_accel + non_grav_accel
}

/// Reset multi-step integrator history on a topology change.
///
/// Port of JEOD's `dyn_body_attach.cc::reset_integrators()` (lines 860, 871)
/// for the Bevy / runner pipeline. Multi-step integrators (Gauss-Jackson,
/// ABM4) accumulate predictor / corrector history from prior steps; when a
/// body's mass / attachment topology changes mid-flight (an attach or
/// detach event), that history is no longer valid because the dynamics it
/// predicted are now different. JEOD addresses this with an explicit
/// `reset_integrators()` call inside `DynBody::attach_child` /
/// `DynBody::detach`. This helper is the one-line equivalent for our
/// adapters.
///
/// Single-step integrators (RK4, RKF4(5)) carry no per-step history and
/// are unaffected — pass `None` for the corresponding state. The function
/// is a no-op for whichever state argument is `None`, so adapters that
/// don't know which integrator a body uses can pass both unconditionally.
///
/// # Where to call this
///
/// - **`astrodyn_bevy::staging_system`**: after each `AttachEvent` /
///   `DetachEvent` is applied to `MassTreeR`.
/// - **`astrodyn_runner::Simulation::attach` / `detach` /
///   `attach_subtree_aligned` / `detach_subtree`**: after the topology
///   mutation completes.
/// - **`astrodyn_runner::Simulation::sync_body_mass_from_tree`**: the
///   supported lower-level path for callers that mutate `mass_tree`
///   directly (via `tree.attach` / `tree.detach`) and then must
///   propagate the new composite mass back into `SimBody`. Resetting
///   the integrator is part of the same sync — see the method's
///   rustdoc for the call sequence.
///
/// # JEOD invariant
///
/// See `JEOD_invariants.md` row IG.37. Failing to call this through a
/// topology change leaves the integrator's `topology_dirty` flag set; the
/// next call to `GaussJacksonState::integrate` /
/// `abm4_translational_step` will panic loudly with the corrective hint.
// JEOD_INV: IG.37 — multi-step integrator history must be reset on topology change
pub fn reset_integrators(
    gj_state: Option<&mut astrodyn_dynamics::GaussJacksonState>,
    abm4_state: Option<&mut astrodyn_dynamics::Abm4State>,
) {
    if let Some(gj) = gj_state {
        gj.reset_for_topology_change();
    }
    if let Some(abm) = abm4_state {
        abm.reset_for_topology_change();
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
        let mass = astrodyn_dynamics::MassProperties::new(500.0);
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
            IntegratorType::Rk4,
            None,
            None,
        );

        // Coupled path with constant forces (no thermal state)
        let mut trans2 = TranslationalState {
            position: pos0,
            velocity: vel0,
        };
        // `<SelfRef>` is the canonical runtime-resolved instantiation
        // — the integration kernel is `<V>`-generic, but every adapter
        // (Bevy + standalone runner) lands at `<SelfRef>`, and this
        // unit test mirrors that boundary.
        let mut thermal = FlatPlateState::<astrodyn_quantities::frame::SelfRef> {
            plates: vec![],
            temperatures: vec![],
            t_pow4_cached: vec![],
            ..Default::default()
        };
        integrate_body_coupled(
            &config,
            &mut trans2,
            None,
            Some(&mass),
            |inter_trans, _inter_rot, _inter_thermal, _time_frac| CoupledStageEval {
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
