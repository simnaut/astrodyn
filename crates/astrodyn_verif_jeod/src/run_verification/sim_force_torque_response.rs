// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! `VerificationCase` constructors for the SIM_force_torque analytical
//! family (`tier3_sim_force_torque_response`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "verif step counts bounded by Tier 3 propagation span (<< usize / f64 mantissa)"
)]
//!
//! JEOD's `models/dynamics/dyn_body/verif/SIM_force_torque/` is an
//! empty-space test rig that verifies force and torque accumulation,
//! non-transmitted forces, and the equations of motion. The recipes
//! here cover the closed-form-checkable cases its `RUN_test/input.py`
//! schedule exercises: F = m·a translation, τ = I·α rotation, the
//! decoupling of CoM force from rotation, and the symmetric ±F impulse
//! that returns a 3-DOF body to rest with a residual displacement.
//!
//! These cases have no JEOD reference CSV. Each recipe pairs with
//! [`CsvReference::SyntheticTimes`] so the parity trait can assert
//! `runner ↔ bevy` bit-identity at every synthetic record while the
//! matching tier3 file asserts the closed-form analytical identity on
//! the runner. The empty-space frame is built the same way the
//! pre-recipe tier3 file did it: a `central: true` source with `mu=0`
//! supplies a root frame the body never references (the body's
//! `GravityControls` is empty), so the body responds only to its
//! external force/torque inputs.
//!
//! The constant-force / constant-torque recipes set
//! `VehicleConfig.external_force` / `external_torque` at scenario
//! build time — the runner copies the value onto `SimBody`, the Bevy
//! adapter inserts `ExternalForceC` / `ExternalTorqueC` on the body
//! entity, and the load persists across every integration step
//! without a `pre_step` hook. The symmetric-impulse recipe additionally
//! installs a `pre_step` closure that flips the inertial-frame force
//! sign at the midpoint record so the second half of the propagation
//! decelerates the body back to rest.

use crate::verification::{
    CsvReference, InitialConditions, PreStepCadence, PreStepClosure, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, Force, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, JeodQuat, MassProperties, RootInertial, RotationModel, RotationalState,
    SimulationBuilder, SimulationTime, Torque, TranslationalState, VehicleConfig,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

/// Empty-space test mass shared by every recipe (kilograms). Matches the
/// value the pre-recipe tier3 file used so the analytical assertions
/// drive identical numerics under the recipe path.
const MASS_KG: f64 = 100.0;

/// Force-test integrator step (seconds). The pre-recipe tier3 file
/// drove `tier3_force_constant_acceleration` and the symmetric-impulse
/// test at `dt = 0.1` (RK4 is exact on the linear ODE so the step does
/// not affect the analytical bound) — preserve it so the SyntheticTimes
/// cadence drives identical integration ticks across runner and bevy.
const DT_FORCE_S: f64 = 0.1;

/// Torque-test integrator step (seconds). The pre-recipe tier3 file
/// used `dt = 0.01` for the τ = I·α tests; tightening over the force
/// step keeps the angular-velocity error well below the 1e-10
/// tolerance.
const DT_TORQUE_S: f64 = 0.01;

/// Total propagation horizon for the constant-force / impulse cases
/// (seconds). Pre-recipe `tier3_force_constant_acceleration` and
/// `tier3_force_symmetric_impulse_returns_to_rest` both ran 10 s.
const T_TOTAL_FORCE_S: f64 = 10.0;

/// Total propagation horizon for the constant-torque case (seconds).
const T_TOTAL_TORQUE_S: f64 = 10.0;

/// Total propagation horizon for the decoupling cases (seconds). The
/// pre-recipe test ran each of the three sub-sims for 5 s.
const T_TOTAL_DECOUPLE_S: f64 = 5.0;

/// Symmetric-impulse pivot point: the inertial-frame external force
/// reverses sign exactly at `T_TOTAL_FORCE_S / 2`. Same value the
/// pre-recipe `tier3_force_symmetric_impulse_returns_to_rest` used.
const HALF_DURATION_S: f64 = 0.5 * T_TOTAL_FORCE_S;

/// Inertia tensor for the torque-only / decoupling cases when the
/// principal axes are equal (kg·m²). Matches the pre-recipe
/// `tier3_force_and_torque_decoupled` body where `I_x = I_y = I_z = 10`.
fn inertia_uniform() -> DMat3 {
    DMat3::from_diagonal(DVec3::splat(10.0))
}

/// Inertia tensor for the constant-torque case. Matches
/// `tier3_torque_constant_angular_acceleration` where the body has
/// `diag(10, 20, 20)` so the closed-form rate is
/// `omega_x = τ_x · t / I_x = 1 rad/s` after 10 s with `τ_x = 1 N·m`.
fn inertia_torque() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(10.0, 0.0, 0.0),
        DVec3::new(0.0, 20.0, 0.0),
        DVec3::new(0.0, 0.0, 20.0),
    )
}

/// Closed-form synthetic-time cadence: number of `dt`-sized steps that
/// span `t_total` seconds. The pre-recipe tier3 file used integer
/// division (e.g. `(10.0 / 0.1) as usize`) — match that exactly so the
/// recipe's record count is bit-identical to the loop count the
/// pre-recipe `step_n` calls drove.
fn num_steps(dt: f64, t_total: f64) -> usize {
    (t_total / dt) as usize
}

/// Recipes opt out of every runner-vs-JEOD tolerance group: the
/// tier3 file asserts closed-form analytical identities directly on
/// the runner-side result (a = F/m, α = τ/I, decoupling, return-to-
/// rest), and the parity trait asserts `runner ↔ bevy` bit-identity
/// at every synthetic record without consulting these tolerances.
fn analytical_tolerances() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

/// Synthetic gravity-source marker for the empty-space root: not one of
/// the six sealed planets, so (per issue #662's strict identity rule) it
/// requires a `define_planet!`-minted marker and `add_source_typed`.
mod tags {
    astrodyn::define_planet!(CentralPoint);
}

/// A negligible gravity source kept in the frame tree but not
/// referenced by any body. `Simulation` requires a root frame; using
/// `central: true` with `mu = 0.0` gives us the frame without any
/// gravitational effect. Mirrors the pre-recipe tier3 file's
/// `add_dummy_central_source` helper exactly.
fn add_dummy_central_source(sb: &mut SimulationBuilder) {
    sb.add_source_typed::<tags::CentralPoint>(
        "central_point",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
}

/// Shared 3-DOF scenario constructor for the constant-force and
/// symmetric-impulse recipes. Builds an empty-space simulation with a
/// single point-mass body at rest, with the given inertial-frame
/// external force pre-set on `VehicleConfig.external_force` so the
/// runner-side `SimBody.external_force` and the Bevy-side
/// `ExternalForceC` start at identical values without a `pre_step`
/// hook.
fn build_free_body_3dof(dt: f64, external_force: DVec3) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    add_dummy_central_source(&mut sb);
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }),
        rot: None,
        mass: Some(super::typed_helpers::mass_typed(&MassProperties::new(
            MASS_KG,
        ))),
        gravity_controls: GravityControls { controls: vec![] },
        external_force: Force::<RootInertial>::from_raw_si(external_force),
        ..VehicleConfig::named("sim-force-torque-response-2")
    });
    sb
}

/// Shared 6-DOF scenario constructor for the constant-torque and
/// decoupling recipes. Builds an empty-space simulation with a single
/// rigid body at rest, with the given inertia tensor and any non-zero
/// external force / body-frame torque pre-set on `VehicleConfig` so
/// the integrator sees them on the first step.
fn build_free_body_6dof(
    dt: f64,
    inertia: DMat3,
    external_force: DVec3,
    external_torque: DVec3,
) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);
    add_dummy_central_source(&mut sb);
    let mass_props = MassProperties::with_inertia(MASS_KG, inertia, DVec3::ZERO);
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }),
        rot: Some(super::typed_helpers::rot_typed(&RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        })),
        mass: Some(super::typed_helpers::mass_typed(&mass_props)),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        external_force: Force::<RootInertial>::from_raw_si(external_force),
        external_torque: Torque::<astrodyn::BodyFrame<astrodyn::SelfRef>>::from_raw_si(
            external_torque,
        ),
        ..VehicleConfig::named("sim-force-torque-response-1")
    });
    sb
}

// ── Test 1: F = m·a on a free body (constant force) ────────────────

/// Inertial-frame force applied throughout the constant-acceleration
/// recipe. Pre-recipe `tier3_force_constant_acceleration` used this
/// arbitrary off-axis vector to exercise all three components.
fn force_constant_acceleration_force() -> DVec3 {
    DVec3::new(3.0, -2.0, 1.0)
}

fn build_force_constant_acceleration(_init: &InitialConditions) -> SimulationBuilder {
    build_free_body_3dof(DT_FORCE_S, force_constant_acceleration_force())
}

/// 3-DOF point-mass at rest with a constant inertial-frame force.
/// After 10 s the analytical sibling asserts `v = F·t/m` and
/// `x = 0.5·F·t²/m` to RK4-roundoff (`< 1e-12` relative).
pub fn force_constant_acceleration() -> VerificationCase {
    VerificationCase {
        name: "tier3_force_constant_acceleration",
        scenario: build_force_constant_acceleration,
        reference: CsvReference::SyntheticTimes {
            dt: DT_FORCE_S,
            num_steps: num_steps(DT_FORCE_S, T_TOTAL_FORCE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Test 2: τ = I·α on a free body (constant torque) ───────────────

/// Body-frame torque applied throughout the constant-α recipe. Pre-
/// recipe `tier3_torque_constant_angular_acceleration` used a pure
/// x-axis torque so the perpendicular axes stay at zero and the
/// closed-form `omega_x = τ_x · t / I_x` check is the only nonzero
/// component.
fn torque_constant_torque() -> DVec3 {
    DVec3::new(1.0, 0.0, 0.0)
}

fn build_torque_constant_angular_acceleration(_init: &InitialConditions) -> SimulationBuilder {
    build_free_body_6dof(
        DT_TORQUE_S,
        inertia_torque(),
        DVec3::ZERO,
        torque_constant_torque(),
    )
}

/// 6-DOF body at rest with a constant body-frame torque about its
/// `x` axis (the larger principal axis is along `y` / `z`, so the
/// inertial coupling is zero by construction). After 10 s the
/// analytical sibling asserts `omega_x = τ_x · t / I_x = 1 rad/s` to
/// RK4-roundoff and `omega_y`, `omega_z` stay below 1e-12 rad/s.
pub fn torque_constant_angular_acceleration() -> VerificationCase {
    VerificationCase {
        name: "tier3_torque_constant_angular_acceleration",
        scenario: build_torque_constant_angular_acceleration,
        reference: CsvReference::SyntheticTimes {
            dt: DT_TORQUE_S,
            num_steps: num_steps(DT_TORQUE_S, T_TOTAL_TORQUE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Test 3: force-at-CoM decouples translation and rotation ────────
//
// The pre-recipe tier3 file ran three sub-sims (force-only, torque-
// only, both) and asserted decoupling between them. Each sub-sim
// gets its own recipe factory here so the parity trait can drive
// each through the bridge independently — the runner-vs-runner
// algebraic comparisons in the tier3 file then read each recipe's
// final body state to verify the decoupling identity.

/// Force-only inertial-frame force for the decoupling family. Pre-
/// recipe used a pure +x vector; preserving it so the closed-form
/// `v_x = F·t/m` literal matches.
fn decouple_force() -> DVec3 {
    DVec3::new(2.0, 0.0, 0.0)
}

/// Torque-only body-frame torque for the decoupling family. Pre-
/// recipe used a pure +z vector.
fn decouple_torque() -> DVec3 {
    DVec3::new(0.0, 0.0, 1.0)
}

fn build_force_and_torque_decoupled_force(_init: &InitialConditions) -> SimulationBuilder {
    build_free_body_6dof(
        DT_TORQUE_S,
        inertia_uniform(),
        decouple_force(),
        DVec3::ZERO,
    )
}

/// Sub-sim A: force-only 6-DOF body. Force at the CoM (no offset on
/// the mass properties) should produce pure translation with zero
/// induced angular velocity.
pub fn force_and_torque_decoupled_force() -> VerificationCase {
    VerificationCase {
        name: "tier3_force_and_torque_decoupled_force",
        scenario: build_force_and_torque_decoupled_force,
        reference: CsvReference::SyntheticTimes {
            dt: DT_TORQUE_S,
            num_steps: num_steps(DT_TORQUE_S, T_TOTAL_DECOUPLE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

fn build_force_and_torque_decoupled_torque(_init: &InitialConditions) -> SimulationBuilder {
    build_free_body_6dof(
        DT_TORQUE_S,
        inertia_uniform(),
        DVec3::ZERO,
        decouple_torque(),
    )
}

/// Sub-sim B: torque-only 6-DOF body. A pure body-frame torque should
/// produce pure rotation with zero induced translation.
pub fn force_and_torque_decoupled_torque() -> VerificationCase {
    VerificationCase {
        name: "tier3_force_and_torque_decoupled_torque",
        scenario: build_force_and_torque_decoupled_torque,
        reference: CsvReference::SyntheticTimes {
            dt: DT_TORQUE_S,
            num_steps: num_steps(DT_TORQUE_S, T_TOTAL_DECOUPLE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

fn build_force_and_torque_decoupled_both(_init: &InitialConditions) -> SimulationBuilder {
    build_free_body_6dof(
        DT_TORQUE_S,
        inertia_uniform(),
        decouple_force(),
        decouple_torque(),
    )
}

/// Sub-sim C: both force and torque applied. The tier3 sibling asserts
/// this run's translational state matches sub-sim A's and its
/// rotational state matches sub-sim B's — equivalence of independent
/// applications and joint application.
pub fn force_and_torque_decoupled_both() -> VerificationCase {
    VerificationCase {
        name: "tier3_force_and_torque_decoupled_both",
        scenario: build_force_and_torque_decoupled_both,
        reference: CsvReference::SyntheticTimes {
            dt: DT_TORQUE_S,
            num_steps: num_steps(DT_TORQUE_S, T_TOTAL_DECOUPLE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: None,
    }
}

// ── Test 4: symmetric ±F impulse pair returns body to rest ──────────

/// Inertial-frame force the symmetric-impulse recipe applies in the
/// first half of the propagation; the `pre_step` closure flips its
/// sign at `t = HALF_DURATION_S`. Pre-recipe
/// `tier3_force_symmetric_impulse_returns_to_rest` used a pure +x
/// vector.
fn impulse_force() -> DVec3 {
    DVec3::new(5.0, 0.0, 0.0)
}

fn build_force_symmetric_impulse(_init: &InitialConditions) -> SimulationBuilder {
    // Initial force is `+impulse_force()` — the `pre_step` factory
    // flips it to `-impulse_force()` at the midpoint record.
    build_free_body_3dof(DT_FORCE_S, impulse_force())
}

/// `pre_step` factory: flips body 0's inertial-frame external force
/// from `+F` to `-F` exactly once, at the record that advances the
/// simulation past the midpoint `t = HALF_DURATION_S`. The
/// constant-+F initial value is wired through `VehicleConfig.external_force`
/// in [`build_force_symmetric_impulse`] above, so the closure only
/// needs to fire the single sign-flip event; the runner side mirrors
/// the pre-recipe tier3 file's two `set_body_external_force` calls
/// bracketing the propagation halves, and the Bevy side updates
/// `ExternalForceC` through the parity trait's `BevySimContext` so
/// both runtimes integrate identical sub-steps.
fn impulse_pre_step(_init: &InitialConditions) -> PreStepClosure {
    Box::new(move |sim, time_s: f64| {
        let half_dt = 0.5 * DT_FORCE_S;
        // Fire on the record advancing the sim from `HALF_DURATION_S`
        // to `HALF_DURATION_S + DT_FORCE_S` — the runner's
        // `set_body_external_force` call lands before
        // `step_until(record.time)`, so the +F half integrates
        // exactly `HALF_DURATION_S / DT_FORCE_S` steps and the -F
        // half integrates the same count, matching the pre-recipe
        // file's bracketing call shape.
        if (time_s - (HALF_DURATION_S + DT_FORCE_S)).abs() < half_dt {
            sim.set_body_external_force(0, -impulse_force());
        }
    })
}

/// 3-DOF body with a symmetric ±F impulse pair: +F for the first
/// 5 s, then -F for the next 5 s. The tier3 sibling asserts the
/// final velocity returns to zero (within roundoff) and the residual
/// displacement matches the closed-form `2 · 0.5 · (F/m) · (T/2)²`.
pub fn force_symmetric_impulse() -> VerificationCase {
    VerificationCase {
        name: "tier3_force_symmetric_impulse_returns_to_rest",
        scenario: build_force_symmetric_impulse,
        reference: CsvReference::SyntheticTimes {
            dt: DT_FORCE_S,
            num_steps: num_steps(DT_FORCE_S, T_TOTAL_FORCE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: Some((impulse_pre_step, PreStepCadence::PerRecord)),
    }
}

// ── Test 5: struct-frame external force / torque (issue #510 part 2) ─
//
// Exercises the new `SimContext::set_body_external_force_struct` /
// `set_body_external_torque_struct` surface across both runtimes.
// The struct-frame interpretation requires three distinct frames at
// runtime (structural ≠ body ≠ inertial), so both `t_struct_body`
// and the initial inertial-body attitude are non-trivial. The pre_step
// fires once at the first record to set the struct-frame load (which
// `VehicleConfig` has no field for), then the load persists through
// the rest of the propagation. Parity is asserted at every record.

/// Structural-frame force the struct-frame parity recipe applies for
/// the entire propagation (N) — set via `pre_step` at record 0 (the
/// only way today, since `VehicleConfig` has no
/// `external_force_struct` field).
fn force_struct_load() -> DVec3 {
    DVec3::new(10.0, -3.0, 5.0)
}

/// Structural-frame torque companion for the torque variant (N·m).
fn torque_struct_load() -> DVec3 {
    DVec3::new(0.5, 0.0, 2.0)
}

/// Non-trivial structural-to-body rotation matrix (30° about z) —
/// distinct from `DMat3::IDENTITY` so the `T_struct_body` factor in
/// the force-collection chain is exercised.
fn t_struct_body_nontrivial() -> DMat3 {
    DMat3::from_rotation_z(std::f64::consts::FRAC_PI_6)
}

/// Non-trivial initial inertial-body attitude (45° about y) — distinct
/// from identity so the `T_inertial_body` factor in
/// `T_inertial_struct = T_struct_body^T * T_inertial_body` is also
/// exercised.
fn initial_attitude_nontrivial() -> JeodQuat {
    JeodQuat::left_quat_from_eigen_rotation(std::f64::consts::FRAC_PI_4, DVec3::new(0.0, 1.0, 0.0))
}

fn build_force_struct_pre_step(_init: &InitialConditions) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, DT_TORQUE_S);
    add_dummy_central_source(&mut sb);
    let mass_props = MassProperties::with_inertia(MASS_KG, inertia_uniform(), DVec3::ZERO);
    sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }),
        rot: Some(super::typed_helpers::rot_typed(&RotationalState {
            quaternion: initial_attitude_nontrivial(),
            ang_vel_body: DVec3::ZERO,
        })),
        mass: Some(super::typed_helpers::mass_typed(&mass_props)),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        t_struct_body: t_struct_body_nontrivial(),
        ..VehicleConfig::named("sim-force-torque-response-0")
    });
    sb
}

fn force_struct_pre_step(_init: &InitialConditions) -> PreStepClosure {
    Box::new(move |sim, time_s: f64| {
        // Fire on the first record only (time_s ≈ DT). The setter is
        // idempotent in principle, but firing once mirrors how a real
        // mission would schedule a one-shot struct-frame load and
        // tests the auto-insert path on the Bevy adapter (component
        // is absent before the first set; afterwards force-collection
        // reads it every step).
        let half_dt = 0.5 * DT_TORQUE_S;
        if (time_s - DT_TORQUE_S).abs() < half_dt {
            sim.set_body_external_force_struct(0, force_struct_load());
        }
    })
}

fn torque_struct_pre_step(_init: &InitialConditions) -> PreStepClosure {
    Box::new(move |sim, time_s: f64| {
        let half_dt = 0.5 * DT_TORQUE_S;
        if (time_s - DT_TORQUE_S).abs() < half_dt {
            sim.set_body_external_torque_struct(0, torque_struct_load());
        }
    })
}

fn both_struct_pre_step(_init: &InitialConditions) -> PreStepClosure {
    Box::new(move |sim, time_s: f64| {
        let half_dt = 0.5 * DT_TORQUE_S;
        if (time_s - DT_TORQUE_S).abs() < half_dt {
            sim.set_body_external_force_struct(0, force_struct_load());
            sim.set_body_external_torque_struct(0, torque_struct_load());
        }
    })
}

/// 6-DOF body with a constant **structural-frame** external force
/// scheduled via `pre_step` at the first record. The runner mirrors
/// `Simulation::set_body_external_force_struct`; the Bevy adapter
/// mirrors `BevySimContext::set_body_external_force_struct` (which
/// writes / auto-inserts `ExternalForceStructC`). Parity asserts the
/// two propagate bit-identical trajectories — the lockstep gate for
/// issue #510 Part 2.
pub fn force_struct_via_pre_step() -> VerificationCase {
    VerificationCase {
        name: "tier3_force_struct_via_pre_step",
        scenario: build_force_struct_pre_step,
        reference: CsvReference::SyntheticTimes {
            dt: DT_TORQUE_S,
            num_steps: num_steps(DT_TORQUE_S, T_TOTAL_DECOUPLE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: Some((force_struct_pre_step, PreStepCadence::PerRecord)),
    }
}

/// Companion to [`force_struct_via_pre_step`]: same scenario, but
/// scheduling a structural-frame torque via `pre_step`. Exercises the
/// `T_struct_body` rotation in the torque branch of the force-collection
/// pipeline (mirrors `simulation/step/integrate.rs:94` →
/// `torque_body = t_struct_body * external_torque_struct`).
pub fn torque_struct_via_pre_step() -> VerificationCase {
    VerificationCase {
        name: "tier3_torque_struct_via_pre_step",
        scenario: build_force_struct_pre_step,
        reference: CsvReference::SyntheticTimes {
            dt: DT_TORQUE_S,
            num_steps: num_steps(DT_TORQUE_S, T_TOTAL_DECOUPLE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: Some((torque_struct_pre_step, PreStepCadence::PerRecord)),
    }
}

/// Both struct-frame force + torque scheduled simultaneously — drives
/// the joint code path through the force-collection branch (both
/// `T_inertial_struct^T * ef_struct` and `t_struct_body * et_struct`
/// active in the same body's per-step pipeline).
pub fn force_and_torque_struct_via_pre_step() -> VerificationCase {
    VerificationCase {
        name: "tier3_force_and_torque_struct_via_pre_step",
        scenario: build_force_struct_pre_step,
        reference: CsvReference::SyntheticTimes {
            dt: DT_TORQUE_S,
            num_steps: num_steps(DT_TORQUE_S, T_TOTAL_DECOUPLE_S),
        },
        duration: Time::new::<second>(0.0),
        tolerances: analytical_tolerances(),
        extras: None,
        pre_step: Some((both_struct_pre_step, PreStepCadence::PerRecord)),
    }
}
