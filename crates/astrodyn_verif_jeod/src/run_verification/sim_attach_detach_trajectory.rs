//! `VerificationCase` constructor for the SIM_verif_attach_detach
//! RUN_simple_attach_detach scenario, attach-only window.
//!
//! Three free-flying 6-DOF bodies (no gravity, no force) are propagated
//! through `Simulation::step()`; at `t = ATTACH_TIME` the recipe fires
//! `attach(veh1 → veh2)` via the [`SimContext::attach`] surface,
//! mirrored by `AttachEvent` on the Bevy side. The runner-vs-JEOD
//! cross-validation lives in `tier3_sim_attach_detach_trajectory.rs`;
//! this recipe drives the runner-vs-Bevy parity surface.
//!
//! ## What is and isn't covered
//!
//! Mirrors the coverage of the hand-rolled
//! `bevy_parity_attach_detach_trajectory_simple` test it replaces:
//! propagates from `t = 0` through the attach window and stops *before*
//! the detach event at `t = 20 s`. The post-detach window is excluded
//! because of issue #308 (`composite_mass_system` reverts
//! `MassPropertiesC` for detached entities before the detach handler
//! reads it). Once #308 lands the recipe can extend through to
//! `t = 30 s` by adding a `detach` call to the `pre_step` closure;
//! until then `KNOWN_PARITY_GAPS` continues to track the post-detach
//! gap separately.
//!
//! ## CSV reference
//!
//! Routes through [`CsvReference::TimesOnly`] reading the existing
//! `kinematic_propagation_simple_kinematic_propagation_state.csv` for
//! cadence (records every 0.1 s). Initial conditions are hardcoded
//! from JEOD `Modified_data/veh{1,2,3}.py`; the parity trait reads
//! only `record.time` from the CSV.

use crate::verification::{
    CsvReference, InitialConditions, PreStepClosure, SimContext, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, GravityControls, IntegratorType, JeodQuat, MassProperties,
    RotationalState, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

/// Integrator timestep matching JEOD's RUN_simple_attach_detach
/// (`SET_test/RUN_simple_attach_detach/input.py`).
const DT: f64 = 0.1;

/// `BodyAttachAligned veh1.attach_to_2` time
/// (`SET_test/RUN_simple_attach_detach/input.py:24`).
const ATTACH_TIME: f64 = 10.0;

/// `BodyDetach veh1.detach_from_2` time. The hand-rolled parity test
/// stops strictly before this (see "What is and isn't covered" above);
/// the recipe currently mirrors that scope so detach is excluded from
/// `pre_step`. The constant is kept here so the duration cap stays
/// expressed in source-file terms.
const DETACH_TIME: f64 = 20.0;

// ── Initial conditions, all from JEOD Modified_data files. ──

fn veh1_mass() -> MassProperties {
    MassProperties::with_inertia(
        1.0,
        DMat3::from_diagonal(DVec3::splat(10.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh2_mass() -> MassProperties {
    MassProperties::with_inertia(
        2.0,
        DMat3::from_diagonal(DVec3::splat(20.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh3_mass() -> MassProperties {
    MassProperties::with_inertia(
        3.0,
        DMat3::from_diagonal(DVec3::splat(30.0)),
        DVec3::new(5.0, 0.0, 0.0),
    )
}

fn veh1_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(-5.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 1.0, 0.0),
    }
}

fn veh1_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::from_array([1.0, 0.0, 0.0, 0.0]),
        ang_vel_body: DVec3::ZERO,
    }
}

fn veh2_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(5.0, 10.0, 0.0),
        velocity: DVec3::ZERO,
    }
}

fn veh2_rot() -> RotationalState {
    let q = JeodQuat::left_quat_from_eigen_rotation(-2.0, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.0, 0.0, 0.2),
    }
}

fn veh3_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(0.063, 13.787, -25.0),
        velocity: DVec3::new(0.0, 0.0, 1.0),
    }
}

fn veh3_rot() -> RotationalState {
    let q = JeodQuat::left_quat_from_eigen_rotation(-15.8, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::ZERO,
    }
}

/// JEOD's `BodyAttachAligned veh1.attach_to_2` link geometry.
fn link_offset_and_rotation() -> (DVec3, DMat3) {
    (DVec3::new(-10.0, 0.0, 0.0), DMat3::IDENTITY)
}

fn body_index_veh1() -> usize {
    0
}
fn body_index_veh2() -> usize {
    1
}

fn build_attach_detach(_init: &InitialConditions) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut b = SimulationBuilder::new(time, DT);
    b.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&veh1_trans()),
        rot: Some(super::typed_helpers::rot_typed(&veh1_rot())),
        mass: Some(super::typed_helpers::mass_typed(&veh1_mass())),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    b.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&veh2_trans()),
        rot: Some(super::typed_helpers::rot_typed(&veh2_rot())),
        mass: Some(super::typed_helpers::mass_typed(&veh2_mass())),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    b.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&veh3_trans()),
        rot: Some(super::typed_helpers::rot_typed(&veh3_rot())),
        mass: Some(super::typed_helpers::mass_typed(&veh3_mass())),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    b.register_in_mass_tree(0, "veh1");
    b.register_in_mass_tree(1, "veh2");
    b.register_in_mass_tree(2, "veh3");
    b
}

/// pre_step: fire `attach(veh1 → veh2)` once when the upcoming step
/// crosses [`ATTACH_TIME`].
///
/// The closure captures a `fired` flag so the attach runs exactly
/// once. The `time > ATTACH_TIME` test fires on the first record whose
/// step lands strictly past `ATTACH_TIME`, mirroring the hand-rolled
/// test's "fire after the step that lands at `t = ATTACH_TIME`"
/// timing — this is the JEOD-faithful `trick.add_read(t, ...)`
/// schedule (the integrator runs from `t-DT → t` with the bodies
/// still separate, then the read job fires at the start of the next
/// dispatch cycle for time `t + DT`).
fn attach_detach_pre_step(_init: &InitialConditions) -> PreStepClosure {
    let mut fired = false;
    Box::new(move |sim: &mut dyn SimContext, time: f64| {
        if !fired && time > ATTACH_TIME {
            let (offset, t_pc) = link_offset_and_rotation();
            sim.attach(body_index_veh1(), body_index_veh2(), offset, t_pc);
            sim.mark_kinematic_only(body_index_veh1());
            fired = true;
        }
    })
}

/// SIM_verif_attach_detach RUN_simple_attach_detach attach-window
/// parity recipe.
pub fn attach_detach_trajectory() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_attach_detach_trajectory_simple",
        scenario: build_attach_detach,
        // TimesOnly: parity reads cadence only; the initial conditions
        // are hardcoded above and the per-record state comparison
        // happens body-by-body between the two runtimes.
        reference: CsvReference::TimesOnly(
            "kinematic_propagation_simple_kinematic_propagation_state.csv",
        ),
        // Stop strictly before the detach event — mirrors the
        // hand-rolled test scope (see file-level docstring "What is
        // and isn't covered").
        duration: Time::new::<second>(DETACH_TIME - DT),
        tolerances: zero_tolerances(),
        extras: None,
        pre_step: Some(attach_detach_pre_step),
    }
}

fn zero_tolerances() -> Tolerances {
    // Parity tests don't compare against JEOD; tolerances aren't
    // exercised. All-zero opts out of every metric group via the
    // documented "all-zero skips" rule in `Tolerances`.
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}
