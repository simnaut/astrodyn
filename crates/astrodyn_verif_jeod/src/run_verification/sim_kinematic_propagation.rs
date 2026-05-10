//! `VerificationCase` constructor for the kinematic-propagation parity
//! family (#395 sub-task A).
//!
//! Builds a parent + kinematic-child topology with no force / no torque,
//! then drives a runtime attach via `pre_step` to install the chain.
//! Mirrors the hand-rolled
//! `bevy_parity_kinematic_propagation.rs::bevy_parity_kinematic_propagation_simple_chain`
//! exactly, including the tick-1 / tick-2 separation between the
//! `AttachEvent` and the kinematic-edge installation.
//!
//! ## Tick-1 / steady-state separation
//!
//! The parity trait's per-record `pre_step` provides the natural
//! sequencing the hand-rolled test painstakingly orchestrates by hand:
//!
//! - **`pre_step` at record 1** (advancing to `t = DT`): fires the
//!   runtime attach. On the runner this is a synchronous
//!   `Simulation::attach` (combine kernel + integrator reset). On
//!   Bevy this writes an `AttachEvent` that `staging_system` drains
//!   at the top of tick 1 (before integration). Both runtimes see
//!   the merged composite-body state when integration runs.
//! - **`pre_step` at record 2** (advancing to `t = 2·DT`): fires
//!   `mark_kinematic_only` on the child. On the runner this sets the
//!   `kinematic_only` flag synchronously. On Bevy our
//!   `BevySimContext::mark_kinematic_only` inserts both
//!   `MassChildOf` (the ECS edge) and `KinematicChildC` (integrator
//!   gating marker) — `composite_mass_system` on tick 2 sees the new
//!   edge and writes the combined mass into the parent's
//!   `MassPropertiesC`, while `integration_system`'s
//!   `Without<KinematicChildC>` filter keeps the child off the
//!   integrator. From tick 2 onwards both runtimes derive the child
//!   from the parent via the same
//!   `propagate_state_via_storage` kernel.
//!
//! This split avoids the documented
//! `composite_mass_system → staging_system` race that would mis-feed
//! `combine_states_at_attach` on the attach tick if `MassChildOf` were
//! installed pre-attach.

use crate::verification::{
    CsvReference, InitialConditions, PreStepClosure, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, GravityControls, IntegratorType, JeodQuat, MassProperties,
    RotationalState, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use glam::{DMat3, DVec3};
use uom::si::f64::Time;
use uom::si::time::second;

/// Sim-time DT shared with the hand-rolled parity test.
const KINEMATIC_DT: f64 = 0.1;

/// Number of `KINEMATIC_DT` ticks to drive — matches the
/// `NUM_STEPS = 30` constant in the pre-#395 hand-rolled parity test.
const KINEMATIC_NUM_STEPS: usize = 30;

fn parent_mass() -> MassProperties {
    let inertia = DMat3::from_diagonal(DVec3::splat(20.0));
    MassProperties::with_inertia(2.0, inertia, DVec3::new(5.0, 0.0, 0.0))
}

fn child_mass() -> MassProperties {
    let inertia = DMat3::from_diagonal(DVec3::splat(10.0));
    MassProperties::with_inertia(1.0, inertia, DVec3::new(5.0, 0.0, 0.0))
}

fn parent_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(5.0, 10.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, 0.5),
    }
}

fn parent_rot() -> RotationalState {
    let q = JeodQuat::left_quat_from_eigen_rotation(-0.5, DVec3::Z);
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.0, 0.0, 0.2),
    }
}

/// Child trans: equal to parent's at t=0 (soft merge — no relative
/// motion at attach).
fn child_initial_trans() -> TranslationalState {
    parent_trans()
}

fn child_initial_rot() -> RotationalState {
    parent_rot()
}

fn link_offset() -> DVec3 {
    DVec3::new(-10.0, 0.0, 0.0)
}

fn link_t_parent_child() -> DMat3 {
    DMat3::IDENTITY
}

/// Build a 2-body parent + child scenario with no gravity / no force,
/// both registered in the mass tree but **not** pre-attached. The
/// `pre_step` factory schedules the runtime attach + kinematic-only
/// transition.
fn build_kinematic_propagation(_init: &InitialConditions) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, KINEMATIC_DT);
    let parent_idx = sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&parent_trans()),
        rot: Some(super::typed_helpers::rot_typed(&parent_rot())),
        mass: Some(super::typed_helpers::mass_typed(&parent_mass())),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let child_idx = sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&child_initial_trans()),
        rot: Some(super::typed_helpers::rot_typed(&child_initial_rot())),
        mass: Some(super::typed_helpers::mass_typed(&child_mass())),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    sb.register_in_mass_tree(parent_idx, "parent");
    sb.register_in_mass_tree(child_idx, "child");
    sb
}

/// `pre_step` factory: fires `attach(child=1, parent=0, ...)` at
/// record 1 (`t = DT`) and `mark_kinematic_only(1)` at record 2
/// (`t = 2·DT`). Subsequent records are no-ops.
fn kinematic_propagation_pre_step(_init: &InitialConditions) -> PreStepClosure {
    Box::new(move |sim, time_s: f64| {
        // Record 1 lands at `t = DT`; record 2 at `t = 2·DT`. Use a
        // half-DT epsilon so floating-point time accumulation can't
        // miss the trigger.
        let half_dt = 0.5 * KINEMATIC_DT;
        if (time_s - KINEMATIC_DT).abs() < half_dt {
            sim.attach(1, 0, link_offset(), link_t_parent_child());
        } else if (time_s - 2.0 * KINEMATIC_DT).abs() < half_dt {
            sim.mark_kinematic_only(1);
        }
    })
}

/// Parent + kinematic-child propagation parity scenario. Mirrors the
/// hand-rolled `bevy_parity_kinematic_propagation.rs` test exactly.
pub fn simple_chain() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_kinematic_propagation_simple_chain",
        scenario: build_kinematic_propagation,
        reference: CsvReference::SyntheticTimes {
            dt: KINEMATIC_DT,
            num_steps: KINEMATIC_NUM_STEPS,
        },
        duration: Time::new::<second>(0.0),
        tolerances: Tolerances {
            position_m: [0.0; 3],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(kinematic_propagation_pre_step),
    }
}
