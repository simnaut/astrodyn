//! `VerificationCase` constructor for the SIM_simple_attach_detach
//! trajectory parity family (#395 sub-task A).
//!
//! Three free-flying 6-DOF rigid bodies (no force / no torque). The
//! `pre_step` factory schedules a runtime attach of veh1 → veh2 at
//! `ATTACH_TIME = 10 s` and the matching `mark_kinematic_only` one
//! tick later, then stops strictly before `DETACH_TIME = 20 s` (the
//! parity comparison's #308 fence — see the hand-rolled
//! `bevy_parity_attach_detach_trajectory.rs` for the dual-write
//! reasoning).
//!
//! Mirrors the hand-rolled
//! `bevy_parity_attach_detach_trajectory.rs::bevy_parity_attach_detach_trajectory_simple`
//! exactly.

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

const ATTACH_PARITY_CSV: &str = "attach_detach_trajectory_parity_times.csv";
const ATTACH_DT: f64 = 0.1;
const ATTACH_TIME: f64 = 10.0;

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

/// JEOD's `BodyAttachAligned veh1.attach_to_2`: child struct origin
/// at (-10, 0, 0) in parent struct frame, identity link rotation.
fn link_offset() -> DVec3 {
    DVec3::new(-10.0, 0.0, 0.0)
}

fn link_t_parent_child() -> DMat3 {
    DMat3::IDENTITY
}

/// Build a 3-body free-flying scenario with all three bodies
/// registered in the mass tree but unattached. The `pre_step` factory
/// schedules veh1.attach_to_2 + mark_kinematic_only at the right
/// records.
fn build_attach_detach_trajectory(_init: &InitialConditions) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, ATTACH_DT);
    let v1 = sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&veh1_trans()),
        rot: Some(super::typed_helpers::rot_typed(&veh1_rot())),
        mass: Some(super::typed_helpers::mass_typed(&veh1_mass())),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let v2 = sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&veh2_trans()),
        rot: Some(super::typed_helpers::rot_typed(&veh2_rot())),
        mass: Some(super::typed_helpers::mass_typed(&veh2_mass())),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    let v3 = sb.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&veh3_trans()),
        rot: Some(super::typed_helpers::rot_typed(&veh3_rot())),
        mass: Some(super::typed_helpers::mass_typed(&veh3_mass())),
        gravity_controls: GravityControls { controls: vec![] },
        integrator: IntegratorType::Rk4,
        ..Default::default()
    });
    sb.register_in_mass_tree(v1, "veh1");
    sb.register_in_mass_tree(v2, "veh2");
    sb.register_in_mass_tree(v3, "veh3");
    sb
}

/// `pre_step` factory: fires `attach(child=v1=0, parent=v2=1, ...)`
/// at the record advancing to `t = ATTACH_TIME` and
/// `mark_kinematic_only(0)` one record later. The duration field on
/// the `VerificationCase` clamps the propagation strictly before
/// `t = DETACH_TIME` so the comparison stops before the dual-write
/// fence the hand-rolled test documents.
fn attach_detach_pre_step(_init: &InitialConditions) -> PreStepClosure {
    Box::new(move |sim, time_s: f64| {
        let half_dt = 0.5 * ATTACH_DT;
        if (time_s - ATTACH_TIME).abs() < half_dt {
            sim.attach(0, 1, link_offset(), link_t_parent_child());
        } else if (time_s - (ATTACH_TIME + ATTACH_DT)).abs() < half_dt {
            sim.mark_kinematic_only(0);
        }
    })
}

/// SIM_simple_attach_detach 3-body trajectory parity scenario.
/// Mirrors the hand-rolled `bevy_parity_attach_detach_trajectory.rs`
/// test exactly. Stops strictly before `DETACH_TIME = 20 s` per the
/// hand-rolled #308 fence.
pub fn simple() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_attach_detach_trajectory_simple",
        scenario: build_attach_detach_trajectory,
        reference: CsvReference::TimesOnly(ATTACH_PARITY_CSV),
        // Stop one record before `t = DETACH_TIME = 20 s`.
        duration: Time::new::<second>(20.0 - ATTACH_DT),
        tolerances: Tolerances {
            position_m: [0.0; 3],
            velocity_m_s: [0.0; 3],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(attach_detach_pre_step),
    }
}
